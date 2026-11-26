# RedEye technology plan

Target: a Quest 3 app, fully on-device with no network use, that shows the
world graded red at display rate and updates perception at about 24 Hz.
Stack: [Blade](https://github.com/kvark/blade) for GPU and OpenXR,
[Meganeura](https://github.com/kvark/meganeura) for inference and training.
[DinoVision](https://github.com/kvark/dinovision) is the reference
integration of both on a Quest.

Status of facts: anything marked (u) is unverified and should be measured or
checked before a decision depends on it.

## Platform: Quest 3

- **SoC**: Snapdragon XR2 Gen 2, Adreno 740, Vulkan. Meganeura's Quest 3S
  qualification found packed int-dot and 64-lane subgroups, but neither
  native shader f16 nor cooperative matrices, so today everything runs on
  scalar f32 kernels with f16 as storage only. Re-check on a Quest 3 with
  current Horizon OS (Meganeura work, item 1).
- **Passthrough Camera API** (Horizon OS, API 34, NDK r27): the two forward
  RGB cameras appear as ordinary Camera2 devices, 1280×960 YUV420 at up to
  60 Hz, with 20–40 ms capture latency (u). `horizonos.permission.HEADSET_CAMERA`
  must be granted at runtime. Meta's vendor tags cannot be resolved, because
  `ACameraManager_getTagFromName` is not exported (Horizon OS v205), so the
  cameras are identified by capability. Whether the standard
  `LENS_INTRINSIC_CALIBRATION` / `LENS_POSE_*` tags are populated is open;
  we need them to map image detections into the world.
- **`XR_FB_passthrough`**: the compositor's own passthrough, stereo-correct at
  display rate. A passthrough layer accepts a style with a
  **256-entry mono→RGBA colour map** (`XrPassthroughColorMapMonoToRgbaFB`)
  and an **edge colour**. That is exactly M1, at zero app GPU cost.
  The app cannot run its own shaders on system passthrough.
- **`XR_META_environment_depth`**: system-computed metric depth, low
  resolution, edge bleeding, unreliable under about 0.2 m. Free for the app.
  It is enough for body-measurement scale and for world-space target positions.
- **Hand tracking** for input (pinch, punch-to-damage). No controllers are
  required.

## Architecture

```text
             compositor (display rate, 72–120 Hz)
  ┌──────────────────────────────────────────────────────────┐
  │ passthrough layer, colour map = Grade::mono_to_rgba_lut  │  world (M1)
  │ projection layer, alpha-blended, Blade-rendered          │  HUD
  └──────────────────────────────────────────────────────────┘
                     ▲ every frame: HUD from latest Observation,
                     │ targets reprojected with current head pose
  ┌──────────────────┴───────────────────┐
  │ render thread (Blade)                │
  └──────────────────▲───────────────────┘
                     │ Observation (tracks, poses, …) + capture pose
  ┌──────────────────┴───────────────────┐   ~24 Hz, never blocks render
  │ perception worker                    │
  │  camera Y/UV → GPU preprocess        │
  │  → Meganeura sessions (scheduled)    │
  │  → decode / NMS / tracker (CPU)      │
  └──────────────────▲───────────────────┘
                     │ YUV420 + timestamp
               Camera2 passthrough camera (left)
```

- **One GPU context** for rendering and inference
  (`SessionConfig::gpu`), so the crate pins Meganeura's Blade revision.
  Sessions are `Send` but not `Sync` and are built on the worker thread.
- **The display path never waits on a network.** The HUD draws the latest
  observation, extrapolated with track velocity and reprojected with the
  live head pose, because detections are 60–100 ms old by the time they are
  drawn.
- **Fallback world view**: [`render::GradePass`](../src/render.rs) draws the
  camera Y plane through the grade ourselves. That is milestone 1, because Blade
  can do it today. It is worse in every other respect: mono, a narrower FOV
  than passthrough, latency, and render submissions that compete with
  inference. DinoVision's live data shows the result: about 12 Hz render
  submission next to a 6 Hz model.

## Stack: findings and required work

### Blade (rev `bba0edc`)

- XR support: one stereo projection layer per frame
  (`present_xr` → `frame_stream.end`). The environment blend mode is the
  first one enumerated (OPAQUE on Quest), and the layer has no alpha flags.
  `context.xr_session()` exposes the session.
- **Needed upstream** (small): let the app submit additional composition
  layers ahead of the projection layer (a passthrough layer), and set
  `BLEND_TEXTURE_SOURCE_ALPHA` / unpremultiplied flags on the projection
  layer so the HUD composites over passthrough.
- `openxr` 0.21 (Blade's version) wraps `Passthrough` and
  `PassthroughLayer`, but not the style or colour-map calls; use the raw
  function pointers. In 0.19, `Passthrough::start` called `pause`, so check
  0.21 or call `xrPassthroughStartFB` directly.

### Meganeura (rev `aba04df`, 0.3)

Meganeura is our crate. What follows is its state today. The gaps are
work items for us, not constraints on RedEye; they are collected under
[Meganeura work for RedEye](#meganeura-work-for-redeye). DinoVision proved
the stack end to end and was not optimised for the Quest.

- **The ONNX importer does not yet cover vision models.** It lacks Resize,
  grouped/depthwise Conv, ConvTranspose, AveragePool, Pad, PRelu/HardSwish,
  TopK, NMS, GridSample and comparison ops. Shapes must be static, and
  `Slice` must have unit steps. None of YOLO*, YOLOX, RT-DETR/D-FINE,
  BlazeFace or the DPT depth head import as-is.
- **The hand-built graph API covers what we need for CNNs and ViTs**:
  `conv2d`, `conv2d_dw` (depthwise), `max_pool_2d`, `upsample_2x`
  (nearest), `concat`/`split`, `group_norm`/`layer_norm`, attention, `clamp`
  (for HardSwish), `argmax_inner`. Helpers provide `upsample_nearest`,
  `conv_transpose_2d` (via dilation, wasteful), `crop_2d` and `pixel_shuffle_w`.
  **Missing**: bilinear resize, average pooling, TopK, NMS, grid sampling.
  Models are therefore written in Rust as graphs (as DinoVision does for
  DINOv3), with weights loaded from safetensors and BatchNorm folded at load.
- **Built-in models** (feature `models`): ResNet-18/50, EfficientNetV2-S
  trunk, SigLIP ViT inside SmolVLM2, Whisper, SmolLM2.
- **Runtime**: `Mode::Inference`; `input_buffer` lets our own compute pass
  write preprocessed input on the shared context (camera AHardwareBuffers
  cannot be imported, so one copy is unavoidable); `step` is async and
  `wait` blocks; `set_submission_chunks` interleaves with rendering (two
  was best in DinoVision); several sessions share one queue, and
  `share_parameter_from` deduplicates weights.
- **Training**: autodiff (`stop_gradient` freezes a backbone; note that
  `Conv2dDw` has no backward pass yet), Adam/SGD/LaProp, cross-entropy, MSE,
  L1 and BCE losses, and safetensors checkpoints. Detection losses (focal,
  GIoU) are composable; Hungarian matching would run on the CPU. Fine-tuning
  heads on a frozen backbone on a desktop GPU, then deploying through the
  same compiler, is proven by DinoVision.
- **Performance evidence on Quest** is a proof-of-concept baseline, not a
  ceiling. DinoVision's 3-layer DINOv3-S/16 at 224 plus a conv decoder took
  about 97 ms per synchronous step (meganeura 0.2, no Adreno tuning).
  Reference point: ncnn Vulkan fp16 on Adreno 740 runs ResNet-18 in about
  23 ms, roughly 0.1 GMAC/ms. **The first engineering task is a calibration
  benchmark** at the current revision (ms/GMAC for a conv net and for ViT-S,
  at candidate resolutions), with GPU timestamps per kernel archetype, so that
  optimisation targets the right kernels.

### Meganeura work for RedEye

Ordered by what unblocks RedEye soonest. Each item lands upstream with
Meganeura's own tests (oracle cases on lavapipe; Quest qualification).

1. **Quest 3 device facts.** Qualification so far ran on a Quest 3S, whose
   Adreno 740 driver exposes packed int-dot and 64-lane subgroups, but
   neither `shaderFloat16` nor cooperative matrices. Re-query on a Quest 3
   with current Horizon OS: f16 arithmetic, `VK_KHR_cooperative_matrix`,
   subgroup size control, integer dot product. The answers decide which of
   the items below pay off. Coopmat and f16 kernels are worth enabling
   wherever the driver exposes them.
2. **Calibration benchmark on device**: a native (non-XR) APK as in
   DinoVision's `android/`, timing conv, depthwise, matmul and attention
   archetypes and whole models, with thermal state recorded.
3. **Ops for the detector and the heads**: bilinear Resize, AveragePool,
   Pad, PRelu/LeakyRelu/HardSwish, TopK, GatherElements, GridSample (for
   deformable attention in D-FINE/DEIM), a native ConvTranspose (instead of
   dilating with zeros), and a grouped Conv generalising `conv2d_dw`.
4. **ONNX importer coverage for vision**: the ops above plus Slice with
   steps, Clip, Min/Max, Where, comparisons, and non-last-axis Softmax. That
   lets candidate models be evaluated from their official exports
   (`examples/onnx_compare.rs`) before anything is hand-written. A model
   gets a hand-built graph only when fusion or layout control needs one.
5. **Adreno kernels**:
   - int8 convolution and matmul on packed `dot4I8Packed`. It exists today
     for LLM GEMV only; extending it to conv is likely the largest win for
     CNNs, given no f16 arithmetic.
   - Subgroup-wide reductions for softmax and norms.
   - f16 *storage* for conv weights, not only matmul and embeddings.
   - Image or texture-backed weight reads, which ncnn and MNN use on Adreno.
   - Fusion of conv, BatchNorm and activation, and of our preprocessing into
     the first conv.
   - Fewer dispatches and barriers per graph.
6. **Training gaps for fine-tuning**: a `Conv2dDw` backward pass (pose and
   face nets are depthwise-heavy), and detection losses (focal,
   GIoU/L1 box regression) with CPU-side Hungarian matching.

## Model plan

Selection rules: permissive licence (Apache/MIT/BSD); quality per FLOP;
a small enough budget to stagger. Missing Meganeura ops are an upstream cost
to weigh, not a veto. Where a better model needs new ops, a no-new-ops model
is used to bring the pipeline up first. Excluded on licence grounds: Ultralytics YOLOv8/11/26
and FastSAM (AGPL); YOLO-World (GPL); DEIMv2, SCRFD weights, MobileCLIP,
DepthPro and CC-BY-NC Depth Anything checkpoints (non-commercial).
**DINOv3** has a custom licence: commercial use is allowed, but
military/weapons use is prohibited and the licence must ship with the app.
DINOv2 is Apache-2.0.

| Capability | First choice | Why / ops | Later / fallback |
|---|---|---|---|
| Detection (M3) | **D-FINE-N / DEIM-D-FINE-N** (3.8M, 7 GFLOPs@640, 42.8–43.0 AP, Apache) at about 512×384 | NMS-free; needs GridSample, TopK and GatherElements in Meganeura | Bring-up: **YOLOX-Nano** (0.9M, 1.08 GFLOPs@416, 25.8 AP, Apache), which needs no new ops: conv, SiLU, nearest upsample, concat, CPU NMS. Also YOLOX-Tiny; RT-DETRv2-S with discrete sampling |
| Tracking (M3) | ByteTrack/OC-SORT-style, CPU, world space | [`perception::track`](../src/perception/track.rs) is the IoU core | appearance from backbone ROI features or OSNet x0.25 |
| Depth (M3–M5) | **Quest Depth API** | metric and free | Depth Anything V2 Small (Apache) at low resolution, scale-aligned to the Depth API |
| Pose (M5) | **RTMPose-t/s Halpe26** on person crops (Apache) | CSPNeXt (depthwise) + GAU attention + SimCC argmax; 26 keypoints including head-top and feet | MoveNet Lightning; BlazePose (adds a person mask) |
| Face (M4) | **BlazeFace** on head crops (Apache) | depthwise conv; PReLU composed from ReLU and per-channel multiply; CPU decode | YuNet (MIT) |
| Masks (M6) | BlazePose mask for the locked target | — | RF-DETR-Seg-Nano (Apache), EoMT-S on a shared ViT |
| Identify (M6) | **SigLIP2-B/16** or **PE-Core-T16** image tower on crops, on demand, with precomputed label embeddings | plain ViT; SigLIP already exists in Meganeura | Florence-2-base on demand |
| OCR (M9) | PP-OCRv6-tiny det + rec (Apache; recent, verify) | DB head (CPU contours) + CTC | PP-OCRv5 mobile |
| Speech (M8) | Silero VAD + Moonshine-tiny (MIT) | CPU, or GPU idle slots | Whisper tiny/base (in Meganeura) |
| Scene text (extra) | SmolVLM2-256M (Apache, in Meganeura) | on demand, seconds | SmolVLM2-500M |

**Shared backbone, if the numbers allow it.** One ViT-S/16 (DINOv2-S for
licence cleanliness, or DINOv3-S for features) at 320×240 (300 tokens,
about 7.6 GMACs) feeding light heads: DETR-style boxes, a mask dot-product
head, DPT-lite depth distilled from Depth Anything V2, and keypoints. The
DINOv2 repo publishes frozen-backbone depth and segmentation heads for
ViT-S/14. This is the main Meganeura training job and the natural
continuation of DinoVision. **Go/no-go: ViT-S/16 at 320×240 under about
15 ms on the Quest.**

## Frame budget at 24 Hz

41.7 ms per perception frame, of which roughly 20 ms of GPU time is
available to inference: the compositor and our HUD need the rest. Estimates
assume the ncnn-like rate of 0.1 GMAC/ms until the calibration benchmark
replaces it.

| Slot | Model | Cadence | Est. GPU ms per frame |
|---|---|---|---|
| Preprocess | YUV→RGB, resize, normalise (compute shader) | every frame | ~1 |
| Detection | D-FINE-N @512×384 (YOLOX-Nano @416×320 during bring-up) | every frame | ~5–17 before Adreno tuning |
| Pose | RTMPose-t, up to 2 crops | every 2nd | ~2 |
| Face | BlazeFace on head crops | every 4th | <0.5 |
| Tracking, NMS, decode | CPU | every frame | 0 (CPU <1 ms) |
| Depth | Depth API | every frame | 0 |
| Identify / OCR / VLM | on demand | bursts | displaces pose and face while running |

## Risks and open questions

1. **Meganeura throughput on Adreno** is the dominant risk, and the one we
   control. Nothing on Quest has been timed at the current revision or
   tuned for it. The calibration benchmark comes first so kernel work
   (int8 dot, subgroups, layout, fusion) is aimed by data. Staggering and
   resolution are the knobs until it lands.
2. **Mapping camera to world.** The camera's FOV and pose differ from the
   passthrough view. We need intrinsics and extrinsics (standard Camera2
   tags, or a one-off calibration) and a head pose at capture time (camera
   timestamp → `xrLocateSpace`).
3. **Colour-map space.** It is not documented whether the passthrough colour
   map is linear or sRGB-encoded. Compare it on device against `GradePass`.
4. **Latency.** Capture (20–40 ms) plus inference plus display means world-space
   tracks with velocity prediction, not image-space boxes.
5. **Thermals.** Sustained GPU load will throttle. Log the thermal state
   alongside timings, as DinoVision does.

## Milestones

- **M0 (done)**: project skeleton; grade with presets, shared by WGSL and the
  LUT; headless preview; tracker and text effects; XR crate type-checks.
- **M1**: on device, the left camera graded through `GradePass` to both eyes.
  Plus the calibration benchmark and Quest 3 device facts (Meganeura items
  1–2).
- **M2**: the Blade change for passthrough and alpha layers; the world view via
  the colour map; HUD text (SDF atlas, typewriter, scroll), data scroll, boot,
  damage, compass. In parallel in Meganeura: vision ops and ONNX coverage
  (items 3–4).
- **M3**: YOLOX-Nano bring-up (GPU preprocessing, CPU NMS, world-space
  tracking with depth, target lock), then D-FINE-N once its ops land. Adreno
  kernel work (item 5) is driven by the benchmark.
- **M4**: pose and depth to size assessment and MATCH; face inset; object ID
  via crop classification.
- **M5**: response menu (VAD, speech-to-text), OCR, threat assessment.
