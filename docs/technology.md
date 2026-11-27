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

- **SoC**: Snapdragon XR2 Gen 2, Adreno 740, Vulkan 1.3. Queried on a
  Quest 3S with Horizon OS 210 (driver `0x80345009`, November 2026):
  `shaderFloat16` and `shaderInt8` (earlier drivers lacked f16
  arithmetic), 16- and 8-bit storage, integer dot product, subgroups of 64
  (64–128 with size control, full subgroups), f32 atomics. **No**
  cooperative matrices, and **no** `VK_EXT_calibrated_timestamps`, which
  Blade needed for any GPU timing (Blade work, item 6). 32 KiB shared
  memory, 1024 invocations per workgroup. The Quest 3 has the same SoC.
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

Blade is also ours. Its XR path today submits one stereo projection layer
per frame (`present_xr` → `frame_stream.end`). The environment blend mode is
the first one enumerated (OPAQUE on Quest), and the layer has no alpha flags.
`context.xr_session()` exposes the session.

### Blade work for RedEye

1. **Composition layers**: let the app submit layers beneath the projection
   layer, starting with an `XR_FB_passthrough` layer, and set
   `BLEND_TEXTURE_SOURCE_ALPHA` / unpremultiplied flags on the projection
   layer so the HUD composites over passthrough.
2. **Passthrough as a feature**: create, start and pause passthrough with the
   session lifecycle, and set the style: the mono→RGBA colour map from
   `Grade::mono_to_rgba_lut`, opacity, and edge colour. `openxr` 0.21 wraps
   `Passthrough`/`PassthroughLayer` but not styles, so this goes through raw
   function pointers. Its 0.19 `Passthrough::start` called `pause`; check 0.21.
3. **Environment depth**: `XR_META_environment_depth` provider and swapchain,
   exposed as a texture the HUD and perception can sample, with its
   per-frame view and projection.
4. **Timing and poses**: locate the head at an arbitrary past time (a camera
   timestamp), not only at the predicted display time, so detections can be
   placed in the world where they were seen.
5. **Example**: an `examples-android` passthrough sample, so this is tested
   in Blade itself and not only through RedEye.
6. **GPU timing without calibrated timestamps.** Adreno lacks
   `VK_EXT_calibrated_timestamps`, and Blade turned timing off without it,
   so Meganeura's profiler had nothing to read on the Quest. Branch
   `timing-without-calibration` anchors each submission's first timestamp
   to the CPU time at submit: durations stay exact, start times are
   approximate. The Quest profile in Frame budget was taken with it.

### Meganeura (rev `b25f763`, 0.3)

Meganeura is our crate. What follows is its state today. The gaps are
work items for us, not constraints on RedEye; they are collected under
[Meganeura work for RedEye](#meganeura-work-for-redeye). DinoVision proved
the stack end to end and was not optimised for the Quest.

- **The ONNX importer runs both our detectors.** Grouped and depthwise
  Conv, ConvTranspose, Resize, TopK and GridSample (`3385527`), strided
  `Slice` (`a964c5b`), and Pad, Clip, Tile, GatherElements, TopK up to
  k = 512 and import-time folding of constant subgraphs (`33d66fb`,
  `b25f763`). Shapes must be static, so `tools/onnx_reference.py` fixes the
  input shape. Checked with `examples/onnx_check.rs` and the tests against
  onnxruntime:
  - **YOLOX-Nano and YOLOX-Tiny** (official exports, whole) match to
    2.1e-5 and 1.9e-5.
  - **D-FINE-N** (`onnx-community/dfine-nano-coco-ONNX`, Apache-2.0) matches
    to 5.5e-5 (logits) and 9.2e-6 (boxes) at 640×480, and 2.0e-5 and
    3.4e-6 at 640×640. Its graph is 6053 nodes and **1967 dispatches**,
    against 370 for the imported YOLOX-Nano export (254 for our hand-built
    graph). Many come from TopK, lowered as one pass per
    selected element, but on the Quest its time is in the convolutions
    (see Frame budget).
  - This export still goes through onnx-simplifier (`--simplify`). It
    builds D-FINE's box-weighting table with `Pow(x, 1/15)` from two scalar
    parameters (`up`, `reg_scale`) through two `Abs` ops, and the importer's
    folder has no `Abs` arm, so the table never becomes a constant and the
    `Pow` exponent is unsupported. (The folder does read initializers; an
    earlier note here blamed them, wrongly.) The simplifier's Pad-into-Conv
    fusion is turned off, because the asymmetric convolution padding it
    produces (`[0, 0, 1, 1]`) does not import.
  - **Moonshine-tiny** (`onnx-community/moonshine-tiny-ONNX`, MIT), the
    speech model for voice commands. After `--simplify`, two gaps remain:
    `InstanceNormalization` (one, in the encoder stem) and a `Gather` of the
    token embedding at runtime indices (the importer makes every graph input
    F32). With those two patched by hand, the encoder (2 s or 4 s of
    16 kHz audio, 310 dispatches) matches onnxruntime to 7e-6, and the
    decoder (449 dispatches per step) to 2e-5.
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

1. **Device facts: done** (Platform, above). The current driver has f16
   arithmetic but no cooperative matrices, so f16 compute in the conv and
   matmul kernels is the lever, not coopmat.
2. **Calibration benchmark on device**: a native (non-XR) APK as in
   DinoVision's `android/`, timing conv, depthwise, matmul and attention
   archetypes and whole models, with thermal state recorded.
3. **D-FINE's cost and import path** (all its ops now import):
   - **Convolution throughput on Adreno.** Measured on the Quest (Frame
     budget), D-FINE-N takes 123 ms of GPU time, 61% of it in 81
     `Conv2dGemm*` dispatches and 18% in matmuls. That is a few percent of
     the Adreno 740's f32 peak. Tiling for 64-wide subgroups and f16
     arithmetic come first.
   - **A real TopK kernel.** D-FINE selects 300 of 1500 anchors at
     640×480 (2000 at 640²); the current lowering takes one max-and-mask
     pass per element. A single-dispatch select (bitonic or radix) cuts
     the dispatch count, but all 841 data-movement dispatches together
     take 10 ms on the Quest, so it comes after the convolutions.
   - Fold `Abs` (D-FINE's box-weighting table) and `Transpose`
     (Moonshine's rotary tables), so raw exports import without
     onnx-simplifier.
   - Asymmetric convolution padding, which simplifiers and TensorFlow
     exports produce.
4. **Voice commands (Moonshine-tiny)**:
   - `InstanceNormalization`, or `GroupNorm` over 3D input. Moonshine's
     stem normalises one 287k-element row, so a single-row reduction kernel
     must not be slow.
   - Integer graph inputs (`graph.input_u32`), so a token-id input can feed
     `Gather` on the embedding.
   - Without onnx-simplifier, the decoder also needs `Trilu`, plus
     `Slice` with runtime bounds, for its causal mask and rotary tables.
5. **Segmentation and depth for the wireframe** (TASKS.md items 4–5;
   scripts in `tools/experiments/segdepth/`):
   - RF-DETR-Seg-Nano: runtime `Sin` and `Cos` (the decoder's sine
     embedding of reference boxes), `Where` on runtime data, and `Einsum`
     in the form `bchw,bnc->bnhw` (the mask head; lowering it to MatMul
     suffices).
   - EdgeTAM: `ReduceMean` over axes [2, 3] (that is `GlobalAveragePool`),
     plus, for runtime box prompts, `ScatterND`, runtime `Sin` and `Cos`,
     and an INT64 input.
   - Depth Anything 3 Small: `ScatterND` writing a constant row into a
     runtime tensor.
   - Depth Anything V2 Small already imports with no gaps.
6. **More importer coverage**: AveragePool, PRelu/HardSwish, Min/Max,
   non-last-axis Softmax. Candidate models are evaluated from their
   exports with `tools/onnx_reference.py` and `examples/onnx_check.rs`
   before anything is hand-written.
7. **Adreno kernels**:
   - int8 convolution and matmul on packed `dot4I8Packed`. It exists today
     for LLM GEMV only; extending it to conv is likely the largest win for
     CNNs, given no f16 arithmetic.
   - Subgroup-wide reductions for softmax and norms.
   - f16 *storage* for conv weights, not only matmul and embeddings.
   - Image or texture-backed weight reads, which ncnn and MNN use on Adreno.
   - Fusion of conv, BatchNorm and activation, and of our preprocessing into
     the first conv.
   - Fewer dispatches and barriers per graph.
8. **Training gaps for fine-tuning**: a `Conv2dDw` backward pass (pose and
   face nets are depthwise-heavy), and detection losses (focal,
   GIoU/L1 box regression) with CPU-side Hungarian matching.

## Model plan

**One detector is the core.** A model that returns a list of boxes, each
with a class name and a score, drives nearly everything. Add tracking and
the platform's depth and these mechanics work:

| Mechanic | From the detector |
|---|---|
| M3 target lock, M7 threat assessment | `person` boxes → tracks → world positions with depth |
| M4 acquisition inset | the head region of a person box, cropped from the camera frame |
| M5 size assessment | box top and bottom plus depth give height; rough, which suits the gag |
| M6 object identification | the class name ("MOTORCYCLE"), plus a per-class spec table |

M1–M2 and M11–M13 need no model at all. Only M8 (speech), M9 (OCR), M10
(face mesh) and M14 need anything else, and they are optional extensions.

**Detectors.** Both are trained on COCO's 80 classes
([`perception::labels`](../src/perception/labels.rs)) and run through the
same GPU preprocessing, worker and tracker. The app prefers D-FINE when its
model is on the device and falls back to YOLOX-Nano.

| | Licence | Size | COCO AP | Compute at the app's input | Path into Meganeura |
|---|---|---|---|---|---|
| **D-FINE-N** | Apache-2.0 | 3.8M | 42.8 | 640×480: ~5.3 GFLOPs, 1967 dispatches | ONNX import of the Transformers export |
| **YOLOX-Nano** | Apache-2.0 | 0.91M | 25.8 | 416×320: 0.83 GFLOPs, 254 dispatches | hand-built graph, official weights |

On the YOLOX demo photo, D-FINE finds BICYCLE 0.95, DOG 0.94 and TRUCK
0.66, with tighter boxes. YOLOX-Nano finds DOG 0.85, BICYCLE 0.80 and CAR
0.80, and calls the pickup a car. D-FINE takes the frame stretched (not
letterboxed) to its input, RGB in `0..=1`, and returns 300 queries of
sigmoid class logits and normalised `cx, cy, w, h` boxes, with no NMS. Its
cost on the Quest is unknown: about six times YOLOX-Nano's arithmetic, plus
a dispatch count Meganeura item 3 should cut.

**YOLOX** (Megvii, Apache-2.0) It is a plain CNN:
conv, depthwise conv, SiLU, max-pool, nearest upsample and concat, all of
which Meganeura's graph builder already has. The Focus slicing goes into
our preprocessing shader. Its output is a dense grid of (box, objectness,
80 class scores), decoded with NMS on the CPU. Official ONNX weights are
published (`yolox_nano.onnx` 3.6 MB, `yolox_tiny.onnx` 20 MB), so the
graph's output can be checked against onnxruntime on the host.

| Variant | Params | GFLOPs | COCO AP | Use |
|---|---|---|---|---|
| YOLOX-Nano @416 | 0.91M | 1.08 | 25.8 | start here; leaves room in the budget |
| YOLOX-Tiny @416 | 5.06M | 6.45 | 32.8 | if Nano misses too much at distance |

Upgrades if quality demands them, in order of effort:

1. **Fine-tune** on egocentric, low-light, near-greyscale passthrough
   captures, possibly with fewer classes. The Quest cameras are not the
   clean photos COCO was shot with (DinoVision saw the same domain gap).
2. **Larger D-FINE** (S: 48.5 AP) if the Quest has room; the same import
   path loads it.
3. **More names** than COCO's 80 via Objects365-trained weights (365
   classes), where an Apache release exists for the chosen architecture.

**Evaluated for the wireframe and voice** (all Apache-2.0 or MIT):
RF-DETR-Seg-Nano and EdgeTAM (segmentation), Depth Anything V2 and V3 Small
(depth), Moonshine-tiny (speech). See TASKS.md items 3–5.

**Licences.** Excluded: Ultralytics YOLOv8/11/26 (AGPL), YOLO-World (GPL),
and DEIMv2 and other non-commercial weights. DINOv3 derivatives carry its
licence (no military or weapons use); DINOv2 is Apache-2.0.

**Optional extensions**, from the survey, if a mechanic beyond the core is
wanted: RTMPose-t (Apache) for real body measurements; BlazeFace (Apache)
for face boxes; PP-OCR mobile (Apache) for text reading; Silero VAD and
Moonshine-tiny (MIT) for the response menu; SmolVLM2-256M (Apache, already
in Meganeura) for scene descriptions.

## Detector implementation

Where the pieces live, and how each was verified (all on lavapipe, in CI):

| Piece | Where | Verified by |
|---|---|---|
| Weight conversion | `tools/yolox_to_safetensors.py`, `tools/fetch_yolox.sh` (pinned SHA-256) | the network refuses a checkpoint whose conv shapes differ |
| Network | `perception/yolox.rs`: CSPDarknet, PAFPN and decoupled head, built in the reference's forward order | `tests/yolox.rs`: worst relative error against onnxruntime 1.2e-5 (Nano, random), 9.5e-5 (Nano, photo), 9.8e-6 (Tiny) |
| Preprocessing | `shaders/preprocess.wgsl` into `Session::input_buffer`, driven by each model's `InputSpec`; CPU reference `preprocess::preprocess` | `tests/gpu.rs`: GPU equals CPU on a colourful frame for both YOLOX letterboxes and D-FINE's stretch |
| YOLOX decode, NMS | `yolox::decode`, `perception::nms` (best class per anchor, class-aware NMS, as the reference) | unit tests; the YOLOX demo photo yields DOG 0.85, BICYCLE 0.80, CAR 0.80 |
| D-FINE | `perception/dfine.rs`: ONNX import, outputs identified by size, best class per query, no NMS; `tools/fetch_dfine.sh` prepares the export | `tests/dfine.rs`: 5.5e-5 against onnxruntime; BICYCLE 0.95, DOG 0.94, TRUCK 0.66 end to end |
| Worker | `perception/worker.rs`: detector built on its thread, frames dropped while busy, tracker, `Observation` | unit tests with a fake detector |
| Display | `render::BoxPass`, the grade pass's placement; XR app outlines confirmed tracks | `tests/gpu.rs`; on-device pending |

Notes:

- The Focus slicing is folded into preprocessing, so the graph input is
  `[12, H/2, W/2]` and needs no strided slice. The weights are
  resolution-independent; the app runs 416×320 to match the cameras' 4:3.
- The hand-built graph and the ONNX import of the whole official export
  agree, so either can serve. The hand-built one stays: it compiles to 254
  dispatches against the import's 370. Its input is the Focus layout our
  preprocessing writes, so it skips the eight strided slices, and it writes
  the flat output the decoder reads, with no reshapes and transposes and no
  Python step to prepare the model.
- The network needed nothing new from Meganeura: `conv2d`, `conv2d_dw`,
  `max_pool_2d`, `upsample_2x`, two-input `concat` (also used as a flat
  append to join the three output levels), `silu` and `sigmoid`.
- The XR app loads `/data/local/tmp/red_eye/dfine_nano.onnx` if present,
  else `yolox_nano.safetensors`. It caps submissions at 24 Hz and uses two
  submission chunks, as DinoVision found best. Both are guesses until
  measured on the Quest.
- On lavapipe a frame takes about 1 s with YOLOX-Nano and 4 s with D-FINE-N;
  that is the CPU emulating a GPU and says nothing about the Quest.

## HUD

The HUD is egui, painted by `blade-egui`, with its monospace text drawn by
the grid from Starcom (`src/hud/grid.rs`, copied, MIT, same author). The
grid puts a whole panel of glyph cells into one batch of textured quads
while egui keeps fonts, shaping and the glyph atlas. That suits the films'
terminal-like readouts and the Quest's per-dispatch cost. Moving the grid
into `blade-egui`, so Starcom and RedEye share it, is a later Blade change.

- **One layout per frame.** `hud::canvas::Canvas` runs egui into an
  offscreen sRGB texture with premultiplied alpha. Each eye then composites
  it with the camera image's placement, so normalised track rectangles map
  straight onto it. Nothing is laid out twice for stereo, and the texture can
  become a compositor quad layer once Blade supports layers.
- **Layout** (`hud::view`, T2's grammar): an ANALYSIS block and a TRACKING
  block in the top corners, with churning filler around real telemetry and
  track identities; a status stack mid-right (scan mode, target, status,
  probability), typed out whenever targeting changes; the state word
  bottom-centre ("SCANNING", then "MATCH ■"); and corner brackets with the
  class name around the matched object. Positions and sizes are fractions
  of the canvas. A dark offset copy under every glyph keeps text legible
  over bright passthrough.
- **Targeting** (`hud::targeting`): a named COCO class. The largest
  confirmed track of that class matches, and the match sticks to its track
  while the track lives. Until voice commands land, the XR app reads the
  name from `/data/local/tmp/red_eye/target`. While no target is named,
  debug boxes outline every confirmed track; once one is, the HUD's
  brackets take over.
- **Font**: Share Tech Mono (Carrois Type Design, SIL OFL 1.1, bundled with
  its licence in `assets/fonts/`), loaded into egui unmodified.
- **Next: voice commands** ("track the motorcycle") through Moonshine-tiny
  (MIT, 27M). It takes the raw 16 kHz waveform, with no spectrogram; a 2 s
  window suffices for commands. Estimated Quest cost is about 38 ms per
  utterance at 2 s (0.76 GMAC encoder plus up to 8 decoder steps of 0.38
  GMAC), one-off rather than per frame. On synthetic espeak speech, free
  greedy decoding gets only 5 of 10 object words ("bicycle" became
  "mysql"). Decoding constrained to the command grammar ("track | target |
  find | lock on" + "the | a" + a COCO label, or "cancel") gets 8 to 10 of
  10. So the app decodes through a token trie of that grammar, precomputed
  offline from the tokenizer, plus a synonym table. Blocked on Meganeura
  item 4.

## Frame budget at 24 Hz

41.7 ms per perception frame, of which roughly 20 ms of GPU time is
available to inference; the compositor and our HUD need the rest. Estimates
assume the ncnn-like rate of 0.1 GMAC/ms until the calibration benchmark
replaces it.

| Slot | Work | Cadence | Est. GPU ms |
|---|---|---|---|
| Preprocess | YUV→RGB, resize, Focus slicing (compute shader) | every frame | ~1 |
| Detection | YOLOX-Nano @416×320, 0.83 GFLOPs | every frame | ~4 |
| or detection | D-FINE-N @640×480, ~5.3 GFLOPs, 1967 dispatches | every frame if it fits, else every 2nd | ~26 + dispatch overhead, before tuning |
| Decode, NMS, tracking | CPU | every frame | 0 (CPU <1 ms) |
| Depth | Depth API | every frame | 0 |

YOLOX-Nano leaves headroom for one optional extension on a staggered
cadence. D-FINE-N does not fit the budget at these estimates until the
TopK kernel and Adreno tuning land; the device measurement decides between
them.

**Measured on a Quest 3S** (Horizon OS 210, the XR app, camera streaming,
November 2026). The display runs at 72 Hz.

| Detector | Render | Detect | Latency | GPU per inference |
|---|---|---|---|---|
| none | 72.0 Hz | — | — | — |
| YOLOX-Nano @416×320 | 68–71 Hz | 18–20 Hz | 32 ms | — |
| D-FINE-N @640×480, 2 submission chunks | 17–19 Hz | 7.5 Hz | 127–132 ms | 123 ms |
| D-FINE-N, 24 chunks | 22–34 Hz | 7 Hz | 130–141 ms | |
| D-FINE-N, 64 chunks | 38–40 Hz | 6.5 Hz | 140–150 ms | |

D-FINE's GPU time by kernel family (Meganeura's profiler, 1967 dispatches):

| Family | Dispatches | GPU ms | Share |
|---|---|---|---|
| convolution (`Conv2dGemm*`) | 81 | 70.9 | 61% |
| matmul | 64 | 20.6 | 18% |
| data movement | 841 | 10.4 | 9% |
| pointwise | 628 | 8.7 | 8% |
| normalisation, reduction | 353 | 5.8 | 5% |

D-FINE occupies the GPU so much of the time that the renderer misses
frames; finer submission chunks let it interleave but slow inference.
YOLOX-Nano fits today. Thermals: SoC at about 36 °C after a few minutes,
no throttling.

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
  LUT; headless preview; tracker, COCO labels and text effects; XR crate
  type-checks.
- **M1**: on device, the left camera graded through `GradePass` to both eyes.
  Plus the calibration benchmark and Quest 3 device facts (Meganeura items
  1–2).
- **M2 (host-verified, not yet on a device)**: the detectors. YOLOX as a
  Meganeura graph with the official weights, and D-FINE-N through the ONNX
  importer, both matching onnxruntime; GPU preprocessing; decode and
  tracking on a worker thread; boxes over the graded view in the XR app.
  Details under "Detector implementation".
- **M3 (HUD part host-verified)**: HUD text and panels through egui and
  the Starcom grid, named-target MATCH with brackets, data blocks. Still to
  come: the Blade XR work (layers, passthrough with the colour map,
  environment depth, past poses), world-space targets, boot and damage.
- **M4**: the rest of the core mechanics: acquisition inset, size
  assessment, threat assessment, compass. Adreno kernel work driven by the
  benchmark; a detector fine-tune on passthrough captures if needed.
- **Later**: optional extensions (speech, OCR, pose).
