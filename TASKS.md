# RedEye: what to explore

A living backlog of the ideas we mean to try, with where each stands. The
why and how in depth live in [docs/concept.md](docs/concept.md) (mechanics
M1–M14) and [docs/technology.md](docs/technology.md) (architecture, models,
upstream work); this file says what is next. Update an entry's status when
it moves.

Status: **done** (verified on the host), **ready** (unblocked, not started),
**blocked** (on the named dependency), **idea** (worth trying, not yet
designed). Owner: RedEye unless marked Blade or Meganeura, which are ours too
but other sessions are working on them.

## 1. Passthrough overlay: the compositor draws the red world

**Status: blocked** (Blade: composition layers, passthrough style).

The system's own passthrough, graded by the 256-entry colour map that
`Grade::mono_to_rgba_lut` already produces, is stereo-correct, at display
rate and costs our GPU nothing. Our layer then carries only the HUD, alpha
over it. Today the app draws the left camera itself through `GradePass`:
mono, narrower than passthrough, and competing with inference for the GPU.

- Blade: submit an `XR_FB_passthrough` layer under ours, set its colour
  map and edge colour, and mark our projection layer alpha-blended
  (technology.md, Blade work items 1–2).
- RedEye: switch the world view, keep `GradePass` as the fallback, and
  check on device whether the colour map is linear or sRGB-encoded.
- The HUD canvas is already a single texture, so it can become a
  compositor quad layer, body-locked with lazy follow.

## 2. Object detection: the rectangles

**Status: done** on the host. D-FINE-N (42.8 AP, ONNX import) is preferred,
YOLOX-Nano (25.8 AP, hand-built graph) is the fallback. Both are COCO-80
and Apache-2.0, and both match onnxruntime.

Next:
- Run on a Quest: frame rate, latency and thermals with the XR app
  (needs the headset).
- Meganeura: a single-dispatch TopK kernel. D-FINE has 1967 dispatches,
  most of them from TopK.
- A capture mode, as DinoVision has, and a fine-tune of the detector on
  egocentric, low-light, near-greyscale passthrough frames.
- D-FINE-S (48.5 AP) if the budget allows; Objects365 weights for more
  class names than COCO's 80.

## 3. Voice control: "track the motorcycle"

**Status: ready** on the RedEye side; **blocked** for running the model
(Meganeura: `InstanceNormalization`, integer graph inputs for the token
`Gather`).

Moonshine-tiny (MIT) takes 2 s of raw 16 kHz audio, about 38 ms per
utterance on the Quest (estimated). Decoding is constrained to the command
grammar, "track / target / find / lock on" + "the / a" + a COCO class, or
"cancel". On synthetic speech that scores 8–10 of 10, against 5 of 10 for
free decoding.

RedEye side:
- Microphone through AAudio, with a `RECORD_AUDIO` runtime grant. Listen
  only while the app is focused, and never store audio.
- An energy-based voice activity detector to cut utterances.
- A tool that turns the grammar into a token trie using the tokenizer, and
  the trie-constrained greedy decoder in Rust.
- A synonym table ("bike" → bicycle, "guy" → person, "phone" → cell phone).
- Commands drive `hud::Targeting`, replacing the adb target file. The
  transcript types out in the HUD ("VOICE INPUT: …").

## 4. Segmentation: silhouettes and masks

**Status: evaluated; blocked** (Meganeura: three importer additions).

Uses: T2's traced outlines around the subject (M6), the wireframe (item 6),
and size assessment (M5). The wireframe shows whatever masks the network
produces, so instance masks for everything and a mask only for the target
both work.

Evaluated through the importer on the YOLOX demo photo. Numbers are from
lavapipe; Quest times are estimated at 0.1 GMAC/ms. Scripts are in
`tools/experiments/segdepth/`.

| Model | Licence | Cost | Masks | Importer gaps |
|---|---|---|---|---|
| **RF-DETR-Seg-Nano** at 312² (first choice) | Apache-2.0 | 22.7 GMAC, ~227 ms | Boxes and instance masks for all of COCO in one model, 78×78. Bicycle 0.95, dog 0.94, truck 0.82; masks right but coarse | runtime `Sin`/`Cos` (×4 each), `Where` on runtime data (×1), `Einsum` `bchw,bnc->bnhw` (×1). With those rewritten: matches onnxruntime (1.7e-3 on mask logits), 1342 dispatches |
| EdgeTAM, box-prompted (sharper, one target) | Apache-2.0 | encoder 19.4 GMAC at 1024² (~194 ms) + decoder 1.8 GMAC | 256×256, IoU 0.97 on dog and truck | encoder: `ReduceMean` over [2,3] (×10, is `GlobalAveragePool`); prompt path: runtime `Sin`/`Cos`, `ScatterND`, an INT64 input (or encode prompts on the CPU) |
| MobileSAM, EfficientViT-SAM-L0 | Apache-2.0 | 35–39 GMAC | — | `ReduceMean` over axis 1 (LayerNorm2d). Too heavy |
| DeepLabV3-MobileNetV2 (VOC) | Apache-2.0 | 8.8 GMAC | Semantic only, no instances | dilated depthwise `Conv`, `AveragePool`, `ArgMax`, UINT8 input |

Excluded: YOLO-seg and FastSAM (AGPL). The only RF-DETR-Seg ONNX found is a
third-party export with its output names swapped. Re-export from
`Roboflow/rf-detr-seg-nano` before relying on it. RF-DETR-Seg could also
replace D-FINE as the detector, since it outputs boxes too.

Next: once Meganeura has the three ops, add a `perception::rfdetr` like
`dfine.rs`, producing detections and a `dense::Mask`. Run it on a staggered
cadence, or on the target's crop.

## 5. Depth

**Status: ready** for the monocular model. **Blocked** for the Quest Depth
API (Blade: environment depth).

- **Depth Anything V2 Small** (Apache-2.0, `onnx-community/depth-anything-v2-small`)
  imports through Meganeura **with no gaps**:
  - at 266², 11.9 GMAC (~119 ms on the Quest), 432 dispatches, matching
    onnxruntime to 5.4e-5;
  - at 224², ~82 ms;
  - at 364×476, 33 GMAC.

  It outputs relative inverse depth at the input resolution. It is
  plausible on the demo photo: near porch and dog, far street and truck,
  crisp wheel edges. Run it on the target's crop, pixel-aligned with the
  camera image and the mask. Its contours are of disparity, so they are
  spaced unevenly in depth, which is fine for the look.
- **Depth Anything 3 Small** (Apache-2.0) outputs depth (up to scale), so its
  contours are evenly spaced. Same cost at 266². One gap: a `ScatterND`
  writing a constant row into a runtime tensor. Rewritten as Slice and
  Concat, it matches onnxruntime to 7.8e-6.
- **The Quest Depth API** (`XR_META_environment_depth`) costs our GPU
  nothing, but it is not wired up yet. It is low resolution with soft edges
  (exact numbers unverified), and it comes from different sensors than the
  camera, so using it means reprojecting through the head pose at capture
  time (Blade work items 3–4). Use it for metric scale (its median over the
  target) and for world-space positions.

Next: a `perception::depth` running DA-V2-S on a crop into a
`dense::DepthMap`, scheduled with segmentation. An estimated ~350 ms per
segmentation-plus-depth cycle means about 3 Hz on the worker, a guess until
measured on the Quest.

Uses: size assessment (M5), world-space targets and lock prediction (M3),
and the wireframe.

## 6. Wireframe: depth contours inside masks

**Status: in progress.** The contour pass is built and tested on synthetic
depth and masks. It needs a real mask source (item 4) and depth (item 5).

Decision: lines of constant depth inside every mask the segmentation
network produces, whatever its classes, plus the mask outline. There is no
split into body parts and no lifted 3D mesh. The films trace subject
outlines, and T2 draws a triangulated mesh in the smile analysis. Contours
that wrap around a shape read the same way, and they tolerate coarse depth,
because the lines come from interpolated values.

- `render::WireframePass`, implemented as one fragment pass over the camera
  image's placement:
  - An instance-id mask and a float depth map, both in camera-image
    coordinates at any resolution.
  - Contour lines from `fract(depth / spacing)`, antialiased with screen-space
    derivatives.
  - The outline where the instance id changes, and a faint fill.
  - One highlighted instance (the locked target) brighter than the rest.
  - A phase that makes the contours drift, for a scanning look.
- Next: feed it from RF-DETR-Seg-Nano masks (item 4) and DA-V2-S depth
  (item 5), first in `examples/hud.rs` on the demo photo, then in the XR app.
  Depth can be wired now; masks wait on the importer.

## 7. HUD mechanics

**Status: ready**, unless noted.

- Data scroll with code-listing text (M2): our own pseudo-assembly
  scrolling through the grid.
- Boot / diagnostic sequence (M12) with real self-tests: camera, models,
  GPU timing.
- Damage (M13): scanlines, jitter and channel split, adapted from Starcom's
  `bend.rs` (CRT curve, Matrix rain).
- Compass rose from head yaw (M11).
- Acquisition inset (M4): a mosaic-resolving crop of the target's head
  region.
- Size assessment (M5): needs depth (item 5); pose refines it.
- Threat assessment (M7): people counts and trajectories, never labelling a
  real person a threat.
- Move the Starcom grid into `blade-egui` (Blade), shared with Starcom.

## 8. Platform

- **First run on a Quest 3** (needs the headset): the graded camera,
  detection and HUD, plus Meganeura's calibration benchmark and Quest 3
  device facts (f16, coopmat, subgroups).
- Camera intrinsics and extrinsics: standard Camera2 tags, or a one-off
  calibration.
- World-space targets: detections placed through depth and the head pose at
  capture time, so locks stay on people as the head moves.
- Thermals: log them with timings, as DinoVision does.
