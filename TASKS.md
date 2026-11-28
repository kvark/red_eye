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

**Status: done** (Blade #428). The app shows the system's passthrough
graded by the colour map, with our layer alpha-blended over it, and falls
back to drawing the camera itself without `XR_FB_passthrough`. Still to
judge with someone wearing it: whether the colour map wants linear or
sRGB values (`/data/local/tmp/red_eye/lut_linear` switches).

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

Measured on a Quest 3S (technology.md, Frame budget): YOLOX-Nano runs at
18–20 Hz with the display at 68–71 Hz. D-FINE-N takes 123 ms of GPU time
per inference and drags the display to 17–19 Hz; 61% of that is
convolution and 18% matmul, while all its TopK and other data-movement
dispatches together take 10 ms.

Register-blocked convolution and matmul kernels in Meganeura (PRs #394,
#395, #396) cut D-FINE to 55 ms of GPU time: the display holds about
60 Hz with detection at 11 Hz and about 80 ms latency. A single-dispatch
TopK (PR #397) then takes it to 757 dispatches and 49.6 ms (measured
natively; the XR app's frame rates with it are still to record).

Next:
- Meganeura: convolutions are half of what is left (25 of 50 ms);
  per-shape tuning of the register blocks, and f16 arithmetic.
- About 560 data-movement and pointwise dispatches take 14 ms: fusion.
- Meanwhile, ship YOLOX-Nano on the device, or cap D-FINE's rate.
- Meganeura: a single-dispatch TopK kernel, after the convolutions.
- A capture mode, as DinoVision has, and a fine-tune of the detector on
  egocentric, low-light, near-greyscale passthrough frames.
- D-FINE-S (48.5 AP) if the budget allows; Objects365 weights for more
  class names than COCO's 80.

## 3. Voice control: "track the motorcycle"

**Status: done** on the Quest's model side, and wired into the app
(`voice`, `android-xr/src/listen.rs`); not yet heard through the headset's
own microphone. Meganeura PR #400 (integer inputs, InstanceNormalization)
is pinned by commit until it merges.

On the Quest, from synthesised speech (two Windows voices):
- every one of 20 commands was right and accepted, at confidence −0.08 to −0.72;
- "hello there" and "what time is it" were rejected, at −6.1 to −9.3;
- each utterance took 42–95 ms, the encoder and up to 6 decoder steps.

`voice::MIN_CONFIDENCE` (−1.5) sits in that gap. Still to check with a
wearer: real voices, room noise, the detector's thresholds, and whether
the HUD's "VOICE INPUT" line reads well.

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

**Status: done** on the host and the Quest for the monocular model
(`perception::depth`, Depth Anything V2 Small). The Quest Depth API is
wired (`perception::environment`), and the app is waiting on the Blade pin
bump (Meganeura #399) to land.

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
  nothing. It is low resolution with soft edges, and it comes from
  different sensors than the camera. `perception::environment` finds where
  each camera ray meets the sensed surface, iterating between the two
  viewpoints. Each confirmed track gets the median over the middle of its
  box, shown as its range in the HUD. The plane that camera-image overlays
  are placed on follows the outlined object, which removes the
  camera-to-eye parallax for it.
  Next:
  - confirm the encoding and row order on the device;
  - use the head pose at capture time (`Context::xr_time_ago`) rather than
    at display time, since the head moves in the ~75 ms between;
  - scale Depth Anything's contours to metres with it.

Done: `perception::depth` runs DA-V2-S on the whole camera frame into a
`dense::DepthMap`, on its own thread (`worker::Background`) at most every
250 ms. `tools/fetch_depth.sh` folds the ImageNet normalisation into the
patch embedding, so the model takes the same 0..1 RGB as D-FINE. On the
Quest, 224×168 takes about 85 ms and 336×252 about 180 ms; both match
onnxruntime to 5e-5 (`tests/depth.rs`). Bringing it up found an Adreno
miscompile in Meganeura's register convolution, fixed in PR #394.

Next: the Depth API for metric scale and world positions; DA-V2 on the
target's crop if whole-frame depth is too coarse.

Uses: size assessment (M5), world-space targets and lock prediction (M3),
and the wireframe.

## 6. Wireframe: depth contours inside masks

**Status: working on the demo photo** (`examples/hud.rs` with a depth
model) and wired into the XR app, not yet seen on the headset. Masks come
from the detector's boxes cut by depth (`dense::silhouettes`): within each
box, the pixels nearer than its median. That follows the object's outline
where it stands in front of its background, but takes in whatever stands
in front of it too (the dog, in the bicycle's box). Item 4's instance masks
replace it when they land.

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
- In the app: the matched target is outlined alone; with no target named,
  every confirmed track is. Contours divide the depth range inside the
  outlines into eight and drift slowly.
- Next: see it on the headset; RF-DETR-Seg-Nano masks (item 4).

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

- **First run: done** on a Quest 3S. The graded camera, detection and HUD
  all work, and the device facts are in technology.md (f16 yes, coopmat
  no). `tools/quest_setup.sh` prepares a connected headset for unattended
  runs. Still to look at with someone wearing it: HUD legibility, colours,
  and how the camera image lines up with the world.
- Meganeura's calibration benchmark (ms/GMAC per kernel archetype), now
  that GPU timing works on Adreno (Blade branch
  `timing-without-calibration`).
- Camera intrinsics and extrinsics: **done**. Camera2 reports them, and
  overlays are projected into each eye through a plane 1.5 m ahead
  (`src/projection.rs`). Off that plane they are off by the camera-to-eye
  parallax, a degree or two at arm's length in the right eye; metric depth
  per object (item 9) removes it.
- World-space targets: detections placed through depth and the head pose at
  capture time, so locks stay on people as the head moves.
- Thermals: log them with timings, as DinoVision does.

## 9. Stereo HUD: object overlays at the object's depth

**Status: idea.** Builds on depth (item 5), world-space targets (item 8)
and, to pay off fully, compositor passthrough (item 1).

Today one HUD canvas is composited identically into both eyes, so every
mark sits at the same apparent depth, wherever the object it describes is.
Anything about an object can instead be drawn per eye at that object's
depth, so it fuses onto the object in true stereo: the box and its label,
the brackets and MATCH, the wireframe.

- Split the HUD in two. Readouts (corner blocks, status stack, state word)
  stay on the body-locked canvas 1.5–2 m away (concept.md, Comfort).
  Object marks become world-space geometry, rendered with each eye's own
  view and projection.
- Placing an object: the ray through its box centre from the camera's
  intrinsics and extrinsics (item 8), at a distance from the Quest Depth
  API's median inside the box (or the mask), or failing that a class size
  prior (a person is about 1.7 m tall). Monocular depth (DA-V2) gives shape,
  not scale, so it rides on that distance.
- The box becomes a quad facing the viewer at the object's distance, sized
  from the box's angular extent; its label sits on the quad's edge in the
  same plane, so text and object share one vergence.
- The wireframe becomes a mesh: the contour pass's grid displaced by the
  depth map, scaled to the object's distance, drawn per eye. The contours
  then wrap around the object in depth, not only on the image.
- Anchor marks in world space with the head pose at capture time, so they
  stay on the object between detections and through capture latency
  (technology.md, risk 4).
- Why: placing marks at the object's depth avoids a vergence conflict
  between them and the object, which is less tiring and reads as part of the
  world instead of a sticker on the visor. Stereo marks over today's mono
  camera image would disagree with the image itself, so measure with
  compositor passthrough (item 1).
- Check on the device: how stable the depth is frame to frame (jitter reads
  as marks swimming), and whether placement error over a few metres is
  tolerable.
