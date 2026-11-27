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

**Status: idea.**

Uses: T2's traced outlines around the subject (M6), the wireframe below
(item 6), and size assessment (M5). Masking only the locked target keeps it
cheap: a box-prompted model on the target's crop, every few frames.

Candidates (evaluate with `tools/onnx_reference.py` and
`examples/onnx_check.rs`):

| Model | Licence | Notes |
|---|---|---|
| EdgeTAM | Apache-2.0 | Box-prompted, carries the mask across frames; image encoder about 19 ms on the Snapdragon NPU (unverified for our GPU path) |
| RF-DETR-Seg-Nano | Apache-2.0 | Boxes and masks for everything in one model; could replace the detector |
| EoMT-S | MIT code, DINOv2/v3 weights | Plain ViT plus queries; heavy at 640 |
| BlazePose (segmentation) | Apache-2.0 | Person mask plus 33 keypoints; people only |

Excluded: YOLO-seg and FastSAM (AGPL).

The first visible result would be an outline from the mask (marching squares,
or edges in a shader) in the HUD preview.

## 5. Depth

**Status: blocked** (Blade: environment depth). **Idea** for the monocular
fallback.

The Quest Depth API (`XR_META_environment_depth`) is free in the sense that
the system computes it and our GPU pays nothing. It is not wired into the app
yet. It is low resolution with soft edges (exact numbers unverified), and it
comes from different sensors than the camera. Using it with camera-image
masks means reprojecting through the head pose at the camera's capture time
(Blade work items 3–4).

The alternative is Depth Anything V2 Small (Apache-2.0) on the target's
crop only. It gives relative depth, pixel-aligned with the camera image
and the mask, with metric scale taken from the Depth API's median over the
target.

Uses: size assessment (M5), world-space targets and lock prediction (M3),
and the wireframe.

## 6. Character wireframe

**Status: idea.** Needs a mask (item 4) and depth (item 5); pose for the
last step.

The films trace subject outlines and, in T2, draw a triangulated mesh in
the smile analysis. The T2 script asks for an outlined body in the bar scene
(not confirmed in the cut). A wireframe that follows the person's shape fits
that grammar. Steps, each convincing on its own:

1. **Grid in the silhouette** (no depth): a screen-space grid clipped to the
   mask, plus the outline. Possible as soon as item 4 exists.
2. **Depth contours**: lines of constant depth inside the mask, from one
   shader (`fract(depth / spacing)`). They wrap around the body like a
   topographic scan, and they tolerate coarse depth, because the lines come
   from interpolated values.
3. **Mesh wireframe**: a grid over the target's box, each vertex lifted to 3D
   with its depth and drawn in perspective. Cells outside the mask or across
   depth jumps are dropped, and a scan sweep animates it in.
4. **Body parts**: pose keypoints (RTMPose, Apache-2.0) split the body into
   torso and limbs. Each part gets rings fitted to the mask's width and to
   depth, so the wireframe sits on specific parts: the clothing-match look,
   highlighting the parts being "measured".

Cost: the mask on the target's crop every 2–4 frames; depth free (Depth API)
or one small crop model; the wireframe is one draw.

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
