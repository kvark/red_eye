# Takeover: continuing RedEye with a Quest 3

This repository was built and tested on a cloud machine with no headset and
no GPU (software Vulkan, lavapipe). Everything below was verified there.
**Nothing has yet run on a Quest.** The next step needs real hardware.

Read [AGENTS.md](AGENTS.md) for the layout, commands and rules, and
[TASKS.md](TASKS.md) for the backlog. This file covers what is specific to
picking the work up.

## Where things stand

- **Branch `claude/kind-archimedes-8vh8rk`.** `main` has everything up to
  D-FINE (PR #2). The branch adds, not yet in a PR (`git log
  origin/main..claude/kind-archimedes-8vh8rk`):
  - the HUD (egui plus Starcom's text grid, named-target MATCH);
  - the depth-contour wireframe pass;
  - the Moonshine and segmentation/depth evaluations (results in
    TASKS.md and `docs/technology.md`, scripts in `tools/experiments/`);
  - TASKS.md and this file.

  Open a PR from it, or keep working on it.
- **Pins:** Meganeura `b25f763`; Blade `bba0edc` (also `blade-egui` and
  `blade-util`); egui 0.34.3; openxr 0.21. CI uses the latest stable Rust
  (1.100 at the time of writing).
- **Verified on the host (lavapipe and CI):**
  - Detectors: D-FINE-N and YOLOX match onnxruntime.
  - GPU preprocessing equals its CPU reference.
  - The grade shader equals its CPU reference.
  - HUD layout (MATCH and brackets) and the wireframe pass (rings inside
    masks).
  - Tracker, targeting and text effects (unit tests).
- **Compiles but has never run:**
  - the whole `android-xr` app (clippy-clean for `aarch64-linux-android`);
  - its Camera2 module (ported from DinoVision, where the original ran on a
    Quest 3);
  - the HUD and detectors on Adreno.

## Set up the machine

Host:
```sh
rustup toolchain install stable && rustup target add aarch64-linux-android
cargo install cargo-apk
python3 -m venv .venv && . .venv/bin/activate && pip install -r tools/requirements.txt
tools/fetch_yolox.sh && tools/fetch_dfine.sh          # models and references into ref/
cargo test --release -- --include-ignored            # unit, GPU, YOLOX and D-FINE tests
cargo run --release --example hud -- ref/dfine_nano_640x480/model.onnx ref/dog.jpg bicycle out
```
Any Vulkan GPU works for the host tests; lavapipe
(`apt install mesa-vulkan-drivers`) is enough.

Android:
- Android SDK with platform 34, NDK r27, and `ANDROID_HOME` /
  `ANDROID_NDK_HOME` set for cargo-apk. Horizon OS's camera API needs API 34.
- **Clone Blade next to this repository** (`../blade`).
  `android-xr/Cargo.toml` takes the OpenXR loader from
  `../blade/examples-android/libs` and the debug keystore from
  `../blade/examples-android/keystore/`, as DinoVision does.
- A Quest 3 in developer mode with USB debugging; `adb devices` lists it.

## First run on the Quest

```sh
adb shell mkdir -p /data/local/tmp/red_eye
adb push ref/dfine_nano_640x480/model.onnx /data/local/tmp/red_eye/dfine_nano.onnx
adb push ref/yolox_nano.safetensors /data/local/tmp/red_eye/
cargo apk run --manifest-path android-xr/Cargo.toml --release --no-logcat   # builds, installs, starts
adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA     # after each fresh install
adb shell am force-stop rust.red_eye_xr
adb shell am start -n rust.red_eye_xr/android.app.NativeActivity            # restart with the camera
adb logcat -v time | grep -E "red_eye|RustStdoutStderr"
```

The first start, before the grant, falls back to the test pattern; the
restart picks the camera up.

**Wear the headset.** Horizon OS will not start an OpenXR activity while the
proximity sensor says it is off; `SensorLockActivity` in the log means
exactly that.

Expected, in order:
1. `=== red_eye starting ===`, then `XR_FB_passthrough true,
   XR_META_environment_depth true` (both are needed later), and the
   swapchain format.
2. `opened camera …`. If you see `camera unavailable …` instead, you get
   the test pattern: check the permission grant, and that passthrough is
   enabled.
3. **The red camera image in both eyes, fusing into one view.** If the eyes
   disagree or the image looks offset, see `eye_placement` in
   `android-xr/src/lib.rs`: Quest frusta are asymmetric, and DinoVision
   documents the measurements and has a camera half-FOV override.
4. `detector ready: D-FINE-N at 640x480 …` (or YOLOX-Nano if the D-FINE
   model is absent). Every 5 s:
   `render X Hz | detect Y Hz, latency Z ms | <targeting phase>`.
5. Debug boxes on confirmed detections: white for people, cyan for
   vehicles.
6. Name a target: `adb shell "echo person > /data/local/tmp/red_eye/target"`.
   The HUD shows SCANNING, then brackets and MATCH when a person is in view.
   `adb shell rm /data/local/tmp/red_eye/target` brings back the boxes.

**Record** these numbers, then put them into `docs/technology.md` (frame
budget, risks) and TASKS.md item 8:
- render Hz with and without the detector;
- detect Hz and latency for D-FINE and for YOLOX (remove
  `dfine_nano.onnx` from the device to get YOLOX);
- thermal behaviour over a few minutes;
- HUD legibility;
- whether colours look right;
- how well the camera image lines up with the real world.

## Likely first problems

- **D-FINE too slow.** It is 1967 dispatches per frame, mostly TopK, which
  Meganeura lowers one pass per selected element. Fall back to YOLOX-Nano
  (254 dispatches) by removing the D-FINE model, and report the timings: a
  single-dispatch TopK kernel is Meganeura's job (`docs/technology.md`,
  Meganeura item 3).
- **The render rate drops while detecting.** Inference and rendering share
  one Vulkan queue. The knobs are `SUBMISSION_CHUNKS` and `DETECT_INTERVAL`
  in `android-xr/src/lib.rs`.
- **Colours washed out or too dark.** The grade, HUD and wireframe convert
  to linear when the swapchain format is sRGB (`is_srgb`). Check the logged
  format.
- **HUD text too small or too large.** The canvas is 1280×960 at one pixel
  per point, and the layout is in fractions of it (`src/hud/view.rs`).
- **Alignment.** Boxes and the HUD are in camera-image space, drawn wherever
  the camera image is placed. Real passthrough, and world-locked overlays,
  wait on the Blade work (TASKS.md items 1 and 5).

## What comes after bring-up

In TASKS.md order of usefulness:
1. **Measurements**: Meganeura's on-device calibration benchmark and the
   Quest 3 device facts (f16, cooperative matrices, subgroups);
   `docs/technology.md`, Meganeura items 1–2.
2. **Depth for the wireframe**: Depth Anything V2 Small imports with no
   gaps. Add `perception::depth` on the target's crop, into
   `dense::DepthMap`, and show `WireframePass` in the app (TASKS.md items
   5–6).
3. **Masks**: RF-DETR-Seg-Nano once Meganeura has runtime `Sin`/`Cos`,
   `Where` and that `Einsum` (TASKS.md item 4).
4. **Voice**: microphone (AAudio, `RECORD_AUDIO`), a voice activity
   detector, and the grammar-constrained Moonshine decoder (TASKS.md item 3;
   `tools/experiments/moonshine/constrained.py` is the reference). The model
   run waits on Meganeura's `InstanceNormalization` and integer inputs.
5. **Passthrough overlay and Depth API**: Blade work, owned by other
   sessions (TASKS.md items 1 and 5).

Blade and Meganeura are ours. Missing ops and XR features go upstream, with
their tests, not into workarounds here (AGENTS.md).
