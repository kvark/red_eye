# AGENTS.md

RedEye is T-800 style augmented vision for Meta Quest 3. Passthrough is
graded red, perception models run on the headset, and a HUD in the films'
visual grammar is drawn over the world. It is Rust on
[Blade](https://github.com/kvark/blade) (GPU, OpenXR) and
[Meganeura](https://github.com/kvark/meganeura) (inference, training).
[DinoVision](https://github.com/kvark/dinovision) is the reference
integration of both on a Quest.

Read first: [docs/concept.md](docs/concept.md) (what we build: mechanics M1–M14)
and [docs/technology.md](docs/technology.md) (how: architecture, model plan,
frame budget, milestones, upstream work).

## Layout

| Path | Responsibility |
|---|---|
| `src/frame.rs` | Camera frame: I420, as Camera2 delivers it |
| `src/source.rs` | `FrameSource`; test pattern and still image |
| `src/grade.rs` | Red grade (presets, LUT for `XR_FB_passthrough`) |
| `src/render.rs`, `src/shaders/` | Blade passes; WGSL lives in `.wgsl` files |
| `src/perception/` | `Detector` trait, COCO labels, NMS, tracking, the `Observation` the HUD reads |
| `src/perception/yolox.rs` | YOLOX network (hand-built Meganeura graph), decode |
| `src/perception/dfine.rs` | D-FINE through Meganeura's ONNX importer, decode |
| `src/perception/preprocess.rs` | Each model's `InputSpec`; GPU preprocessing into the session's input buffer, and its CPU reference |
| `src/perception/worker.rs` | Detector thread: drops frames while busy, publishes observations |
| `src/hud/` | Mechanics as time-driven state; text effects |
| `src/hud/canvas.rs` | egui laid out once per frame into an offscreen texture, composited per eye |
| `src/hud/grid.rs` | Monospace text as one quad batch, copied from Starcom (keep close to it) |
| `src/hud/targeting.rs`, `src/hud/view.rs` | Named target and its match; the T2-style panels and MATCH |
| `assets/fonts/` | Share Tech Mono (SIL OFL), the HUD face, with its licence |
| `android-xr/` | OpenXR app (`cdylib`); `camera.rs` is the Camera2 passthrough camera |
| `examples/preview.rs` | Headless render of the grade presets to PNG |
| `examples/detect.rs` | Detector on an image, graded view with boxes to PNG |
| `examples/hud.rs` | Detector, tracker, a named target and the HUD over a photo, to PNGs |
| `tests/gpu.rs` | GPU tests (`#[ignore]`, run on lavapipe or a real device) |
| `tests/yolox.rs`, `tests/dfine.rs` | Detectors against onnxruntime references (need `ref/`) |
| `examples/onnx_check.rs` | An ONNX export through Meganeura's importer against onnxruntime |
| `tools/` | Python host tooling: model fetch and preparation, references (`fetch_yolox.sh`, `fetch_dfine.sh`, `onnx_reference.py`) |

## Commands

```sh
cargo test                                  # unit tests
pip install -r tools/requirements.txt && tools/fetch_yolox.sh && tools/fetch_dfine.sh   # models and references into ref/
cargo test --release -- --include-ignored   # + GPU and detector tests (lavapipe: apt install mesa-vulkan-drivers)
cargo run --release --example preview -- [image] [out-dir]
cargo run --release --example detect -- ref/yolox_nano.safetensors ref/dog.jpg out/detect.png
cargo run --release --example detect -- ref/dfine_nano_640x480/model.onnx ref/dog.jpg out/detect.png
cargo run --release --example hud -- ref/dfine_nano_640x480/model.onnx ref/dog.jpg bicycle out
python tools/onnx_reference.py model.onnx ref/candidate 1,3,480,640 [image] [--simplify]   # evaluate a candidate export
cargo run --release --example onnx_check -- ref/candidate
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --target aarch64-linux-android -p red-eye-xr -- -D warnings
# Device (needs Android SDK 34, NDK r27, cargo-apk, and Blade cloned at ../blade):
cargo apk run --manifest-path android-xr/Cargo.toml --release --no-logcat
adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA
adb shell mkdir -p /data/local/tmp/red_eye && adb push ref/yolox_nano.safetensors /data/local/tmp/red_eye/
adb push ref/dfine_nano_640x480/model.onnx /data/local/tmp/red_eye/dfine_nano.onnx   # preferred when present
adb shell "echo motorcycle > /data/local/tmp/red_eye/target"   # name a target until voice commands land
```

## Rules

- **Everything stays on the device.** The app makes no network calls and
  sends no telemetry. Weights are pushed with `adb` or bundled as assets, and
  are never committed.
- **Meganeura and Blade are ours.** When an op, kernel or XR feature is
  missing, add it upstream with that repo's tests instead of working around it
  here, and record the work in `docs/technology.md`.
- **YOLOX weights bind by position.** The official exports are anonymous,
  so `conv.NNN` is the NNN-th convolution in forward order. Any change to
  `yolox::Network` must keep the reference's order; `tests/yolox.rs` catches
  a slip.
- **Pins move together.** `blade-graphics` must be the exact revision
  Meganeura pins, so the `Arc<Context>` types match; `blade-egui` and
  `blade-util` share that revision. `openxr` and `egui` must be the versions
  Blade uses. Bump them in one commit.
- **The display path never waits on inference.** The renderer draws the
  latest `Observation`; models run on a worker thread and frames are dropped,
  never queued.
- **The grade has one definition.** `Grade::map` is the reference. The
  shader mirrors it and `tests/gpu.rs` enforces agreement. Uniform structs
  use only 16-byte (`vec4`) members.
- **Licences.** Only Apache/MIT/BSD-class models and fonts. AGPL (Ultralytics),
  GPL and non-commercial weights are out. DINOv3 derivatives carry its
  licence (no military or weapons use; ship the licence). List every model's
  licence in `docs/technology.md`.
- **Ethics.** No recognition of strangers' faces (enrolled, consenting users
  only). Never label a real person as a threat. Release the camera when the
  app loses focus.
- **Tests.** Pure logic gets unit tests next to the code. GPU tests are
  `#[ignore = "requires a working GPU context"]` and must pass on lavapipe.
  Device-only behaviour is documented with the `adb` steps that exercise it.
- **Style.** Doc comments say *why*, not just what. `cargo fmt`, and clippy
  clean on host and `aarch64-linux-android`. Commit subjects are imperative
  and sentence case with no prefix; the body explains why.
