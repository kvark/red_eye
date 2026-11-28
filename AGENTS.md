# AGENTS.md

MAR (movie augmented reality) puts visions from films over the world on
a Meta Quest 3/3S, each as an app of its own. Perception models run on the
headset, and overlays are drawn over the system's passthrough. It is Rust on
[Blade](https://github.com/kvark/blade) (GPU, OpenXR) and
[Meganeura](https://github.com/kvark/meganeura) (inference, training).
[DinoVision](https://github.com/kvark/dinovision) is the reference
integration of both on a Quest.

The first app is **RedEye**, T-800 style augmented vision: the world graded
red, objects detected, tracked, ranged and outlined, and a HUD in the
films' visual grammar, with targets named by voice. A second app, a green
"seeing the code" view inspired by a different film, is planned on the
same stack. Each app is an original take on a look; no film's assets,
glyph sets or characters are reproduced.

Read first: [docs/concept.md](docs/concept.md) (what we build: mechanics M1–M14)
and [docs/technology.md](docs/technology.md) (how: architecture, model plan,
frame budget, milestones, upstream work). [TASKS.md](TASKS.md) is the
backlog of what to explore next, with each idea's status; keep it current.
[HANDOFF.md](HANDOFF.md) is the guide for continuing on a machine with a
Quest 3: setup, first run, what to measure.

## Layout

A Cargo workspace. `mar-core` is everything the apps share and builds and
tests on a desktop; `mar-shell` is the Android XR app they run in; each app
is a crate implementing `mar_shell::Vision`, built into an APK of its own.

| Path | Responsibility |
|---|---|
| `mar-core/src/frame.rs` | Camera frame: I420, as Camera2 delivers it |
| `mar-core/src/source.rs` | `FrameSource`; test pattern and still image |
| `mar-core/src/grade.rs` | Colour grade of the camera's luminance (presets, LUT for `XR_FB_passthrough`) |
| `mar-core/src/render.rs`, `mar-core/src/shaders/` | Blade passes (grade, boxes, depth-contour wireframe), offscreen rendering with overlays; WGSL lives in `.wgsl` files |
| `mar-core/src/projection.rs` | Camera model from Camera2's calibration; placing camera-image overlays into each eye |
| `mar-core/src/perception/` | `Detector` trait, COCO labels, NMS, tracking, the `Observation` apps read |
| `mar-core/src/perception/yolox.rs` | YOLOX network (hand-built Meganeura graph), decode |
| `mar-core/src/perception/dfine.rs` | D-FINE through Meganeura's ONNX importer, decode |
| `mar-core/src/perception/preprocess.rs` | Each model's `InputSpec`; GPU preprocessing into the session's input buffer, and its CPU reference |
| `mar-core/src/perception/worker.rs` | Detector thread, and `Background` for other per-frame jobs (depth): drop frames while busy, publish the latest result |
| `mar-core/src/perception/dense.rs` | Per-pixel results in camera-image coordinates: instance masks, depth, depth-cut silhouettes |
| `mar-core/src/perception/depth.rs` | Depth Anything V2 Small through Meganeura's ONNX importer: relative depth |
| `mar-core/src/perception/environment.rs` | The headset's metric depth (`XR_META_environment_depth`): GPU resample, read-back, object ranges |
| `mar-core/src/voice/` | Voice commands: utterance detection, Moonshine-tiny through Meganeura's ONNX importer, decoding constrained to the command grammar |
| `mar-core/src/hud/` | HUD building blocks: the egui canvas composited per eye, monospace text as one quad batch (`grid.rs`, copied from Starcom; keep close to it), text effects |
| `mar-core/assets/fonts/` | Share Tech Mono (SIL OFL), the HUD face, with its licence |
| `mar-core/examples/` | `preview` (grade presets), `detect` (detector on an image), `voice` (commands in WAVs), `onnx_check` (an export against onnxruntime) |
| `mar-core/tests/` | GPU tests (`#[ignore]`, run on lavapipe or a real device), and models against onnxruntime references (need `ref/`) |
| `mar-shell/src/lib.rs` | The `Vision` trait and the XR app: session and frames, passthrough, environment depth, the model threads, focus, rates |
| `mar-shell/src/camera.rs` | The Camera2 passthrough camera and its calibration |
| `mar-shell/src/microphone.rs`, `listen.rs` | AAudio microphone, and the voice thread |
| `red-eye/src/app.rs` | RedEye as a `Vision`: grade, boxes, the outlined object's contours, ranges, targeting by voice |
| `red-eye/src/hud/` | RedEye's HUD: mechanics as time-driven state, named target and its match, the T2-style panels |
| `red-eye/examples/hud.rs` | Detector, tracker, a named target and the HUD over a photo, to PNGs |
| `tools/` | Python host tooling: model fetch and preparation, references |

## Commands

```sh
cargo test                                  # unit tests
pip install -r tools/requirements.txt && tools/fetch_yolox.sh && tools/fetch_dfine.sh   # models and references into ref/
tools/fetch_depth.sh && tools/fetch_depth.sh 224 168   # Depth Anything V2 Small, normalisation folded in
tools/fetch_moonshine.sh [track_the_bicycle.wav]   # Moonshine-tiny, 2 s window, and the command grammar
cargo run --release --example voice -- ref/moonshine_tiny_2s a.wav b.wav   # what each recording commands, and how sure
cargo test --release -- --include-ignored   # + GPU and detector tests (lavapipe: apt install mesa-vulkan-drivers)
cargo run --release --example preview -- [image] [out-dir]
cargo run --release --example detect -- ref/yolox_nano.safetensors ref/dog.jpg out/detect.png
cargo run --release --example detect -- ref/dfine_nano_640x480/model.onnx ref/dog.jpg out/detect.png
cargo run --release --example hud -- ref/dfine_nano_640x480/model.onnx ref/dog.jpg bicycle out
python tools/onnx_reference.py model.onnx ref/candidate 1,3,480,640 [image] [--simplify]   # evaluate a candidate export
cargo run --release --example onnx_check -- ref/candidate
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --target aarch64-linux-android -p mar-shell -p red-eye -- -D warnings
# Device (needs Android SDK 34, NDK r27, cargo-apk, and Blade cloned at ../blade):
cargo apk run --manifest-path red-eye/Cargo.toml --release --lib --no-logcat
adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA
adb shell pm grant rust.red_eye_xr com.oculus.permission.USE_SCENE   # the headset's depth, for ranges
adb shell pm grant rust.red_eye_xr android.permission.RECORD_AUDIO   # voice commands
adb shell mkdir -p /data/local/tmp/red_eye/moonshine/encoder /data/local/tmp/red_eye/moonshine/decoder
adb push ref/moonshine_tiny_2s/commands.txt /data/local/tmp/red_eye/moonshine/   # and encoder/, decoder/ model.onnx alike
adb shell mkdir -p /data/local/tmp/red_eye && adb push ref/yolox_nano.safetensors /data/local/tmp/red_eye/
adb push ref/dfine_nano_640x480/model.onnx /data/local/tmp/red_eye/dfine_nano.onnx   # preferred when present
adb push ref/depth_small_224x168/model.onnx /data/local/tmp/red_eye/depth_small.onnx   # the wireframe's depth
adb shell "echo motorcycle > /data/local/tmp/red_eye/target"   # name a target without voice
```

## Rules

- **Everything stays on the device.** The apps make no network calls and
  sends no telemetry. Weights are pushed with `adb` or bundled as assets, and
  are never committed.
- **Meganeura and Blade are ours.** When an op, kernel or XR feature is
  missing, add it upstream with that repo's tests instead of working around it
  here, and record the work in `docs/technology.md`.
- **YOLOX weights bind by position.** The official exports are anonymous,
  so `conv.NNN` is the NNN-th convolution in forward order. Any change to
  `yolox::Network` must keep the reference's order; `mar-core/tests/yolox.rs` catches
  a slip.
- **Pins move together.** `blade-graphics` must be the exact revision
  Meganeura pins, so the `Arc<Context>` types match; `blade-egui` and
  `blade-util` share that revision. `openxr` and `egui` must be the versions
  Blade uses. Bump them in one commit.
- **The display path never waits on inference.** The renderer draws the
  latest `Observation`; models run on a worker thread and frames are dropped,
  never queued.
- **The grade has one definition.** `Grade::map` is the reference. The
  shader mirrors it and `mar-core/tests/gpu.rs` enforces agreement. Uniform structs
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
