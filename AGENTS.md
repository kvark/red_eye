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
| `src/perception/` | Detector trait, COCO labels, tracking, the `Observation` the HUD reads |
| `src/hud/` | Mechanics as time-driven state; text effects |
| `android-xr/` | OpenXR app (`cdylib`); `camera.rs` is the Camera2 passthrough camera |
| `examples/preview.rs` | Headless render of the grade presets to PNG |
| `tests/gpu.rs` | GPU tests (`#[ignore]`, run on lavapipe or a real device) |

## Commands

```sh
cargo test                                  # unit tests
cargo test -- --include-ignored             # + GPU tests (lavapipe: apt install mesa-vulkan-drivers)
cargo run --release --example preview -- [image] [out-dir]
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --target aarch64-linux-android -p red-eye-xr -- -D warnings
# Device (needs Android SDK 34, NDK r27, cargo-apk, and Blade cloned at ../blade):
cargo apk run --manifest-path android-xr/Cargo.toml --release --no-logcat
adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA
```

## Rules

- **Everything stays on the device.** The app makes no network calls and
  sends no telemetry. Weights are pushed with `adb` or bundled as assets, and
  are never committed.
- **Meganeura and Blade are ours.** When an op, kernel or XR feature is
  missing, add it upstream with that repo's tests instead of working around it
  here, and record the work in `docs/technology.md`.
- **Pins move together.** `blade-graphics` must be the exact revision
  Meganeura pins, so the `Arc<Context>` types match. `openxr` must be the
  version Blade uses. Bump them in one commit.
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
