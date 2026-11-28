# MAR

Movie augmented reality: visions from films over the world on a Meta Quest
3/3S, each an app of its own. Perception runs on the headset with no
network connections, and overlays are drawn over the system's passthrough.

Built on [Blade](https://github.com/kvark/blade) for GPU and OpenXR, and
[Meganeura](https://github.com/kvark/meganeura) for on-device inference.

- `mar-core`: what every app shares, from camera frames to detectors,
  depth, voice and the overlay passes. Builds and tests on a desktop.
- `mar-shell`: the Android XR app they run in.
- `red-eye`: **RedEye**, T-800 style augmented vision.

A second app, a green "seeing the code" view, is planned on the same stack.

- [docs/concept.md](docs/concept.md): RedEye's vision broken down into
  mechanics.
- [docs/technology.md](docs/technology.md): architecture, model plan, frame
  budget, milestones.
- [TASKS.md](TASKS.md): what to explore next, and where each idea stands.
- [HANDOFF.md](HANDOFF.md): continuing on a machine with a Quest.
- [AGENTS.md](AGENTS.md): layout, commands and rules for contributors.

## RedEye

The world graded red by the compositor's passthrough. Objects are detected
(D-FINE-N or YOLOX), tracked, ranged with the headset's depth, and the one
you face is outlined with depth contours. A HUD in the visual grammar of
*The Terminator* and *Terminator 2* runs over it. Say "track the motorcycle"
and it scans, brackets the match and types "MATCH". On a Quest 3S it renders
at about 45 Hz beside detection at 10 Hz.

```sh
pip install -r tools/requirements.txt && tools/fetch_yolox.sh && tools/fetch_dfine.sh
cargo test --release -- --include-ignored
cargo run --release --example detect -- ref/dfine_nano_640x480/model.onnx photo.jpg out/detect.png
cargo apk run --manifest-path red-eye/Cargo.toml --release --lib --no-logcat
```

## License

MIT. Models and fonts have their own terms; see `docs/technology.md`.
