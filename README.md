# RedEye

T-800 augmented vision for Meta Quest 3: the world graded red, people and
vehicles detected, tracked and locked onto, and a HUD drawn in the visual
grammar of *The Terminator* and *Terminator 2*. Everything runs on the
headset, with no network connections.

Built on [Blade](https://github.com/kvark/blade) for GPU and OpenXR, and
[Meganeura](https://github.com/kvark/meganeura) for on-device inference.

- [docs/concept.md](docs/concept.md): the films' vision broken down into
  mechanics.
- [docs/technology.md](docs/technology.md): architecture, model plan, frame
  budget, milestones.
- [TASKS.md](TASKS.md): what to explore next, and where each idea stands.
- [HANDOFF.md](HANDOFF.md): continuing on a machine with a Quest 3.
- [AGENTS.md](AGENTS.md): layout, commands and rules for contributors.

## Status

- **Grade**: T1, T2 and damaged presets, shared between a WGSL pass and the
  `XR_FB_passthrough` colour map, and verified to agree on the GPU.
- **Detectors**: D-FINE-N (through Meganeura's ONNX importer) and YOLOX
  (a hand-built Meganeura graph), both Apache-2.0 on COCO's 80 classes,
  match their official models in onnxruntime. Preprocessing runs on the
  GPU; decoding and tracking run on a worker thread.
- **HUD**: egui with Starcom's monospace grid, in the films' layout. Name a
  class and it scans, then brackets the match and types "MATCH".
- **XR app**: shows the graded left passthrough camera with the HUD and
  tracked detections over it. It type-checks for Android but has not yet
  run on a headset.

```sh
pip install -r tools/requirements.txt && tools/fetch_yolox.sh && tools/fetch_dfine.sh
cargo test --release -- --include-ignored
cargo run --release --example detect -- ref/dfine_nano_640x480/model.onnx photo.jpg out/detect.png
```

## License

MIT. Models and fonts have their own terms; see `docs/technology.md`.
