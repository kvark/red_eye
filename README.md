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
- [AGENTS.md](AGENTS.md): layout, commands and rules for contributors.

## Status

- **Grade**: T1, T2 and damaged presets, shared between a WGSL pass and the
  `XR_FB_passthrough` colour map, and verified to agree on the GPU.
- **Detector**: YOLOX (Apache-2.0, COCO's 80 classes) runs as a Meganeura
  graph and matches the official ONNX model in onnxruntime. Preprocessing
  runs on the GPU; decoding, NMS and tracking run on a worker thread.
- **XR app**: shows the graded left passthrough camera with tracked
  detections outlined. It type-checks for Android but has not yet run on a
  headset.

```sh
pip install -r tools/requirements.txt && tools/fetch_yolox.sh
cargo test --release -- --include-ignored
cargo run --release --example detect -- ref/yolox_nano.safetensors photo.jpg out/detect.png
```

## License

MIT. Models and fonts have their own terms; see `docs/technology.md`.
