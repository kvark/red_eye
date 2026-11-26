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

Milestone 0, the skeleton:

- The red grade (T1, T2 and damaged presets) is shared between a WGSL pass
  and the `XR_FB_passthrough` colour map, and the two are verified to agree on
  the GPU.
- A headless preview renders the grades.
- An IoU tracker and the HUD text effects are in place.
- The XR app shows the graded left passthrough camera. It type-checks but has
  not yet run on a device.

```sh
cargo test -- --include-ignored
cargo run --release --example preview -- photo.jpg out
```

## License

MIT. Models and fonts have their own terms; see `docs/technology.md`.
