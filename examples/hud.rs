//! The HUD over a photo: detection, tracking, a named target, and the
//! T2-style panels, rendered headless at three moments.
//!
//! ```text
//! cargo run --release --example hud -- ref/dfine_nano_640x480/model.onnx ref/dog.jpg bicycle [out-dir] [depth.onnx]
//! ```
//!
//! Writes `<out-dir>/hud-scanning.png` (target named, nothing found yet),
//! `hud-match.png` (MATCH typing out) and `hud-locked.png` (settled). A
//! `.onnx` model runs D-FINE (prepared at 640×480), a `.safetensors` one
//! YOLOX-Nano. With a depth model (`tools/fetch_depth.sh 224 168`), the
//! tracks are cut out of their boxes by depth and drawn as a wireframe of
//! depth contours under the HUD.

use std::path::Path;

use red_eye::frame::{Eye, Frame};
use red_eye::grade::Grade;
use red_eye::hud::canvas::{Canvas, HudOverlay};
use red_eye::hud::{Hud, Telemetry};
use red_eye::perception::dense;
use red_eye::perception::depth::DepthAnything;
use red_eye::perception::dfine::Dfine;
use red_eye::perception::track::{Tracker, TrackerConfig};
use red_eye::perception::yolox::{Config, Yolox};
use red_eye::perception::{Detector, Label};
use red_eye::render::{Overlay, WireframeOverlay, WireframePass, WireframeStyle};

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, image, target, rest @ ..] = &args[..] else {
        eprintln!(
            "usage: hud <dfine.onnx | yolox.safetensors> <image> <target class> [out-dir] [depth.onnx]"
        );
        std::process::exit(2);
    };
    let out_dir = Path::new(rest.first().map_or("out", String::as_str)).to_path_buf();
    std::fs::create_dir_all(&out_dir).unwrap();
    let target = Label::from_name(&target.replace('_', " "))
        .unwrap_or_else(|| panic!("{target} is not a COCO class"));

    let rgb = image::open(image)
        .unwrap_or_else(|e| panic!("cannot open {image}: {e}"))
        .to_rgb8();
    let frame = Frame::from_rgb8(rgb.width(), rgb.height(), rgb.as_raw(), Eye::Left);

    let context = red_eye::init_context(None).expect("no GPU");
    let mut detector: Box<dyn Detector> = if model.ends_with(".onnx") {
        Box::new(Dfine::new(Path::new(model), [640, 480], Some(context.clone())).expect("D-FINE"))
    } else {
        let config = Config::nano().at_input(416, 320);
        Box::new(Yolox::new(config, Path::new(model), Some(context.clone())).expect("YOLOX"))
    };
    // A still photo stands in for a few camera frames, so tracks confirm.
    let mut tracker = Tracker::new(TrackerConfig::default());
    let start = std::time::Instant::now();
    for _ in 0..3 {
        tracker.update(&detector.detect(&frame));
    }
    let telemetry = Telemetry {
        // Offline: no display rate to report.
        render_hz: 0.0,
        detect_hz: 3.0 / start.elapsed().as_secs_f32(),
        latency_ms: start.elapsed().as_secs_f32() * 1e3 / 3.0,
    };
    let tracks = tracker.tracks().to_vec();

    // Depth for the wireframe, if a model was given.
    let depth = rest.get(1).map(|path| {
        let mut model =
            DepthAnything::new(Path::new(path), [224, 168], Some(context.clone())).expect("depth");
        model.run_frame(&frame)
    });
    let mut wireframe = WireframePass::new(&context, blade_graphics::TextureFormat::Rgba8Unorm);

    // The canvas covers the photo, laid out as on a 1280-wide headset view.
    let pixels_per_point = frame.width as f32 / 1280.0;
    let mut canvas = Canvas::new(
        &context,
        [frame.width, frame.height],
        pixels_per_point,
        blade_graphics::TextureFormat::Rgba8Unorm,
    );
    let mut hud = Hud::default();
    hud.set_target(Some(target), 0.0);

    // Nothing in view until 1.0 s, then the photo's tracks.
    let moments = [
        ("scanning", 0.6, false),
        ("match", 1.25, true),
        ("locked", 3.0, true),
    ];
    hud.targeting.update(&[], 0.0);
    for (name, now, found) in moments {
        if found {
            hud.targeting.update(&tracks, 1.0);
        }
        let shown: &[_] = if found { &tracks } else { &[] };
        // Outline the matched target, or every track while scanning.
        let matched = match hud.targeting.phase() {
            red_eye::hud::targeting::Phase::Match { track, .. } => Some(track),
            _ => None,
        };
        let boxes: Vec<_> = shown
            .iter()
            .filter(|t| t.is_confirmed(2) && (matched.is_none() || Some(t.id) == matched))
            .enumerate()
            .map(|(i, t)| (i as u8 + 1, t.rect))
            .collect();
        let silhouettes = depth
            .as_ref()
            .map(|depth| (dense::silhouettes(&boxes, depth, 320, 240), depth));
        let mut wire = silhouettes.as_ref().and_then(|(mask, depth)| {
            let (lo, hi) = dense::masked_range(mask, depth)?;
            Some(WireframeOverlay {
                pass: &mut wireframe,
                mask,
                depth,
                style: WireframeStyle {
                    spacing: ((hi - lo) / 8.0).max(1e-3),
                    highlight: if matched.is_some() { 1 } else { 0 },
                    dim: 1.0,
                    ..WireframeStyle::default()
                },
            })
        });
        let mut overlay = HudOverlay {
            canvas: &mut canvas,
            hud: &mut hud,
            now,
            tracks: shown,
            telemetry,
        };
        let pixels = red_eye::render::render_offscreen(
            &context,
            &frame,
            &Grade::t2(),
            now,
            &[],
            &mut match wire {
                Some(ref mut wire) => vec![wire as &mut dyn Overlay, &mut overlay],
                None => vec![&mut overlay as &mut dyn Overlay],
            },
        );
        let path = out_dir.join(format!("hud-{name}.png"));
        image::RgbaImage::from_raw(frame.width, frame.height, pixels)
            .unwrap()
            .save(&path)
            .unwrap();
        println!("wrote {} ({:?})", path.display(), hud.targeting.phase());
    }
    canvas.destroy(&context);
    wireframe.destroy(&context);
}
