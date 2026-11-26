//! Run a detector on an image and render the graded view with boxes.
//!
//! ```text
//! cargo run --release --example detect -- ref/yolox_nano.safetensors photo.jpg [out.png] [nano|tiny]
//! cargo run --release --example detect -- ref/dfine_nano_640x480/model.onnx photo.jpg [out.png]
//! ```
//!
//! A `.safetensors` checkpoint (`tools/fetch_yolox.sh`) runs YOLOX; an
//! `.onnx` model (`tools/fetch_dfine.sh`, prepared at 640×480) runs D-FINE.
//! The image is turned into the same I420 frame the headset camera
//! produces, so this exercises the device path: GPU preprocessing, the
//! network, decode.

use std::path::Path;
use std::time::Instant;

use red_eye::frame::{Eye, Frame};
use red_eye::grade::Grade;
use red_eye::perception::Detector;
use red_eye::perception::dfine::Dfine;
use red_eye::perception::yolox::{Config, Yolox};
use red_eye::render::BoxInstance;

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (weights, image) = match &args[..] {
        [w, i, ..] => (w.clone(), i.clone()),
        _ => {
            eprintln!(
                "usage: detect <yolox.safetensors | dfine.onnx> <image> [out.png] [nano|tiny]"
            );
            std::process::exit(2);
        }
    };
    let output = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "out/detect.png".into());
    let config = match args.get(3).map(String::as_str) {
        Some("tiny") => Config::tiny(),
        _ => Config::nano(),
    }
    .at_input(416, 320);

    let rgb = image::open(&image)
        .unwrap_or_else(|e| panic!("cannot open {image}: {e}"))
        .to_rgb8();
    let frame = Frame::from_rgb8(rgb.width(), rgb.height(), rgb.as_raw(), Eye::Left);

    // One context for the detector and the renderer, as on the headset.
    let context = red_eye::init_context(None).expect("no GPU");
    let mut detector: Box<dyn Detector> = if weights.ends_with(".onnx") {
        Box::new(
            Dfine::new(Path::new(&weights), [640, 480], Some(context.clone())).expect("D-FINE"),
        )
    } else {
        Box::new(Yolox::new(config, Path::new(&weights), Some(context.clone())).expect("YOLOX"))
    };
    // The first run includes pipeline warm-up; time the second.
    detector.detect(&frame);
    let start = Instant::now();
    let detections = detector.detect(&frame);
    println!(
        "{} detections in {:.1} ms",
        detections.len(),
        start.elapsed().as_secs_f64() * 1e3
    );
    for d in &detections {
        println!(
            "  {:<14} {:.2}  [{:.3}, {:.3}] - [{:.3}, {:.3}]",
            d.label.name().to_uppercase(),
            d.score,
            d.rect.min[0],
            d.rect.min[1],
            d.rect.max[0],
            d.rect.max[1]
        );
    }

    let boxes: Vec<_> = detections
        .iter()
        .map(|d| BoxInstance::new(&d.rect, red_eye::hud::outline_color(d.label.category())))
        .collect();
    let pixels = red_eye::render::render_graded(&context, &frame, &Grade::t2(), 0.0, &boxes);
    if let Some(dir) = Path::new(&output).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    image::RgbaImage::from_raw(frame.width, frame.height, pixels)
        .unwrap()
        .save(&output)
        .unwrap();
    println!("wrote {output}");
}
