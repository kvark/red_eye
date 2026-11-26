//! YOLOX through Meganeura against the official ONNX model in onnxruntime.
//!
//! Needs a GPU (lavapipe is enough) and the reference files, which are
//! large and not redistributed here:
//!
//! ```text
//! pip install -r tools/requirements.txt
//! tools/fetch_yolox.sh
//! cargo test --release --test yolox -- --include-ignored
//! ```

use std::path::Path;

use red_eye::perception::yolox::{self, Config, Yolox};
use red_eye::perception::{Label, nms};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytemuck::pod_collect_to_vec(&bytes)
}

fn check_against_reference(variant: &str, config: Config, reference: &str) {
    let weights = format!("ref/yolox_{variant}.safetensors");
    let mut model = Yolox::new(config.clone(), Path::new(&weights), None).expect("model");
    let input = read_f32(&Path::new(reference).join("input.bin"));
    let expected = read_f32(&Path::new(reference).join("output.bin"));
    assert_eq!(input.len(), config.input_len());
    let got = yolox::to_anchor_major(model.run(&input), &config);
    assert_eq!(got.len(), expected.len());

    // Box offsets are logits of order 1-10, the rest are probabilities.
    let mut worst = 0.0f32;
    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        let error = (g - e).abs() / (1.0 + e.abs());
        if error > worst {
            worst = error;
        }
        assert!(
            error < 2e-3,
            "anchor {} channel {}: meganeura {g} vs onnxruntime {e}",
            i / config.outputs_per_anchor(),
            i % config.outputs_per_anchor()
        );
    }
    println!("{variant} {reference}: worst relative error {worst:.2e}");
}

#[test]
#[ignore = "requires a GPU and ref/ YOLOX files (see the module docs)"]
fn nano_matches_onnxruntime() {
    check_against_reference("nano", Config::nano(), "ref/yolox_nano_reference");
    check_against_reference("nano", Config::nano(), "ref/yolox_nano_dog");
}

#[test]
#[ignore = "requires a GPU and ref/ YOLOX files (see the module docs)"]
fn tiny_matches_onnxruntime() {
    check_against_reference("tiny", Config::tiny(), "ref/yolox_tiny_reference");
}

/// The classic YOLOX demo picture: a dog, a bicycle and a parked vehicle.
#[test]
#[ignore = "requires a GPU and ref/ YOLOX files (see the module docs)"]
fn nano_finds_the_dog_and_the_bicycle() {
    let config = Config::nano();
    let mut model = Yolox::new(
        config.clone(),
        Path::new("ref/yolox_nano.safetensors"),
        None,
    )
    .expect("model");
    let input = read_f32(Path::new("ref/yolox_nano_dog/input.bin"));
    let output = model.run(&input).to_vec();
    // The reference letterboxed a 768×576 picture into 416².
    let letterbox = yolox::Letterbox::new([768, 576], config.input_size);
    let mut dets = yolox::decode(&output, &config, &letterbox, 0.35);
    nms(&mut dets, 0.45);
    let names: Vec<_> = dets.iter().map(|d| d.label.name()).collect();
    println!("{names:?}");
    for wanted in ["dog", "bicycle"] {
        assert!(names.contains(&wanted), "no {wanted} in {names:?}");
    }
    let dog = dets.iter().find(|d| d.label == Label(16)).unwrap();
    // The dog sits in the lower-left of the picture.
    assert!(
        dog.rect.center()[0] < 0.4 && dog.rect.center()[1] > 0.5,
        "{dog:?}"
    );
}

/// End to end through the public path: an RGB photo becomes an I420 frame,
/// is preprocessed on the GPU, run, decoded and suppressed.
#[test]
#[ignore = "requires a GPU and ref/ YOLOX files (see the module docs)"]
fn detector_on_a_frame() {
    use red_eye::frame::{Eye, Frame};
    use red_eye::perception::Detector;

    let rgb = image::open("ref/dog.jpg").expect("ref/dog.jpg").to_rgb8();
    let frame = Frame::from_rgb8(rgb.width(), rgb.height(), rgb.as_raw(), Eye::Left);
    // A 4:3 input, as on the headset.
    let config = Config::nano().at_input(416, 320);
    let mut model =
        Yolox::new(config, Path::new("ref/yolox_nano.safetensors"), None).expect("model");
    let dets = model.detect(&frame);
    for d in &dets {
        println!("{:>8} {:.2} {:?}", d.label.name(), d.score, d.rect);
    }
    let names: Vec<_> = dets.iter().map(|d| d.label.name()).collect();
    for wanted in ["dog", "bicycle"] {
        assert!(names.contains(&wanted), "no {wanted} in {names:?}");
    }
}
