//! D-FINE through Meganeura's ONNX importer against onnxruntime.
//!
//! Needs a GPU (lavapipe is enough) and the prepared export:
//!
//! ```text
//! pip install -r tools/requirements.txt
//! tools/fetch_dfine.sh
//! cargo test --release --test dfine -- --include-ignored
//! ```

use std::path::Path;

use red_eye::frame::{Eye, Frame};
use red_eye::perception::Detector;
use red_eye::perception::dfine::Dfine;

const DIR: &str = "ref/dfine_nano_640x480";
const SIZE: [u32; 2] = [640, 480];

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytemuck::pod_collect_to_vec(&bytes)
}

fn worst_relative_error(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len());
    got.iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs() / (1.0 + e.abs()))
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "requires a GPU and ref/dfine_nano_640x480 (tools/fetch_dfine.sh)"]
fn dfine_matches_onnxruntime() {
    let dir = Path::new(DIR);
    let mut model = Dfine::new(&dir.join("model.onnx"), SIZE, None).expect("model");
    assert_eq!(model.queries(), 300);
    let input = read_f32(&dir.join("input.bin"));
    let (logits, boxes) = model.run(&input);
    let logits_error = worst_relative_error(logits, &read_f32(&dir.join("output_0.bin")));
    let boxes_error = worst_relative_error(boxes, &read_f32(&dir.join("output_1.bin")));
    println!("worst relative error: logits {logits_error:.2e}, boxes {boxes_error:.2e}");
    assert!(logits_error < 1e-3 && boxes_error < 1e-3);
}

/// End to end through the public path, on the YOLOX demo photo that
/// YOLOX-Nano scores 0.80-0.85.
#[test]
#[ignore = "requires a GPU and ref/dfine_nano_640x480 (tools/fetch_dfine.sh)"]
fn dfine_detects_on_a_frame() {
    let rgb = image::open("ref/dog.jpg").expect("ref/dog.jpg").to_rgb8();
    let frame = Frame::from_rgb8(rgb.width(), rgb.height(), rgb.as_raw(), Eye::Left);
    let mut model = Dfine::new(&Path::new(DIR).join("model.onnx"), SIZE, None).expect("model");
    let dets = model.detect(&frame);
    for d in &dets {
        println!("{:>8} {:.2} {:?}", d.label.name(), d.score, d.rect);
    }
    let find = |name: &str| dets.iter().find(|d| d.label.name() == name);
    let dog = find("dog").expect("no dog");
    let bicycle = find("bicycle").expect("no bicycle");
    assert!(dog.score > 0.85 && bicycle.score > 0.85, "{dets:?}");
    // The dog sits in the lower-left, the bicycle across the middle.
    assert!(
        dog.rect.center()[0] < 0.4 && dog.rect.center()[1] > 0.5,
        "{dog:?}"
    );
    assert!(
        (0.3..0.6).contains(&bicycle.rect.center()[0]),
        "{bicycle:?}"
    );
    assert!(
        find("car").or(find("truck")).is_some(),
        "the parked vehicle is missing: {dets:?}"
    );
}
