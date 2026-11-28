//! Depth Anything V2 Small through Meganeura's ONNX importer against
//! onnxruntime.
//!
//! Needs a GPU (lavapipe is enough) and the prepared exports:
//!
//! ```text
//! pip install -r tools/requirements.txt
//! tools/fetch_depth.sh && tools/fetch_depth.sh 224 168
//! cargo test --release --test depth -- --include-ignored
//! ```

mod common;

use std::path::Path;

use mar_core::frame::{Eye, Frame};
use mar_core::perception::depth::DepthAnything;

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

fn matches_onnxruntime(size: [u32; 2]) {
    let dir = format!("ref/depth_small_{}x{}", size[0], size[1]);
    let dir = Path::new(&dir);
    let mut model = DepthAnything::new(&dir.join("model.onnx"), size, None).expect("model");
    let input = read_f32(&dir.join("input.bin"));
    let error = worst_relative_error(model.run(&input), &read_f32(&dir.join("output_0.bin")));
    println!("{size:?}: worst relative error {error:.2e}");
    assert!(error < 1e-3);
}

#[test]
#[ignore = "requires a GPU and ref/depth_small_336x252 (tools/fetch_depth.sh)"]
fn depth_matches_onnxruntime() {
    common::at_references();
    matches_onnxruntime([336, 252]);
}

#[test]
#[ignore = "requires a GPU and ref/depth_small_224x168 (tools/fetch_depth.sh 224 168)"]
fn small_depth_matches_onnxruntime() {
    common::at_references();
    matches_onnxruntime([224, 168]);
}

/// End to end through the public path: on the YOLOX demo photo the porch
/// floor and the dog are nearer (larger) than the parked truck and the
/// street behind it.
#[test]
#[ignore = "requires a GPU and ref/depth_small_224x168 (tools/fetch_depth.sh 224 168)"]
fn depth_orders_the_demo_photo() {
    common::at_references();
    let rgb = image::open("ref/dog.jpg").expect("ref/dog.jpg").to_rgb8();
    let frame = Frame::from_rgb8(rgb.width(), rgb.height(), rgb.as_raw(), Eye::Left);
    let size = [224, 168];
    let dir = format!("ref/depth_small_{}x{}", size[0], size[1]);
    let mut model =
        DepthAnything::new(&Path::new(&dir).join("model.onnx"), size, None).expect("model");
    let depth = model.run_frame(&frame);
    assert_eq!([depth.width, depth.height], size);
    // Mean over a region in normalised image coordinates.
    let mean = |x0: f32, x1: f32, y0: f32, y1: f32| {
        let (w, h) = (depth.width as f32, depth.height as f32);
        let mut sum = 0.0;
        let mut count = 0;
        for y in (y0 * h) as u32..(y1 * h) as u32 {
            for x in (x0 * w) as u32..(x1 * w) as u32 {
                sum += depth.values[(y * depth.width + x) as usize];
                count += 1;
            }
        }
        sum / count as f32
    };
    let dog = mean(0.18, 0.40, 0.55, 0.90);
    let floor = mean(0.0, 1.0, 0.93, 1.0);
    let truck = mean(0.62, 0.88, 0.14, 0.28);
    println!("dog {dog:.3}, floor {floor:.3}, truck {truck:.3}");
    assert!(
        dog > truck && floor > truck,
        "dog {dog}, floor {floor}, truck {truck}"
    );
}
