//! Render one frame through each grade preset, headless, and save PNGs.
//!
//! ```text
//! cargo run --release --example preview -- [input.jpg] [output-dir]
//! ```
//!
//! Without an input it renders the test pattern. Writes
//! `<output-dir>/preview-{t1,t2,damaged}.png` (default `out/`). Runs on any
//! Vulkan device, including lavapipe, so it doubles as a smoke test of the
//! render path.

use mar_core::frame::{Eye, Frame};
use mar_core::grade::Grade;
use mar_core::source::{FrameSource, TestPattern};

fn main() {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let frame = match args.next().filter(|a| a != "-") {
        Some(path) => {
            let rgb = image::open(&path)
                .unwrap_or_else(|e| panic!("cannot open {path}: {e}"))
                .to_rgb8();
            Frame::from_rgb8(rgb.width(), rgb.height(), rgb.as_raw(), Eye::Left)
        }
        None => {
            let [w, h] = TestPattern::CAMERA_SIZE;
            TestPattern::new(w, h).next_frame().unwrap().clone()
        }
    };
    let out_dir = std::path::PathBuf::from(args.next().unwrap_or_else(|| "out".into()));
    std::fs::create_dir_all(&out_dir).unwrap();

    let context = mar_core::init_context(None).expect("no GPU");
    for (name, grade) in [
        ("t1", Grade::t1()),
        ("t2", Grade::t2()),
        ("damaged", Grade::damaged()),
    ] {
        let pixels = mar_core::render::render_graded(&context, &frame, &grade, 0.0, &[]);
        let path = out_dir.join(format!("preview-{name}.png"));
        image::RgbaImage::from_raw(frame.width, frame.height, pixels)
            .unwrap()
            .save(&path)
            .unwrap();
        println!("wrote {}", path.display());
    }
}
