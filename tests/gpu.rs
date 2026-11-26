//! GPU tests. They need a Vulkan device; lavapipe is enough:
//!
//! ```text
//! cargo test -- --include-ignored
//! ```

use red_eye::frame::{Eye, Frame};
use red_eye::grade::Grade;

/// The WGSL grade and `Grade::map` must agree, or the passthrough colour map
/// and the self-rendered fallback would show two different looks.
#[test]
#[ignore = "requires a working GPU context"]
fn shader_grade_matches_reference() {
    let context = red_eye::init_context(None).expect("no GPU");
    // One row per luminance level, two pixels wide so that 4:2:0 is happy
    // and linear filtering at the texel centre returns the exact value.
    let (width, height) = (2u32, 256u32);
    let mut frame = Frame::new(width, height, Eye::Left);
    for (y, row) in frame.luma_mut().chunks_mut(width as usize).enumerate() {
        row.fill(y as u8);
    }
    let grade = Grade::default().tone_only();
    let pixels = red_eye::render::render_graded(&context, &frame, &grade, 0.0);

    let mut worst = 0.0f32;
    for y in 0..height as usize {
        let expected = grade.map(y as f32 / 255.0);
        let got = &pixels[y * width as usize * 4..][..3];
        for c in 0..3 {
            worst = worst.max((got[c] as f32 / 255.0 - expected[c]).abs());
        }
    }
    assert!(
        worst <= 1.5 / 255.0,
        "shader deviates from reference by {worst}"
    );
}
