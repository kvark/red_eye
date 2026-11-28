//! RedEye's HUD on the GPU. It needs a Vulkan device; lavapipe is enough:
//!
//! ```text
//! cargo test -p red-eye -- --include-ignored
//! ```

use mar_core::frame::{Eye, Frame};
use mar_core::grade::Grade;

/// The HUD lands where its layout puts it: with a matched target, the state
/// word in the bottom-centre band and brackets at the track's corners, over
/// a frame that is otherwise graded black.
#[test]
#[ignore = "requires a working GPU context"]
fn hud_draws_match_and_brackets() {
    use mar_core::hud::canvas::Canvas;
    use mar_core::perception::track::Track;
    use mar_core::perception::{Label, Rect};
    use red_eye::hud::{Hud, HudOverlay, Telemetry};

    let context = mar_core::init_context(None).expect("no GPU");
    let (w, h) = (640u32, 480u32);
    let frame = Frame::new(w, h, Eye::Left);
    let bicycle = Label::from_name("bicycle").unwrap();
    let tracks = [Track {
        id: 7,
        label: bicycle,
        rect: Rect::new(0.25, 0.25, 0.5, 0.6),
        score: 0.9,
        hits: 5,
        misses: 0,
        age: 5,
    }];
    let mut canvas = Canvas::new(
        &context,
        [w, h],
        w as f32 / 1280.0,
        blade_graphics::TextureFormat::Rgba8Unorm,
    );
    let mut hud = Hud::default();
    hud.set_target(Some(bicycle), 0.0);
    let render = |hud: &mut Hud, canvas: &mut Canvas| {
        let mut overlay = HudOverlay {
            canvas,
            hud,
            // Long enough after the match for every line to have typed out.
            now: 5.0,
            tracks: &tracks,
            telemetry: Telemetry::default(),
        };
        mar_core::render::render_offscreen(
            &context,
            &frame,
            &Grade::t1(),
            0.0,
            &[],
            &mut [&mut overlay],
        )
    };
    let pixels = render(&mut hud, &mut canvas);
    canvas.destroy(&context);

    let bright = |x0: f32, x1: f32, y0: f32, y1: f32| {
        let mut count = 0;
        for y in (y0 * h as f32) as u32..(y1 * h as f32) as u32 {
            for x in (x0 * w as f32) as u32..(x1 * w as f32) as u32 {
                let p = &pixels[((y * w + x) * 4) as usize..][..3];
                if p.iter().all(|&c| c > 200) {
                    count += 1;
                }
            }
        }
        count
    };
    // "MATCH" and its cursor, bottom centre.
    assert!(bright(0.3, 0.7, 0.80, 0.92) > 300, "no state word");
    // The bracket at the track's top-left corner.
    assert!(bright(0.24, 0.30, 0.24, 0.27) > 20, "no bracket");
    // Inside the track, away from the HUD's panels: untouched black.
    assert_eq!(
        bright(0.30, 0.45, 0.30, 0.55),
        0,
        "the HUD leaks into the middle"
    );
}
