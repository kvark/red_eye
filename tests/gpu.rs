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
    let pixels = red_eye::render::render_graded(&context, &frame, &grade, 0.0, &[]);

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

/// The GPU preprocessing must produce exactly what the CPU reference does,
/// on a frame with real chroma, for every input the detectors use:
/// YOLOX's letterboxed BGR Focus layout in both letterbox directions, and
/// D-FINE's stretched RGB in `0..=1`.
#[test]
#[ignore = "requires a working GPU context"]
fn preprocess_matches_reference() {
    use blade_graphics as gpu;
    use red_eye::perception::preprocess::{self, GpuPreprocess};
    use red_eye::perception::{dfine, yolox};

    let context = red_eye::init_context(None).expect("no GPU");
    let (width, height) = (320u32, 240u32);
    let rgb: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                [
                    (x * 255 / width) as u8,
                    (y * 255 / height) as u8,
                    ((x ^ y) & 0xff) as u8,
                ]
            })
        })
        .collect();
    let frame = Frame::from_rgb8(width, height, &rgb, Eye::Left);

    let specs = [
        yolox::Config::nano().at_input(416, 320).input_spec(),
        yolox::Config::nano().input_spec(),
        dfine::input_spec([640, 480]),
    ];
    for spec in specs {
        let mut expected = Vec::new();
        let cpu_fit = preprocess::preprocess(&frame, &spec, &mut expected);

        let output = context.create_buffer(gpu::BufferDesc {
            name: "preprocess-output",
            size: (spec.len() * 4) as u64,
            memory: gpu::Memory::Shared,
        });
        let mut pre = GpuPreprocess::new(&context);
        let gpu_fit = pre.run(&context, &frame, &spec, output.into());
        pre.destroy(&context);
        assert_eq!(gpu_fit, cpu_fit);
        let got: &[f32] =
            unsafe { std::slice::from_raw_parts(output.data() as *const f32, spec.len()) };
        let worst = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (g - e).abs())
            .fold(0.0f32, f32::max);
        context.destroy_buffer(output);
        // A twentieth of one 0..=255 step, in the spec's own units.
        let bound = 0.05 * spec.value_scale;
        assert!(
            worst < bound,
            "{spec:?}: GPU preprocessing deviates by {worst}"
        );
    }
}

/// Box outlines land on the image coordinates they were given.
#[test]
#[ignore = "requires a working GPU context"]
fn boxes_outline_their_rect() {
    use red_eye::perception::Rect;
    use red_eye::render::BoxInstance;

    let context = red_eye::init_context(None).expect("no GPU");
    // Black frame; at 800 px the offscreen helper draws 2 px lines.
    let frame = Frame::new(800, 800, Eye::Left);
    let rect = Rect::new(0.25, 0.25, 0.75, 0.75);
    let boxes = [BoxInstance::new(&rect, [1.0, 1.0, 1.0, 1.0])];
    let pixels = red_eye::render::render_graded(&context, &frame, &Grade::t1(), 0.0, &boxes);
    let at = |x: usize, y: usize| &pixels[(y * 800 + x) * 4..][..3];
    assert_eq!(at(200, 400), [255, 255, 255], "left edge");
    assert_eq!(at(400, 600), [255, 255, 255], "bottom edge");
    assert!(
        at(400, 400).iter().all(|&c| c < 10),
        "inside stays graded black"
    );
    assert!(
        at(100, 100).iter().all(|&c| c < 10),
        "outside stays graded black"
    );
}

/// The HUD lands where its layout puts it: with a matched target, the state
/// word in the bottom-centre band and brackets at the track's corners, over
/// a frame that is otherwise graded black.
#[test]
#[ignore = "requires a working GPU context"]
fn hud_draws_match_and_brackets() {
    use red_eye::hud::canvas::{Canvas, HudOverlay};
    use red_eye::hud::{Hud, Telemetry};
    use red_eye::perception::track::Track;
    use red_eye::perception::{Label, Rect};

    let context = red_eye::init_context(None).expect("no GPU");
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
        red_eye::render::render_offscreen(
            &context,
            &frame,
            &Grade::t1(),
            0.0,
            &[],
            Some(&mut overlay),
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
