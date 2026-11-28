//! GPU tests. They need a Vulkan device; lavapipe is enough:
//!
//! ```text
//! cargo test -- --include-ignored
//! ```

use mar_core::frame::{Eye, Frame};
use mar_core::grade::Grade;

/// The WGSL grade and `Grade::map` must agree, or the passthrough colour map
/// and the self-rendered fallback would show two different looks.
#[test]
#[ignore = "requires a working GPU context"]
fn shader_grade_matches_reference() {
    let context = mar_core::init_context(None).expect("no GPU");
    // One row per luminance level, two pixels wide so that 4:2:0 is happy
    // and linear filtering at the texel centre returns the exact value.
    let (width, height) = (2u32, 256u32);
    let mut frame = Frame::new(width, height, Eye::Left);
    for (y, row) in frame.luma_mut().chunks_mut(width as usize).enumerate() {
        row.fill(y as u8);
    }
    let grade = Grade::default().tone_only();
    let pixels = mar_core::render::render_graded(&context, &frame, &grade, 0.0, &[]);

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
    use mar_core::perception::preprocess::{self, GpuPreprocess};
    use mar_core::perception::{dfine, yolox};

    let context = mar_core::init_context(None).expect("no GPU");
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
    use mar_core::perception::Rect;
    use mar_core::render::BoxInstance;

    let context = mar_core::init_context(None).expect("no GPU");
    // Black frame; at 800 px the offscreen helper draws 2 px lines.
    let frame = Frame::new(800, 800, Eye::Left);
    let rect = Rect::new(0.25, 0.25, 0.75, 0.75);
    let boxes = [BoxInstance::new(&rect, [1.0, 1.0, 1.0, 1.0])];
    let pixels = mar_core::render::render_graded(&context, &frame, &Grade::t1(), 0.0, &boxes);
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

/// Depth contours stay inside the mask and wrap the shape: a dome of depth
/// inside a disc mask draws rings, with gaps between them, an outline at
/// the disc's edge, and nothing outside it.
#[test]
#[ignore = "requires a working GPU context"]
fn wireframe_draws_contours_inside_the_mask() {
    use mar_core::perception::dense::{DepthMap, Mask};
    use mar_core::render::{WireframeOverlay, WireframePass, WireframeStyle};

    let context = mar_core::init_context(None).expect("no GPU");
    let (w, h) = (320u32, 320u32);
    let frame = Frame::new(w, h, Eye::Left);
    let inside = |u: f32, v: f32| (u - 0.5).powi(2) + (v - 0.5).powi(2) < 0.3f32.powi(2);
    // Mask and depth at their own, coarser resolutions, as models give them.
    let mut mask = Mask::new(80, 80);
    mask.paint(1, inside);
    let mut depth = DepthMap::new(40, 40);
    depth.fill(|u, v| {
        let r2 = (u - 0.5).powi(2) + (v - 0.5).powi(2);
        2.0 - (0.35f32.powi(2) - r2).max(0.0).sqrt()
    });
    let mut pass = WireframePass::new(&context, blade_graphics::TextureFormat::Rgba8Unorm);
    let mut overlay = WireframeOverlay {
        pass: &mut pass,
        mask: &mask,
        depth: &depth,
        style: WireframeStyle {
            spacing: 0.02,
            ..Default::default()
        },
    };
    let pixels = mar_core::render::render_offscreen(
        &context,
        &frame,
        &Grade::t1(),
        0.0,
        &[],
        &mut [&mut overlay],
    );
    pass.destroy(&context);

    let (mut lit, mut dark, mut outside_lit, mut edge_lit) = (0, 0, 0, 0);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let value = pixels[((y * w + x) * 4) as usize];
            let r = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
            if r > 0.32 {
                outside_lit += (value > 30) as u32;
            } else if r < 0.27 {
                lit += (value > 150) as u32;
                dark += (value < 40) as u32;
            } else if (0.295..0.30).contains(&r) {
                edge_lit += (value > 150) as u32;
            }
        }
    }
    assert_eq!(outside_lit, 0, "the wireframe leaks outside the mask");
    assert!(lit > 500, "no contour lines inside: {lit}");
    // Rings bunch up towards the dome's edge, so a fair share of the disc
    // is lit; a fill would light all of it.
    let lit_share = lit as f32 / (lit + dark) as f32;
    assert!(
        lit_share < 0.4,
        "contours should be lines, not a fill: {lit_share}"
    );
    assert!(edge_lit > 20, "no outline: {edge_lit}");
    // Separate rings: dark-to-lit crossings along the row through the centre.
    let row = h / 2;
    let at = |x: u32| pixels[((row * w + x) * 4) as usize];
    let crossings = (1..w).filter(|&x| at(x - 1) < 100 && at(x) >= 100).count();
    assert!(crossings >= 6, "too few rings crossed: {crossings}");
}

/// The environment depth resample reads both layers of a depth texture
/// onto its grid by nearest texel. The Quest's is 16-bit; the shader reads
/// any depth format alike, and 32-bit float is the one every Blade has.
#[test]
#[ignore = "requires a working GPU context"]
fn environment_depth_resamples_both_layers() {
    use blade_graphics as gpu;
    use mar_core::perception::environment::{DepthView, GRID, Resample};
    use mar_core::projection::Pose;

    let context = mar_core::init_context(None).expect("no GPU");
    let (width, height) = (64u32, 48u32);
    // Each texel holds its own coordinates and layer, so any misplaced
    // read shows.
    let value = |x: u32, y: u32, layer: u32| (layer * 20000 + y * 300 + x * 3) as f32 / 65535.0;
    let format = gpu::TextureFormat::Depth32Float;
    let texture = context.create_texture(gpu::TextureDesc {
        name: "environment-depth",
        format,
        size: gpu::Extent {
            width,
            height,
            depth: 1,
        },
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 2,
        mip_level_count: 1,
        usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
        sample_count: 1,
        external: None,
    });
    let view = context.create_texture_view(
        texture,
        gpu::TextureViewDesc {
            name: "environment-depth",
            format,
            dimension: gpu::ViewDimension::D2Array,
            subresources: &Default::default(),
        },
    );
    let layer_bytes = (width * height * 4) as u64;
    let staging = context.create_buffer(gpu::BufferDesc {
        name: "environment-depth-staging",
        size: 2 * layer_bytes,
        memory: gpu::Memory::Shared,
    });
    let texels: Vec<f32> = (0..2)
        .flat_map(|l| (0..height).flat_map(move |y| (0..width).map(move |x| value(x, y, l))))
        .collect();
    unsafe {
        std::ptr::copy_nonoverlapping(
            texels.as_ptr() as *const u8,
            staging.data(),
            texels.len() * 4,
        );
    }

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "environment-depth",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    encoder.init_texture(texture);
    {
        let mut transfer = encoder.transfer("upload");
        for layer in 0..2 {
            transfer.copy_buffer_to_texture(
                staging.at(layer as u64 * layer_bytes),
                width * 4,
                gpu::TexturePiece {
                    texture,
                    mip_level: 0,
                    array_layer: layer,
                    origin: [0; 3],
                },
                gpu::Extent {
                    width,
                    height,
                    depth: 1,
                },
            );
        }
    }
    let view_of = DepthView {
        pose: Pose::IDENTITY,
        fov: [-0.5, 0.5, 0.5, -0.5],
    };
    let mut resample = Resample::new(&context);
    resample.record(&mut encoder, view, [view_of; 2], 0.1, f32::INFINITY);
    let sync = context.submit(&mut encoder);
    let _ = context.wait_for(&sync, !0);
    let depth = resample.read().expect("a recorded resample");
    assert!(resample.read().is_none(), "read twice");

    let [gw, gh] = GRID;
    let mut worst = 0.0f32;
    for layer in 0..2 {
        for gy in 0..gh {
            for gx in 0..gw {
                let x = ((gx as f32 + 0.5) * width as f32 / gw as f32) as u32;
                let y = ((gy as f32 + 0.5) * height as f32 / gh as f32) as u32;
                let expected = value(x, y, layer);
                let got = depth.raw[((layer * gh + gy) * gw + gx) as usize];
                worst = worst.max((got - expected).abs());
            }
        }
    }
    resample.destroy(&context);
    context.destroy_command_encoder(&mut encoder);
    context.destroy_buffer(staging);
    context.destroy_texture_view(view);
    context.destroy_texture(texture);
    assert!(worst < 1e-6, "resampled depth deviates by {worst}");
}
