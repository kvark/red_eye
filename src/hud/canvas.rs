//! The HUD canvas: egui, laid out once per frame into an offscreen texture.
//!
//! egui owns fonts, glyph rasterisation and layout; blade-egui paints its
//! meshes, and [`super::grid`] paints the monospace text in batches between
//! them. The result lands in one premultiplied-alpha sRGB texture, which
//! [`Canvas::composite`] then places into any target: the preview image, or
//! each eye of the headset. Nothing is laid out or painted twice for
//! stereo, and the same texture can later become a compositor quad layer.

use blade_egui as be;
use blade_graphics as gpu;

use super::grid;
use crate::render::Placement;

/// The HUD's monospace face: Share Tech Mono (Carrois Type Design, SIL OFL
/// 1.1, `assets/fonts/OFL.txt`). Square, technical capitals in the spirit
/// of the films' readouts.
const FONT: &[u8] = include_bytes!("../../assets/fonts/ShareTechMono-Regular.ttf");
const FONT_NAME: &str = "Share Tech Mono";

/// The canvas renders in sRGB, as egui and the grid expect; sampling it
/// returns linear values.
const FORMAT: gpu::TextureFormat = gpu::TextureFormat::Rgba8UnormSrgb;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    placement: [f32; 4],
    flags: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct CompositeData {
    params: CompositeParams,
    hud_texture: gpu::TextureView,
    hud_sampler: gpu::Sampler,
}

pub struct Canvas {
    /// Pixels.
    size: [u32; 2],
    pixels_per_point: f32,
    ctx: egui::Context,
    painter: be::GuiPainter,
    grid: grid::Renderer,
    texture: gpu::Texture,
    view: gpu::TextureView,
    initialized: bool,
    composite: gpu::RenderPipeline,
    sampler: gpu::Sampler,
}

impl Canvas {
    /// A canvas of `size` pixels, laid out at `pixels_per_point`, that
    /// composites into targets of `target_format`.
    pub fn new(
        context: &gpu::Context,
        size: [u32; 2],
        pixels_per_point: f32,
        target_format: gpu::TextureFormat,
    ) -> Self {
        let ctx = egui::Context::default();
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            FONT_NAME.to_owned(),
            std::sync::Arc::new(egui::FontData::from_static(FONT)),
        );
        for family in [egui::FontFamily::Monospace, egui::FontFamily::Proportional] {
            fonts
                .families
                .entry(family)
                .or_default()
                .insert(0, FONT_NAME.to_owned());
        }
        ctx.set_fonts(fonts);

        let info = gpu::SurfaceInfo {
            format: FORMAT,
            alpha: gpu::AlphaMode::PreMultiplied,
        };
        let painter = be::GuiPainter::new(info, context);
        let grid = grid::Renderer::new(info, context);
        let extent = gpu::Extent {
            width: size[0],
            height: size[1],
            depth: 1,
        };
        let texture = context.create_texture(gpu::TextureDesc {
            name: "hud",
            format: FORMAT,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "hud",
                format: FORMAT,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );

        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("../shaders/hud_composite.wgsl"),
            naga_module: None,
        });
        let layout = <CompositeData as gpu::ShaderData>::layout();
        let composite = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "hud-composite",
            data_layouts: &[&layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState::default(),
            depth_stencil: None,
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: target_format,
                blend: Some(gpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "hud",
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            size,
            pixels_per_point,
            ctx,
            painter,
            grid,
            texture,
            view,
            initialized: false,
            composite,
            sampler,
        }
    }

    pub fn size(&self) -> [u32; 2] {
        self.size
    }

    /// The canvas extent in egui points.
    pub fn size_points(&self) -> egui::Vec2 {
        egui::vec2(
            self.size[0] as f32 / self.pixels_per_point,
            self.size[1] as f32 / self.pixels_per_point,
        )
    }

    /// Lay out one HUD frame with `draw` and record painting it into the
    /// canvas. `time` drives egui's clock; the HUD's own animation takes
    /// time from its state, not from egui. Grid text in `draw` goes through
    /// [`grid::Builder`], with glyph caches the caller keeps across frames.
    pub fn render(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        time: f64,
        mut draw: impl FnMut(&mut egui::Ui),
    ) {
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, self.size_points());
        let input = egui::RawInput {
            screen_rect: Some(screen),
            time: Some(time),
            viewports: [(
                egui::ViewportId::ROOT,
                egui::ViewportInfo {
                    native_pixels_per_point: Some(self.pixels_per_point),
                    inner_rect: Some(screen),
                    ..Default::default()
                },
            )]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let output = self.ctx.run_ui(input, &mut draw);
        let jobs = self.ctx.tessellate(output.shapes, output.pixels_per_point);
        let screen = be::ScreenDescriptor {
            physical_size: (self.size[0], self.size[1]),
            scale_factor: output.pixels_per_point,
        };
        self.painter
            .update_textures(encoder, &output.textures_delta, context);
        self.grid
            .update_atlas(encoder, &output.textures_delta, context);
        if !self.initialized {
            encoder.init_texture(self.texture);
            self.initialized = true;
        }
        let mut pass = encoder.render(
            "hud",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view: self.view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::TransparentBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        );
        self.grid
            .paint(&mut pass, &mut self.painter, &jobs, &screen, context);
    }

    /// Draw the canvas into a pass, placed like the grade pass places the
    /// camera image. `linear_target` is set when the target encodes sRGB
    /// itself, as XR swapchains usually do.
    pub fn composite(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        placement: Placement,
        linear_target: bool,
        opacity: f32,
    ) {
        let mut pc = pass.with(&self.composite);
        pc.bind(
            0,
            &CompositeData {
                params: CompositeParams {
                    placement: placement.transform,
                    flags: [if linear_target { 1.0 } else { 0.0 }, opacity, 0.0, 0.0],
                },
                hud_texture: self.view,
                hud_sampler: self.sampler,
            },
        );
        pc.draw(0, 3, 0, 1);
    }

    /// Call after submitting the frame that rendered and composited this
    /// canvas.
    pub fn after_submit(&mut self, sync_point: &gpu::SyncPoint) {
        self.painter.after_submit(sync_point);
        self.grid.after_submit(sync_point);
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        self.painter.destroy(context);
        self.grid.destroy(context);
        context.destroy_render_pipeline(&mut self.composite);
        context.destroy_sampler(self.sampler);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

/// One HUD frame as a [`crate::render::Overlay`]: lays out `hud` on the
/// canvas before the pass and composites it within.
pub struct HudOverlay<'a> {
    pub canvas: &'a mut Canvas,
    pub hud: &'a mut super::Hud,
    /// Seconds, the HUD's clock.
    pub now: f32,
    pub tracks: &'a [crate::perception::track::Track],
    pub telemetry: super::Telemetry,
}

impl crate::render::Overlay for HudOverlay<'_> {
    fn prepare(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        let (hud, now, tracks, telemetry) =
            (&mut *self.hud, self.now, self.tracks, &self.telemetry);
        self.canvas.render(context, encoder, now as f64, |ui| {
            hud.draw(ui, now, tracks, telemetry)
        });
    }

    fn draw(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        placement: Placement,
        linear_target: bool,
    ) {
        self.canvas.composite(pass, placement, linear_target, 1.0);
    }

    fn submitted(&mut self, sync_point: &gpu::SyncPoint) {
        self.canvas.after_submit(sync_point);
    }
}
