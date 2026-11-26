//! Blade rendering: the graded camera view.
//!
//! [`GradePass`] owns a single-channel texture holding the latest camera
//! luminance and draws it through the red grade. Only the Y plane is
//! uploaded: the grade is defined over luminance, and Camera2 delivers
//! YUV420, so the plane can be copied as-is with no colour conversion on
//! the CPU.
//!
//! On the headset this is the fallback world view. The preferred one is
//! the compositor's own passthrough with the grade baked into its colour
//! map (see [`crate::grade`]), which leaves this pass unused and our layer
//! carrying only the HUD.

use blade_graphics as gpu;

use crate::frame::Frame;
use crate::grade::{Grade, GradeUniforms};

/// `Placement` in `shaders/grade.wgsl`: clip-space scale (xy) and offset
/// (zw) of the camera quad.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Placement {
    pub transform: [f32; 4],
}

impl Placement {
    /// The quad fills the target exactly.
    pub const FULLSCREEN: Self = Self {
        transform: [1.0, 1.0, 0.0, 0.0],
    };

    /// Letterbox an image of `image_aspect` (width / height) into a target
    /// of `target_aspect`, preserving its shape.
    pub fn fit(image_aspect: f32, target_aspect: f32) -> Self {
        let (sx, sy) = if image_aspect > target_aspect {
            (1.0, target_aspect / image_aspect)
        } else {
            (image_aspect / target_aspect, 1.0)
        };
        Self {
            transform: [sx, sy, 0.0, 0.0],
        }
    }
}

#[derive(blade_macros::ShaderData)]
struct GradeData {
    params: GradeUniforms,
    placement: Placement,
    luma_texture: gpu::TextureView,
    luma_sampler: gpu::Sampler,
}

/// Draws camera luminance through the grade.
pub struct GradePass {
    pipeline: gpu::RenderPipeline,
    sampler: gpu::Sampler,
    texture: gpu::Texture,
    view: gpu::TextureView,
    staging: gpu::Buffer,
    size: [u32; 2],
    initialized: bool,
}

impl GradePass {
    /// `color_format` must match the render target; on an XR surface,
    /// `XrSurface::format()`. `size` is the camera frame size.
    pub fn new(context: &gpu::Context, color_format: gpu::TextureFormat, size: [u32; 2]) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("shaders/grade.wgsl"),
            naga_module: None,
        });
        let layout = <GradeData as gpu::ShaderData>::layout();
        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "grade",
            data_layouts: &[&layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState::default(),
            depth_stencil: None,
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState::from(color_format)],
            multisample_state: gpu::MultisampleState::default(),
        });
        let sampler = context.create_sampler(gpu::SamplerDesc {
            name: "grade",
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            ..Default::default()
        });
        let (texture, view, staging) = Self::create_storage(context, size);
        Self {
            pipeline,
            sampler,
            texture,
            view,
            staging,
            size,
            initialized: false,
        }
    }

    fn create_storage(
        context: &gpu::Context,
        size: [u32; 2],
    ) -> (gpu::Texture, gpu::TextureView, gpu::Buffer) {
        let format = gpu::TextureFormat::R8Unorm;
        let extent = gpu::Extent {
            width: size[0],
            height: size[1],
            depth: 1,
        };
        let texture = context.create_texture(gpu::TextureDesc {
            name: "camera-luma",
            format,
            size: extent,
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::COPY | gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        let view = context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "camera-luma",
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let staging = context.create_buffer(gpu::BufferDesc {
            name: "camera-luma-staging",
            size: size[0] as u64 * size[1] as u64,
            memory: gpu::Memory::Upload,
        });
        (texture, view, staging)
    }

    /// Record the upload of `frame`'s luminance into `encoder`.
    ///
    /// The staging buffer is reused, so at most one upload may be in flight:
    /// call once per submitted frame, after the previous submission that
    /// read the staging buffer has been waited on.
    pub fn upload(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        frame: &Frame,
    ) {
        let size = [frame.width, frame.height];
        if size != self.size {
            context.destroy_buffer(self.staging);
            context.destroy_texture_view(self.view);
            context.destroy_texture(self.texture);
            (self.texture, self.view, self.staging) = Self::create_storage(context, size);
            self.size = size;
            self.initialized = false;
        }
        let bytes = frame.luma();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.staging.data(), bytes.len());
        }
        context.sync_buffer(
            self.staging.into(),
            bytes.len() as u64,
            gpu::BufferTarget::Data,
        );
        if !self.initialized {
            encoder.init_texture(self.texture);
            self.initialized = true;
        }
        let mut transfer = encoder.transfer("camera-luma");
        transfer.copy_buffer_to_texture(
            self.staging.at(0),
            frame.width,
            self.texture.into(),
            gpu::Extent {
                width: frame.width,
                height: frame.height,
                depth: 1,
            },
        );
    }

    /// Draw into a render pass. `linear_target` is set when the target
    /// format encodes sRGB itself, as XR swapchains usually do.
    pub fn draw(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        grade: &Grade,
        placement: Placement,
        time: f32,
        linear_target: bool,
    ) {
        let mut pc = pass.with(&self.pipeline);
        pc.bind(
            0,
            &GradeData {
                params: grade.uniforms(time, linear_target),
                placement,
                luma_texture: self.view,
                luma_sampler: self.sampler,
            },
        );
        pc.draw(0, 3, 0, 1);
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.pipeline);
        context.destroy_sampler(self.sampler);
        context.destroy_buffer(self.staging);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

/// Grade one frame offscreen at its own resolution and read back RGBA8.
///
/// Headless and synchronous: for previews, tests and dataset tooling, never
/// for the display path.
pub fn render_graded(context: &gpu::Context, frame: &Frame, grade: &Grade, time: f32) -> Vec<u8> {
    let size = gpu::Extent {
        width: frame.width,
        height: frame.height,
        depth: 1,
    };
    let format = gpu::TextureFormat::Rgba8Unorm;
    let target = context.create_texture(gpu::TextureDesc {
        name: "offscreen",
        format,
        size,
        dimension: gpu::TextureDimension::D2,
        array_layer_count: 1,
        mip_level_count: 1,
        usage: gpu::TextureUsage::TARGET | gpu::TextureUsage::COPY,
        sample_count: 1,
        external: None,
    });
    let view = context.create_texture_view(
        target,
        gpu::TextureViewDesc {
            name: "offscreen",
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &Default::default(),
        },
    );
    let byte_count = (size.width * size.height * 4) as usize;
    let readback = context.create_buffer(gpu::BufferDesc {
        name: "offscreen-readback",
        size: byte_count as u64,
        memory: gpu::Memory::Shared,
    });

    let mut pass = GradePass::new(context, format, [frame.width, frame.height]);
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "offscreen",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    encoder.init_texture(target);
    pass.upload(context, &mut encoder, frame);
    {
        let mut rp = encoder.render(
            "grade",
            gpu::RenderTargetSet {
                colors: &[gpu::RenderTarget {
                    view,
                    init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                    finish_op: gpu::FinishOp::Store,
                }],
                depth_stencil: None,
            },
        );
        pass.draw(&mut rp, grade, Placement::FULLSCREEN, time, false);
    }
    {
        let mut transfer = encoder.transfer("readback");
        transfer.copy_texture_to_buffer(target.into(), readback.into(), size.width * 4, size);
    }
    let sync = context.submit(&mut encoder);
    let finished = context.wait_for(&sync, 30_000).unwrap_or(false);
    let mut pixels = vec![0u8; byte_count];
    if finished {
        unsafe {
            std::ptr::copy_nonoverlapping(readback.data(), pixels.as_mut_ptr(), byte_count);
        }
    } else {
        log::error!("offscreen render timed out");
    }

    pass.destroy(context);
    context.destroy_command_encoder(&mut encoder);
    context.destroy_buffer(readback);
    context.destroy_texture_view(view);
    context.destroy_texture(target);
    pixels
}
