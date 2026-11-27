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
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Placement {
    /// Rows of a 3×3 projective map from the camera quad's local space
    /// (x right, y up, -1..1) to clip space `(x, y, w)`; each row's fourth
    /// value is unused. Projective, so an image can be placed through the
    /// camera's pose and a plane in front of it, not only scaled.
    pub rows: [[f32; 4]; 3],
}

impl Placement {
    /// The quad fills the target exactly.
    pub const FULLSCREEN: Self = Self::affine([1.0, 1.0, 0.0, 0.0]);

    /// Scale (`x, y`) and offset (`z, w`) the quad in clip space.
    pub const fn affine(transform: [f32; 4]) -> Self {
        let [sx, sy, ox, oy] = transform;
        Self {
            rows: [[sx, 0.0, ox, 0.0], [0.0, sy, oy, 0.0], [0.0, 0.0, 1.0, 0.0]],
        }
    }

    /// Letterbox an image of `image_aspect` (width / height) into a target
    /// of `target_aspect`, preserving its shape.
    pub fn fit(image_aspect: f32, target_aspect: f32) -> Self {
        let (sx, sy) = if image_aspect > target_aspect {
            (1.0, target_aspect / image_aspect)
        } else {
            (image_aspect / target_aspect, 1.0)
        };
        Self::affine([sx, sy, 0.0, 0.0])
    }

    /// Where a point of the quad's local space lands, in normalised device
    /// coordinates (divided through).
    pub fn map(&self, local: [f32; 2]) -> [f32; 2] {
        let [x, y, w] = self.rows.map(|r| r[0] * local[0] + r[1] * local[1] + r[2]);
        [x / w, y / w]
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

/// How the wireframe looks; see [`WireframePass`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WireframeStyle {
    /// Display-referred colour and overall opacity.
    pub color: [f32; 4],
    /// Depth units between contour lines.
    pub spacing: f32,
    /// Contour line width, in target pixels.
    pub line_px: f32,
    /// Offset of the contours, in fractions of `spacing`; animate it for a
    /// scanning drift.
    pub phase: f32,
    /// Instance drawn at full strength, 0 for none.
    pub highlight: u8,
    /// Alpha of the fill inside masks.
    pub fill: f32,
    /// How much fainter instances other than `highlight` are.
    pub dim: f32,
}

impl Default for WireframeStyle {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0, 0.85],
            spacing: 0.1,
            line_px: 1.5,
            phase: 0.0,
            highlight: 0,
            fill: 0.06,
            dim: 0.35,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WireframeParams {
    placement: [[f32; 4]; 3],
    color: [f32; 4],
    contour: [f32; 4],
    style: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct WireframeData {
    params: WireframeParams,
    mask_texture: gpu::TextureView,
    depth_texture: gpu::TextureView,
}

/// A texture filled from the CPU through its own staging buffer.
struct Plane {
    texture: gpu::Texture,
    view: gpu::TextureView,
    staging: gpu::Buffer,
    size: [u32; 2],
    format: gpu::TextureFormat,
    texel_bytes: u32,
    initialized: bool,
}

impl Plane {
    fn new(context: &gpu::Context, name: &str, format: gpu::TextureFormat, size: [u32; 2]) -> Self {
        let texel_bytes = format.block_info().size as u32;
        let texture = context.create_texture(gpu::TextureDesc {
            name,
            format,
            size: gpu::Extent {
                width: size[0],
                height: size[1],
                depth: 1,
            },
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
                name,
                format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        let staging = context.create_buffer(gpu::BufferDesc {
            name,
            size: (size[0] * size[1] * texel_bytes) as u64,
            memory: gpu::Memory::Upload,
        });
        Self {
            texture,
            view,
            staging,
            size,
            format,
            texel_bytes,
            initialized: false,
        }
    }

    /// Record an upload of `bytes`, a `size` grid, recreating the texture
    /// when the size changes. One upload may be in flight at a time.
    fn upload(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        name: &str,
        size: [u32; 2],
        bytes: &[u8],
    ) {
        if size != self.size {
            let fresh = Self::new(context, name, self.format, size);
            std::mem::replace(self, fresh).destroy(context);
        }
        assert_eq!(bytes.len(), (size[0] * size[1] * self.texel_bytes) as usize);
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
        let mut transfer = encoder.transfer(name);
        transfer.copy_buffer_to_texture(
            self.staging.at(0),
            size[0] * self.texel_bytes,
            self.texture.into(),
            gpu::Extent {
                width: size[0],
                height: size[1],
                depth: 1,
            },
        );
    }

    fn destroy(self, context: &gpu::Context) {
        context.destroy_buffer(self.staging);
        context.destroy_texture_view(self.view);
        context.destroy_texture(self.texture);
    }
}

/// The HUD's wireframe: lines of constant depth inside segmentation masks,
/// with each mask's outline and a faint fill. Masks and depth are
/// [`crate::perception::dense`] grids spanning the camera image, so this is
/// placed exactly like [`GradePass`]. Contours come from interpolated depth,
/// so coarse depth still draws smooth lines that wrap around the shape.
pub struct WireframePass {
    pipeline: gpu::RenderPipeline,
    mask: Plane,
    depth: Plane,
}

impl WireframePass {
    pub fn new(context: &gpu::Context, color_format: gpu::TextureFormat) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("shaders/wireframe.wgsl"),
            naga_module: None,
        });
        let layout = <WireframeData as gpu::ShaderData>::layout();
        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "wireframe",
            data_layouts: &[&layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState::default(),
            depth_stencil: None,
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: color_format,
                blend: Some(gpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });
        Self {
            pipeline,
            mask: Plane::new(
                context,
                "wireframe-mask",
                gpu::TextureFormat::R8Unorm,
                [1, 1],
            ),
            depth: Plane::new(
                context,
                "wireframe-depth",
                gpu::TextureFormat::R32Float,
                [1, 1],
            ),
        }
    }

    /// Record uploads of the latest mask and depth. The staging buffers are
    /// reused, so call once the previous submission that read them is done.
    pub fn upload(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        mask: &crate::perception::dense::Mask,
        depth: &crate::perception::dense::DepthMap,
    ) {
        self.mask.upload(
            context,
            encoder,
            "wireframe-mask",
            [mask.width, mask.height],
            &mask.ids,
        );
        self.depth.upload(
            context,
            encoder,
            "wireframe-depth",
            [depth.width, depth.height],
            bytemuck::cast_slice(&depth.values),
        );
    }

    pub fn draw(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        placement: Placement,
        style: &WireframeStyle,
        linear_target: bool,
    ) {
        let mut pc = pass.with(&self.pipeline);
        pc.bind(
            0,
            &WireframeData {
                params: WireframeParams {
                    placement: placement.rows,
                    color: style.color,
                    contour: [
                        style.spacing.max(1e-6),
                        style.line_px,
                        style.phase,
                        f32::from(style.highlight),
                    ],
                    style: [
                        style.fill,
                        style.dim,
                        if linear_target { 1.0 } else { 0.0 },
                        0.0,
                    ],
                },
                mask_texture: self.mask.view,
                depth_texture: self.depth.view,
            },
        );
        pc.draw(0, 3, 0, 1);
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.pipeline);
        self.mask.destroy(context);
        self.depth.destroy(context);
    }
}

/// One box outline: `rect` is `[min x, min y, max x, max y]` in normalised
/// image coordinates (y down), as perception reports them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoxInstance {
    pub rect: [f32; 4],
    pub color: [f32; 4],
}

impl BoxInstance {
    pub fn new(rect: &crate::perception::Rect, color: [f32; 4]) -> Self {
        Self {
            rect: [rect.min[0], rect.min[1], rect.max[0], rect.max[1]],
            color,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoxParams {
    placement: [[f32; 4]; 3],
    viewport: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct BoxData {
    params: BoxParams,
    boxes: gpu::BufferPiece,
}

/// Draws box outlines in camera-image coordinates, with the same
/// [`Placement`] as [`GradePass`] so they sit on what the camera saw. A
/// debugging view until the HUD draws targets properly.
pub struct BoxPass {
    pipeline: gpu::RenderPipeline,
    buffer: gpu::Buffer,
    count: u32,
}

impl BoxPass {
    /// Boxes beyond this are not drawn.
    pub const CAPACITY: usize = 128;

    pub fn new(context: &gpu::Context, color_format: gpu::TextureFormat) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("shaders/boxes.wgsl"),
            naga_module: None,
        });
        let layout = <BoxData as gpu::ShaderData>::layout();
        let pipeline = context.create_render_pipeline(gpu::RenderPipelineDesc {
            name: "boxes",
            data_layouts: &[&layout],
            vertex: shader.at("vs_main"),
            vertex_fetches: &[],
            primitive: gpu::PrimitiveState::default(),
            depth_stencil: None,
            fragment: Some(shader.at("fs_main")),
            color_targets: &[gpu::ColorTargetState {
                format: color_format,
                blend: Some(gpu::BlendState::ALPHA_BLENDING),
                write_mask: gpu::ColorWrites::ALL,
            }],
            multisample_state: gpu::MultisampleState::default(),
        });
        let buffer = context.create_buffer(gpu::BufferDesc {
            name: "boxes",
            size: (Self::CAPACITY * std::mem::size_of::<BoxInstance>()) as u64,
            memory: gpu::Memory::Shared,
        });
        Self {
            pipeline,
            buffer,
            count: 0,
        }
    }

    /// Replace the boxes. The buffer is shared with in-flight draws, so call
    /// only once the previous frame's submission has completed.
    pub fn set_boxes(&mut self, context: &gpu::Context, boxes: &[BoxInstance]) {
        let boxes = &boxes[..boxes.len().min(Self::CAPACITY)];
        let bytes: &[u8] = bytemuck::cast_slice(boxes);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.buffer.data(), bytes.len());
        }
        context.sync_buffer(
            self.buffer.into(),
            bytes.len().max(1) as u64,
            gpu::BufferTarget::Data,
        );
        self.count = boxes.len() as u32;
    }

    pub fn draw(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        placement: Placement,
        target_size: [u32; 2],
        thickness: f32,
    ) {
        if self.count == 0 {
            return;
        }
        let mut pc = pass.with(&self.pipeline);
        pc.bind(
            0,
            &BoxData {
                params: BoxParams {
                    placement: placement.rows,
                    viewport: [target_size[0] as f32, target_size[1] as f32, thickness, 0.0],
                },
                boxes: self.buffer.into(),
            },
        );
        pc.draw(0, 6, 0, self.count * 4);
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_render_pipeline(&mut self.pipeline);
        context.destroy_buffer(self.buffer);
    }
}

/// Grade one frame offscreen at its own resolution, draw `boxes` over it,
/// and read back RGBA8.
///
/// Headless and synchronous: for previews, tests and dataset tooling, never
/// for the display path.
pub fn render_graded(
    context: &gpu::Context,
    frame: &Frame,
    grade: &Grade,
    time: f32,
    boxes: &[BoxInstance],
) -> Vec<u8> {
    render_offscreen(context, frame, grade, time, boxes, &mut [])
}

/// Something drawn over the graded frame, such as the HUD, which records
/// work of its own before the frame's pass.
pub trait Overlay {
    /// Record anything that must precede the pass, such as rendering into
    /// an offscreen texture.
    fn prepare(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder);
    /// Draw into the frame's pass, over the grade and boxes.
    fn draw(&self, pass: &mut gpu::RenderCommandEncoder, placement: Placement, linear_target: bool);
    /// The frame was submitted at `sync_point`.
    fn submitted(&mut self, sync_point: &gpu::SyncPoint);
}

/// [`render_graded`] with [`Overlay`]s drawn over it, in order.
pub fn render_offscreen(
    context: &gpu::Context,
    frame: &Frame,
    grade: &Grade,
    time: f32,
    boxes: &[BoxInstance],
    overlays: &mut [&mut dyn Overlay],
) -> Vec<u8> {
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
    let mut box_pass = BoxPass::new(context, format);
    box_pass.set_boxes(context, boxes);
    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "offscreen",
        buffer_count: 1,
        manual_barriers: false,
    });
    encoder.start();
    encoder.init_texture(target);
    pass.upload(context, &mut encoder, frame);
    for overlay in overlays.iter_mut() {
        overlay.prepare(context, &mut encoder);
    }
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
        // Lines scale with the image so they read the same at any size.
        let thickness = (size.width.max(size.height) as f32 / 400.0).max(1.0);
        box_pass.draw(
            &mut rp,
            Placement::FULLSCREEN,
            [size.width, size.height],
            thickness,
        );
        for overlay in overlays.iter() {
            overlay.draw(&mut rp, Placement::FULLSCREEN, false);
        }
    }
    {
        let mut transfer = encoder.transfer("readback");
        transfer.copy_texture_to_buffer(target.into(), readback.into(), size.width * 4, size);
    }
    let sync = context.submit(&mut encoder);
    for overlay in overlays.iter_mut() {
        overlay.submitted(&sync);
    }
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
    box_pass.destroy(context);
    context.destroy_command_encoder(&mut encoder);
    context.destroy_buffer(readback);
    context.destroy_texture_view(view);
    context.destroy_texture(target);
    pixels
}

/// One wireframe frame as an [`Overlay`]: uploads `mask` and `depth`, then
/// draws them.
pub struct WireframeOverlay<'a> {
    pub pass: &'a mut WireframePass,
    pub mask: &'a crate::perception::dense::Mask,
    pub depth: &'a crate::perception::dense::DepthMap,
    pub style: WireframeStyle,
}

impl Overlay for WireframeOverlay<'_> {
    fn prepare(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        self.pass.upload(context, encoder, self.mask, self.depth);
    }

    fn draw(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        placement: Placement,
        linear_target: bool,
    ) {
        self.pass.draw(pass, placement, &self.style, linear_target);
    }

    fn submitted(&mut self, _sync_point: &gpu::SyncPoint) {}
}
