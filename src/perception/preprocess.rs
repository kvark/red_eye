//! Camera frame to network input on the GPU.
//!
//! The frame's I420 bytes are copied once into a host-visible buffer, and a
//! compute pass on the inference session's own context resamples,
//! converts and rearranges them straight into the session's input buffer
//! (`Session::input_buffer`). Nothing per-pixel runs on the CPU, and the
//! pass is submitted on the same queue just ahead of the model, so it needs
//! no further synchronisation. The CPU reference is
//! [`super::yolox::preprocess`].

use blade_graphics as gpu;

use super::yolox::{Config, Letterbox};
use crate::frame::Frame;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    frame: [u32; 4],
    input: [u32; 4],
    scale: [f32; 4],
}

#[derive(blade_macros::ShaderData)]
struct Data {
    params: Params,
    frame: gpu::BufferPiece,
    output: gpu::BufferPiece,
}

pub struct GpuPreprocess {
    pipeline: gpu::ComputePipeline,
    encoder: gpu::CommandEncoder,
    frame_buffer: gpu::Buffer,
    frame_capacity: u64,
    /// The previous submission, which reads `frame_buffer`: it must finish
    /// before the buffer is overwritten.
    last: Option<gpu::SyncPoint>,
}

impl GpuPreprocess {
    pub fn new(context: &gpu::Context) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("../shaders/yolox_preprocess.wgsl"),
            naga_module: None,
        });
        let layout = <Data as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "yolox-preprocess",
            data_layouts: &[&layout],
            compute: shader.at("main"),
            full_subgroup_size: None,
        });
        let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "yolox-preprocess",
            buffer_count: 2,
            manual_barriers: false,
        });
        Self {
            pipeline,
            encoder,
            frame_buffer: Self::create_frame_buffer(context, 4),
            frame_capacity: 4,
            last: None,
        }
    }

    fn create_frame_buffer(context: &gpu::Context, size: u64) -> gpu::Buffer {
        context.create_buffer(gpu::BufferDesc {
            name: "yolox-frame",
            size,
            memory: gpu::Memory::Shared,
        })
    }

    /// Upload `frame` and submit its preprocessing into `output`, which must
    /// hold `config.input_len()` floats. Returns how the frame was fitted.
    ///
    /// The caller submits the model after this on the same context; queue
    /// order and the model's own initial barrier make the write visible.
    pub fn run(
        &mut self,
        context: &gpu::Context,
        frame: &Frame,
        config: &Config,
        output: gpu::BufferPiece,
    ) -> Letterbox {
        if let Some(sync) = self.last.take() {
            let _ = context.wait_for(&sync, !0);
        }
        let bytes = frame.data.len() as u64;
        let padded = bytes.next_multiple_of(4);
        if padded > self.frame_capacity {
            context.destroy_buffer(self.frame_buffer);
            self.frame_buffer = Self::create_frame_buffer(context, padded);
            self.frame_capacity = padded;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.data.as_ptr(),
                self.frame_buffer.data(),
                frame.data.len(),
            );
        }
        context.sync_buffer(self.frame_buffer.into(), padded, gpu::BufferTarget::Data);

        let [in_w, in_h] = config.input_size;
        let letterbox = Letterbox::new([frame.width, frame.height], config.input_size);
        let [rw, rh] = letterbox.resized();
        let params = Params {
            frame: [frame.width, frame.height, rw, rh],
            input: [in_w, in_h, in_w / 2, in_h / 2],
            scale: [
                frame.width as f32 / rw as f32,
                frame.height as f32 / rh as f32,
                114.0,
                0.0,
            ],
        };

        self.encoder.start();
        {
            let mut pass = self.encoder.compute("yolox-preprocess");
            let mut pc = pass.with(&self.pipeline);
            pc.bind(
                0,
                &Data {
                    params,
                    frame: self.frame_buffer.into(),
                    output,
                },
            );
            pc.dispatch([in_w.div_ceil(8), in_h.div_ceil(8), 1]);
        }
        self.last = Some(context.submit(&mut self.encoder));
        letterbox
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        if let Some(sync) = self.last.take() {
            let _ = context.wait_for(&sync, !0);
        }
        context.destroy_buffer(self.frame_buffer);
        context.destroy_command_encoder(&mut self.encoder);
        context.destroy_compute_pipeline(&mut self.pipeline);
    }
}
