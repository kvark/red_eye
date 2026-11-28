//! Camera frame to network input, on the GPU, for any of our detectors.
//!
//! Each network states what its input looks like in an [`InputSpec`]:
//! size, how the frame is fitted (letterboxed or stretched), channel order,
//! value range, and layout. [`GpuPreprocess`] copies the frame's I420
//! bytes once into a host-visible buffer, and a compute pass on the
//! inference session's own context resamples, converts and lays them out
//! straight into the session's input buffer (`Session::input_buffer`).
//! Nothing per-pixel runs on the CPU, and the pass is submitted on the same
//! queue just ahead of the model, so it needs no further synchronisation.
//!
//! [`preprocess`] is the CPU reference the shader must match.

use blade_graphics as gpu;

use crate::frame::Frame;

/// What a network wants its input to look like.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputSpec {
    /// `[width, height]`.
    pub size: [u32; 2],
    /// Keep the frame's aspect ratio, place it top-left, and pad the rest
    /// (YOLOX). Otherwise stretch it over the whole input (the DETR family).
    pub letterbox: bool,
    /// Padding value, in output units.
    pub pad: f32,
    /// BGR channel order (YOLOX, trained through OpenCV) rather than RGB.
    pub bgr: bool,
    /// Factor applied to `0..=255` values: 1 for YOLOX, 1/255 for D-FINE.
    pub value_scale: f32,
    /// YOLOX's Focus space-to-depth layout, `[12, H/2, W/2]`, rather than
    /// plain `[3, H, W]`.
    pub focus: bool,
}

impl InputSpec {
    /// Floats in the input tensor.
    pub fn len(&self) -> usize {
        3 * (self.size[0] * self.size[1]) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Index of input pixel `(x, y)`, colour `c` in the spec's order.
    pub fn index(&self, x: u32, y: u32, c: u32) -> usize {
        let [w, h] = self.size;
        if self.focus {
            // Phases in the reference's order: top-left, bottom-left,
            // top-right, bottom-right.
            let phase = (x % 2) * 2 + (y % 2);
            (((phase * 3 + c) * (h / 2) + y / 2) * (w / 2) + x / 2) as usize
        } else {
            ((c * h + y) * w + x) as usize
        }
    }
}

/// How a frame was fitted into the network input: scaled by `scale` per
/// axis and placed at the top-left corner, covering `resized` pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub scale: [f32; 2],
    pub resized: [u32; 2],
    pub frame_size: [u32; 2],
}

impl Fit {
    pub fn new(frame_size: [u32; 2], spec: &InputSpec) -> Self {
        let [iw, ih] = spec.size;
        let [fw, fh] = frame_size;
        if spec.letterbox {
            let s = (iw as f32 / fw as f32).min(ih as f32 / fh as f32);
            Self {
                scale: [s, s],
                // Truncated, as the YOLOX reference does it.
                resized: [(fw as f32 * s) as u32, (fh as f32 * s) as u32],
                frame_size,
            }
        } else {
            Self {
                scale: [iw as f32 / fw as f32, ih as f32 / fh as f32],
                resized: spec.size,
                frame_size,
            }
        }
    }

    /// A point in input pixels, in normalised frame coordinates.
    pub fn to_frame(&self, point: [f32; 2]) -> [f32; 2] {
        [
            point[0] / (self.scale[0] * self.frame_size[0] as f32),
            point[1] / (self.scale[1] * self.frame_size[1] as f32),
        ]
    }
}

/// CPU reference: fit the frame into the input with bilinear resampling
/// (as `cv2.INTER_LINEAR` does), convert BT.601 YUV to the spec's colour
/// order and range, and write its layout.
pub fn preprocess(frame: &Frame, spec: &InputSpec, out: &mut Vec<f32>) -> Fit {
    let fit = Fit::new([frame.width, frame.height], spec);
    let [rw, rh] = fit.resized;
    out.clear();
    out.resize(spec.len(), spec.pad);

    let (fw, fh) = (frame.width as usize, frame.height as usize);
    let luma = frame.luma();
    let (u_plane, v_plane) = frame.chroma();
    let (cw, ch) = (fw / 2, fh / 2);
    // Bilinear sample of a plane at continuous pixel coordinates.
    let sample = |plane: &[u8], w: usize, h: usize, x: f32, y: f32| -> f32 {
        let x = x.clamp(0.0, (w - 1) as f32);
        let y = y.clamp(0.0, (h - 1) as f32);
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: usize, yy: usize| plane[yy * w + xx] as f32;
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
        top + (bottom - top) * fy
    };
    let (sx, sy) = (fw as f32 / rw as f32, fh as f32 / rh as f32);
    for y in 0..rh {
        let src_y = (y as f32 + 0.5) * sy - 0.5;
        for x in 0..rw {
            let src_x = (x as f32 + 0.5) * sx - 0.5;
            let l = sample(luma, fw, fh, src_x, src_y);
            // Chroma sits at half resolution, sited between luma pairs.
            let (cx, cy) = ((src_x - 0.5) * 0.5, (src_y - 0.5) * 0.5);
            let u = sample(u_plane, cw, ch, cx, cy) - 128.0;
            let v = sample(v_plane, cw, ch, cx, cy) - 128.0;
            let rgb = [
                l + 1.402 * v,
                l - 0.344_136 * u - 0.714_136 * v,
                l + 1.772 * u,
            ];
            for c in 0..3 {
                let value = rgb[if spec.bgr { 2 - c } else { c }];
                out[spec.index(x, y, c as u32)] = value.clamp(0.0, 255.0) * spec.value_scale;
            }
        }
    }
    fit
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    frame: [u32; 4],
    input: [u32; 4],
    scale: [f32; 4],
    flags: [u32; 4],
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
            source: include_str!("../shaders/preprocess.wgsl"),
            naga_module: None,
        });
        let layout = <Data as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "preprocess",
            data_layouts: &[&layout],
            compute: shader.at("main"),
            full_subgroup_size: None,
        });
        let encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "preprocess",
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
            name: "preprocess-frame",
            size,
            memory: gpu::Memory::Shared,
        })
    }

    /// Upload `frame` and submit its preprocessing into `output`, which must
    /// hold `spec.len()` floats. Returns how the frame was fitted.
    ///
    /// The caller submits the model after this on the same context; queue
    /// order and the model's own initial barrier make the write visible.
    pub fn run(
        &mut self,
        context: &gpu::Context,
        frame: &Frame,
        spec: &InputSpec,
        output: gpu::BufferPiece,
    ) -> Fit {
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

        let [in_w, in_h] = spec.size;
        let fit = Fit::new([frame.width, frame.height], spec);
        let [rw, rh] = fit.resized;
        let params = Params {
            frame: [frame.width, frame.height, rw, rh],
            input: [in_w, in_h, in_w / 2, in_h / 2],
            scale: [
                frame.width as f32 / rw as f32,
                frame.height as f32 / rh as f32,
                spec.pad,
                spec.value_scale,
            ],
            flags: [spec.bgr as u32, spec.focus as u32, 0, 0],
        };

        self.encoder.start();
        {
            let mut pass = self.encoder.compute("preprocess");
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
        fit
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Eye;

    fn yolox_like(size: [u32; 2]) -> InputSpec {
        InputSpec {
            size,
            letterbox: true,
            pad: 114.0,
            bgr: true,
            value_scale: 1.0,
            focus: true,
        }
    }

    #[test]
    fn focus_phases_follow_the_reference_order() {
        let spec = yolox_like([4, 4]);
        // top-left, bottom-left, top-right, bottom-right; B, G, R in each.
        assert_eq!(spec.index(0, 0, 0), 0);
        assert_eq!(spec.index(0, 1, 0), 3 * 4);
        assert_eq!(spec.index(1, 0, 0), 6 * 4);
        assert_eq!(spec.index(1, 1, 2), 11 * 4);
        // (3, 2) is top-right phase, G: plane 7, row 1, column 1.
        assert_eq!(spec.index(3, 2, 1), 7 * 4 + 2 + 1);
    }

    #[test]
    fn chw_index_is_planar() {
        let spec = InputSpec {
            focus: false,
            ..yolox_like([4, 2])
        };
        assert_eq!(spec.index(3, 1, 2), 2 * 8 + 4 + 3);
    }

    #[test]
    fn letterbox_fits_the_camera_frame() {
        let fit = Fit::new([1280, 960], &yolox_like([416, 320]));
        assert_eq!(fit.resized, [416, 312]);
        let square = Fit::new([1280, 960], &yolox_like([416, 416]));
        assert_eq!(square.resized, [416, 312]);
    }

    #[test]
    fn stretch_covers_the_input_and_maps_back() {
        let spec = InputSpec {
            letterbox: false,
            ..yolox_like([640, 640])
        };
        let fit = Fit::new([1280, 960], &spec);
        assert_eq!(fit.resized, [640, 640]);
        assert_eq!(fit.to_frame([320.0, 160.0]), [0.5, 0.25]);
    }

    #[test]
    fn grey_pads_and_converts_in_either_order() {
        let mut frame = Frame::new(64, 32, Eye::Left);
        frame.luma_mut().fill(200);
        for bgr in [true, false] {
            let spec = InputSpec {
                bgr,
                ..yolox_like([64, 64])
            };
            let mut input = Vec::new();
            let fit = preprocess(&frame, &spec, &mut input);
            assert_eq!(fit.resized, [64, 32]);
            for c in 0..3 {
                assert!((input[spec.index(10, 10, c)] - 200.0).abs() < 1e-3);
            }
            assert_eq!(input[spec.index(10, 40, 0)], 114.0);
        }
    }

    #[test]
    fn red_lands_in_the_right_channel() {
        let rgb = [255u8, 0, 0].repeat(16 * 16);
        let frame = Frame::from_rgb8(16, 16, &rgb, Eye::Left);
        let spec = InputSpec {
            size: [16, 16],
            letterbox: false,
            pad: 0.0,
            bgr: false,
            value_scale: 1.0 / 255.0,
            focus: false,
        };
        let mut input = Vec::new();
        preprocess(&frame, &spec, &mut input);
        let (r, b) = (input[spec.index(8, 8, 0)], input[spec.index(8, 8, 2)]);
        assert!(r > 0.9 && b < 0.1, "r {r} b {b}");
    }
}
