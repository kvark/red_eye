//! YOLOX, the core detector: boxes with COCO class names.
//!
//! YOLOX (Megvii, Apache-2.0, <https://arxiv.org/abs/2107.08430>) is a plain
//! CNN: CSPDarknet backbone, PAFPN neck, decoupled anchor-free head. Every
//! op it uses exists in Meganeura's graph builder, so the network is built
//! here by hand, mirroring the reference `yolox/models/*.py` module by
//! module.
//!
//! # Weights
//!
//! `tools/yolox_to_safetensors.py` converts the official ONNX exports, whose
//! BatchNorm is already folded into each convolution and whose tensor names
//! are anonymous. A convolution is identified by its position in forward
//! execution order, `conv.NNN.{weight,bias}`, so [`Builder`] must create
//! convolutions in exactly that order. Shapes are checked on load, and
//! `tests/yolox.rs` compares the whole output against onnxruntime.
//!
//! # Input
//!
//! The model's first layer, Focus, is a space-to-depth rearrangement. It
//! is a strided slice, which the graph does not need to express: the
//! preprocessing that already resamples and converts the camera frame
//! writes the rearranged layout directly. The graph input is
//! `[12, H/2, W/2]`: four phase planes (top-left, bottom-left, top-right,
//! bottom-right) of BGR in `0..=255`, the letterboxed frame padded with 114.
//!
//! # Output
//!
//! One block per stride (8, 16, 32), each `[5 + classes, h * w]`
//! channel-major: box offsets `dx, dy, log w, log h` in grid units, then
//! sigmoid objectness, then sigmoid class scores. [`decode`] turns that into
//! [`Detection`]s in normalised frame coordinates.

use std::sync::Arc;

use meganeura::{Graph, NodeId};

use super::preprocess::GpuPreprocess;
use super::{Detection, Label, Rect};
use crate::frame::Frame;

/// Strides of the three detection levels.
pub const STRIDES: [u32; 3] = [8, 16, 32];
/// Name of the graph input.
pub const INPUT: &str = "images";
/// Value YOLOX pads letterboxed images with.
const PAD_VALUE: f32 = 114.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Depth multiplier: how many bottlenecks per CSP layer.
    pub depth: f32,
    /// Width multiplier: channels per layer.
    pub width: f32,
    /// Depthwise-separable 3×3 convolutions (Nano).
    pub depthwise: bool,
    /// Network input, `[width, height]`, each a multiple of 32. The weights
    /// are resolution-independent; 416×320 matches the cameras' 4:3.
    pub input_size: [u32; 2],
    pub num_classes: u32,
}

impl Config {
    /// YOLOX-Nano: 0.91M parameters, 1.08 GFLOPs at 416², 25.8 COCO AP.
    pub fn nano() -> Self {
        Self {
            depth: 0.33,
            width: 0.25,
            depthwise: true,
            input_size: [416, 416],
            num_classes: 80,
        }
    }

    /// YOLOX-Tiny: 5.06M parameters, 6.45 GFLOPs at 416², 32.8 COCO AP.
    pub fn tiny() -> Self {
        Self {
            depth: 0.33,
            width: 0.375,
            depthwise: false,
            input_size: [416, 416],
            num_classes: 80,
        }
    }

    pub fn at_input(self, width: u32, height: u32) -> Self {
        assert!(
            width.is_multiple_of(32) && height.is_multiple_of(32),
            "YOLOX input must be a multiple of the largest stride"
        );
        Self {
            input_size: [width, height],
            ..self
        }
    }

    /// Values per anchor: 4 box, 1 objectness, then the classes.
    pub fn outputs_per_anchor(&self) -> usize {
        5 + self.num_classes as usize
    }

    /// Grid `[width, height]` of each level.
    pub fn grids(&self) -> [[u32; 2]; 3] {
        STRIDES.map(|s| [self.input_size[0] / s, self.input_size[1] / s])
    }

    pub fn anchors(&self) -> usize {
        self.grids().iter().map(|g| (g[0] * g[1]) as usize).sum()
    }

    pub fn output_len(&self) -> usize {
        self.anchors() * self.outputs_per_anchor()
    }

    /// Length of the Focus-rearranged input, `12 * H/2 * W/2`.
    pub fn input_len(&self) -> usize {
        3 * (self.input_size[0] * self.input_size[1]) as usize
    }

    /// Python's `int(c * width)`: truncation, as the reference does it.
    fn channels(&self, base: u32) -> u32 {
        (base as f32 * self.width) as u32
    }

    fn repeats(&self, base: u32) -> u32 {
        ((base as f32 * self.depth).round() as u32).max(1)
    }
}

/// One convolution, in creation order. Depthwise ones have `groups ==
/// in_channels == out_channels`; there are no other grouped convolutions.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvSpec {
    pub in_channels: u32,
    pub out_channels: u32,
    pub kernel: u32,
    pub stride: u32,
    pub depthwise: bool,
}

impl ConvSpec {
    /// The PyTorch weight shape `[out, in / groups, k, k]`.
    pub fn weight_shape(&self) -> [usize; 4] {
        let in_per_group = if self.depthwise { 1 } else { self.in_channels };
        [
            self.out_channels as usize,
            in_per_group as usize,
            self.kernel as usize,
            self.kernel as usize,
        ]
    }

    fn name(index: usize) -> String {
        format!("conv.{index:03}")
    }
}

/// A feature map flowing through the builder. Batch is always one.
#[derive(Clone, Copy)]
struct Tensor {
    node: NodeId,
    c: u32,
    h: u32,
    w: u32,
}

impl Tensor {
    fn spatial(&self) -> u32 {
        self.h * self.w
    }
}

/// Builds the network, recording each convolution as it is created so the
/// weights can be bound by position.
struct Builder<'g> {
    g: &'g mut Graph,
    depthwise: bool,
    convs: Vec<ConvSpec>,
}

impl Builder<'_> {
    /// Convolution plus bias, no activation: `nn.Conv2d` with BN folded.
    fn conv_raw(&mut self, x: Tensor, out_c: u32, k: u32, stride: u32, depthwise: bool) -> Tensor {
        let name = ConvSpec::name(self.convs.len());
        let spec = ConvSpec {
            in_channels: x.c,
            out_channels: out_c,
            kernel: k,
            stride,
            depthwise,
        };
        let weight = self.g.parameter(
            &format!("{name}.weight"),
            &[spec.weight_shape().iter().product()],
        );
        let pad = (k - 1) / 2;
        let node = if depthwise {
            assert_eq!(x.c, out_c, "depthwise convolutions keep the channel count");
            self.g
                .conv2d_dw(x.node, weight, 1, x.c, x.h, x.w, k, k, stride, pad, pad)
        } else {
            self.g
                .conv2d(x.node, weight, 1, x.c, x.h, x.w, out_c, k, k, stride, pad)
        };
        let h = (x.h + 2 * pad - k) / stride + 1;
        let w = (x.w + 2 * pad - k) / stride + 1;
        let bias = self.g.parameter(&format!("{name}.bias"), &[out_c as usize]);
        let node = self.g.add_per_channel(node, bias, out_c, h * w);
        self.convs.push(spec);
        Tensor {
            node,
            c: out_c,
            h,
            w,
        }
    }

    /// `BaseConv`: convolution, folded BN, SiLU.
    fn base_conv(&mut self, x: Tensor, out_c: u32, k: u32, stride: u32) -> Tensor {
        let y = self.conv_raw(x, out_c, k, stride, false);
        self.silu(y)
    }

    fn silu(&mut self, x: Tensor) -> Tensor {
        Tensor {
            node: self.g.silu(x.node),
            ..x
        }
    }

    /// `DWConv`: depthwise k×k then pointwise 1×1, each with SiLU.
    fn dw_conv(&mut self, x: Tensor, out_c: u32, k: u32, stride: u32) -> Tensor {
        let y = self.conv_raw(x, x.c, k, stride, true);
        let y = self.silu(y);
        self.base_conv(y, out_c, 1, 1)
    }

    /// The `Conv` alias the reference picks per model: `DWConv` for Nano.
    fn conv(&mut self, x: Tensor, out_c: u32, k: u32, stride: u32) -> Tensor {
        if self.depthwise {
            self.dw_conv(x, out_c, k, stride)
        } else {
            self.base_conv(x, out_c, k, stride)
        }
    }

    fn concat(&mut self, a: Tensor, b: Tensor) -> Tensor {
        assert_eq!((a.h, a.w), (b.h, b.w));
        Tensor {
            node: self.g.concat(a.node, b.node, 1, a.c, b.c, a.spatial()),
            c: a.c + b.c,
            ..a
        }
    }

    fn upsample(&mut self, x: Tensor) -> Tensor {
        Tensor {
            node: self.g.upsample_2x(x.node, 1, x.c, x.h, x.w),
            h: x.h * 2,
            w: x.w * 2,
            ..x
        }
    }

    /// `Bottleneck` with expansion 1.0, as `CSPLayer` instantiates it.
    fn bottleneck(&mut self, x: Tensor, out_c: u32, shortcut: bool) -> Tensor {
        let y = self.base_conv(x, out_c, 1, 1);
        let y = self.conv(y, out_c, 3, 1);
        if shortcut && x.c == out_c {
            Tensor {
                node: self.g.add(y.node, x.node),
                ..y
            }
        } else {
            y
        }
    }

    /// `CSPLayer`: two 1×1 branches, bottlenecks on the first, concat, 1×1.
    fn csp(&mut self, x: Tensor, out_c: u32, n: u32, shortcut: bool) -> Tensor {
        let hidden = out_c / 2;
        let mut x1 = self.base_conv(x, hidden, 1, 1);
        let x2 = self.base_conv(x, hidden, 1, 1);
        for _ in 0..n {
            x1 = self.bottleneck(x1, hidden, shortcut);
        }
        let joined = self.concat(x1, x2);
        self.base_conv(joined, out_c, 1, 1)
    }

    /// `SPPBottleneck`: 1×1, max pools of 5, 9 and 13 concatenated, 1×1.
    fn spp(&mut self, x: Tensor, out_c: u32) -> Tensor {
        let x = self.base_conv(x, x.c / 2, 1, 1);
        let mut joined = x;
        for k in [5, 9, 13] {
            let pooled = Tensor {
                node: self.g.max_pool_2d(x.node, 1, x.c, x.h, x.w, k, k, 1, k / 2),
                ..x
            };
            joined = self.concat(joined, pooled);
        }
        self.base_conv(joined, out_c, 1, 1)
    }
}

/// The built network: graph plus the convolution layout its weights bind to.
pub struct Network {
    pub config: Config,
    pub graph: Graph,
    pub convs: Vec<ConvSpec>,
}

impl Network {
    pub fn new(config: Config) -> Self {
        let mut graph = Graph::new();
        let [in_w, in_h] = config.input_size;
        let input = graph.input(INPUT, &[config.input_len()]);
        let mut b = Builder {
            g: &mut graph,
            depthwise: config.depthwise,
            convs: Vec::new(),
        };
        let x = Tensor {
            node: input,
            c: 12,
            h: in_h / 2,
            w: in_w / 2,
        };

        // CSPDarknet.
        let base = config.channels(64);
        let base_depth = config.repeats(3);
        let x = b.base_conv(x, base, 3, 1); // stem (Focus conv)
        let x = b.conv(x, base * 2, 3, 2);
        let x = b.csp(x, base * 2, base_depth, true); // dark2
        let x = b.conv(x, base * 4, 3, 2);
        let dark3 = b.csp(x, base * 4, base_depth * 3, true);
        let x = b.conv(dark3, base * 8, 3, 2);
        let dark4 = b.csp(x, base * 8, base_depth * 3, true);
        let x = b.conv(dark4, base * 16, 3, 2);
        let x = b.spp(x, base * 16);
        let dark5 = b.csp(x, base * 16, base_depth, false);

        // PAFPN.
        let [c0, c1, c2] = [256, 512, 1024].map(|c| config.channels(c));
        let n = config.repeats(3);
        let fpn_out0 = b.base_conv(dark5, c1, 1, 1);
        let up = b.upsample(fpn_out0);
        let f_out0 = b.concat(up, dark4);
        let f_out0 = b.csp(f_out0, c1, n, false);
        let fpn_out1 = b.base_conv(f_out0, c0, 1, 1);
        let up = b.upsample(fpn_out1);
        let f_out1 = b.concat(up, dark3);
        let pan_out2 = b.csp(f_out1, c0, n, false);
        let p_out1 = b.conv(pan_out2, c0, 3, 2);
        let p_out1 = b.concat(p_out1, fpn_out1);
        let pan_out1 = b.csp(p_out1, c1, n, false);
        let p_out0 = b.conv(pan_out1, c1, 3, 2);
        let p_out0 = b.concat(p_out0, fpn_out0);
        let pan_out0 = b.csp(p_out0, c2, n, false);

        // Decoupled head, one level at a time, in the reference's order:
        // stem, class branch and prediction, box branch, box and
        // objectness predictions.
        let head = config.channels(256);
        let mut output: Option<(NodeId, u32)> = None;
        for x in [pan_out2, pan_out1, pan_out0] {
            let x = b.base_conv(x, head, 1, 1);
            let cls = b.conv(x, head, 3, 1);
            let cls = b.conv(cls, head, 3, 1);
            let cls = b.conv_raw(cls, config.num_classes, 1, 1, false);
            let reg = b.conv(x, head, 3, 1);
            let reg = b.conv(reg, head, 3, 1);
            let boxes = b.conv_raw(reg, 4, 1, 1, false);
            let obj = b.conv_raw(reg, 1, 1, 1, false);

            let obj = Tensor {
                node: b.g.sigmoid(obj.node),
                ..obj
            };
            let cls = Tensor {
                node: b.g.sigmoid(cls.node),
                ..cls
            };
            let level = b.concat(boxes, obj);
            let level = b.concat(level, cls);
            // Levels are appended end to end: a concat of flat buffers.
            let len = level.c * level.spatial();
            output = Some(match output {
                None => (level.node, len),
                Some((prev, prev_len)) => (
                    b.g.concat(prev, level.node, 1, prev_len, len, 1),
                    prev_len + len,
                ),
            });
        }
        let convs = b.convs;
        let (output, len) = output.unwrap();
        debug_assert_eq!(len as usize, config.output_len());
        graph.set_outputs(vec![output]);
        Self {
            config,
            graph,
            convs,
        }
    }

    /// Check a converted checkpoint against this network's layout and copy
    /// it into the session.
    pub fn load_weights(
        &self,
        session: &mut meganeura::Session,
        weights: &meganeura::data::safetensors::SafeTensorsModel,
    ) -> Result<(), String> {
        let info = weights.tensor_info();
        let expected = self.convs.len() * 2;
        if info.len() != expected {
            return Err(format!(
                "checkpoint has {} tensors, the network expects {expected}: \
                 a different YOLOX variant?",
                info.len()
            ));
        }
        for (index, spec) in self.convs.iter().enumerate() {
            let name = ConvSpec::name(index);
            let weight_name = format!("{name}.weight");
            let shape = info
                .get(&weight_name)
                .map(|i| i.shape.clone())
                .ok_or_else(|| format!("{weight_name} missing"))?;
            if shape != spec.weight_shape() {
                return Err(format!(
                    "{weight_name} is {shape:?}, the network expects {:?}",
                    spec.weight_shape()
                ));
            }
            let weight = weights
                .tensor_f32_auto(&weight_name)
                .map_err(|e| format!("{weight_name}: {e}"))?;
            let bias_name = format!("{name}.bias");
            let bias = weights
                .tensor_f32_auto(&bias_name)
                .map_err(|e| format!("{bias_name}: {e}"))?;
            if bias.len() != spec.out_channels as usize {
                return Err(format!("{bias_name} has {} values", bias.len()));
            }
            session.set_parameter(&weight_name, &weight);
            session.set_parameter(&bias_name, &bias);
        }
        Ok(())
    }
}

/// A compiled YOLOX ready to run, with its GPU preprocessing.
pub struct Yolox {
    pub config: Config,
    session: meganeura::Session,
    preprocess: Option<GpuPreprocess>,
    output: Vec<f32>,
    pub score_threshold: f32,
    pub nms_threshold: f32,
}

impl Yolox {
    /// Compile the network on `gpu` (a fresh context if `None`) and load a
    /// checkpoint made by `tools/yolox_to_safetensors.py`.
    ///
    /// The session is not `Sync`; build it on the thread that will run it.
    pub fn new(
        config: Config,
        weights: &std::path::Path,
        gpu: Option<Arc<blade_graphics::Context>>,
    ) -> Result<Self, String> {
        let weights = meganeura::data::safetensors::SafeTensorsModel::load(weights.to_path_buf())
            .map_err(|e| format!("{}: {e}", weights.display()))?;
        let network = Network::new(config);
        let (mut session, _) = meganeura::build(
            &network.graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                gpu,
                ..Default::default()
            },
        );
        network.load_weights(&mut session, &weights)?;
        let config = network.config;
        let preprocess = Some(GpuPreprocess::new(&session.context()));
        Ok(Self {
            output: vec![0.0; config.output_len()],
            config,
            session,
            preprocess,
            score_threshold: 0.35,
            nms_threshold: 0.45,
        })
    }

    pub fn session(&mut self) -> &mut meganeura::Session {
        &mut self.session
    }

    /// Run on an already preprocessed input (see [`preprocess`]) and return
    /// the raw output in the layout described in the module docs.
    pub fn run(&mut self, input: &[f32]) -> &[f32] {
        self.session.set_input(INPUT, input);
        self.session.step();
        self.session.wait_read_output(0, &mut self.output);
        &self.output
    }

    /// Preprocess `frame` on the GPU straight into the session's input and
    /// run the network. Returns the raw output and how the frame was fitted.
    pub fn run_frame(&mut self, frame: &Frame) -> (&[f32], Letterbox) {
        let context = self.session.context();
        let input = self
            .session
            .input_buffer(INPUT)
            .expect("the graph has an input named INPUT");
        // The previous step was waited on by `wait_read_output`, so the
        // input buffer is free to overwrite.
        let letterbox = self.preprocess.as_mut().expect("alive until drop").run(
            &context,
            frame,
            &self.config,
            input,
        );
        self.session.step();
        self.session.wait_read_output(0, &mut self.output);
        (&self.output, letterbox)
    }
}

impl Drop for Yolox {
    fn drop(&mut self) {
        if let Some(preprocess) = self.preprocess.take() {
            preprocess.destroy(&self.session.context());
        }
    }
}

impl super::Detector for Yolox {
    fn detect(&mut self, frame: &Frame) -> Vec<Detection> {
        let (score, nms) = (self.score_threshold, self.nms_threshold);
        let letterbox = self.run_frame(frame).1;
        let mut detections = decode(&self.output, &self.config, &letterbox, score);
        super::nms(&mut detections, nms);
        detections
    }
}

/// How a frame was fitted into the network input: scaled by `scale` and
/// placed at the top-left corner, the rest padded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub frame_size: [u32; 2],
}

impl Letterbox {
    pub fn new(frame_size: [u32; 2], input_size: [u32; 2]) -> Self {
        let scale = (input_size[0] as f32 / frame_size[0] as f32)
            .min(input_size[1] as f32 / frame_size[1] as f32);
        Self { scale, frame_size }
    }

    /// The scaled frame's extent inside the input, as the reference
    /// truncates it.
    pub fn resized(&self) -> [u32; 2] {
        self.frame_size.map(|s| (s as f32 * self.scale) as u32)
    }
}

/// Focus-rearranged index of input pixel `(x, y)`, colour `c` (B, G, R).
fn focus_index(input_size: [u32; 2], x: u32, y: u32, c: u32) -> usize {
    let (half_w, half_h) = (input_size[0] / 2, input_size[1] / 2);
    let phase = (x % 2) * 2 + (y % 2);
    (((phase * 3 + c) * half_h + y / 2) * half_w + x / 2) as usize
}

/// CPU reference preprocessing: letterbox the frame into the network input
/// with bilinear resampling (as `cv2.INTER_LINEAR` does), convert BT.601
/// YUV to BGR in `0..=255`, and write the Focus layout. The GPU path in
/// `shaders/yolox_preprocess.wgsl` must match this.
pub fn preprocess(frame: &Frame, config: &Config, out: &mut Vec<f32>) -> Letterbox {
    let input = config.input_size;
    let letterbox = Letterbox::new([frame.width, frame.height], input);
    let [rw, rh] = letterbox.resized();
    out.clear();
    out.resize(config.input_len(), PAD_VALUE);

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
            let bgr = [
                l + 1.772 * u,
                l - 0.344_136 * u - 0.714_136 * v,
                l + 1.402 * v,
            ];
            for (c, value) in bgr.into_iter().enumerate() {
                out[focus_index(input, x, y, c as u32)] = value.clamp(0.0, 255.0);
            }
        }
    }
    letterbox
}

/// Turn the raw output into detections in normalised frame coordinates,
/// keeping each anchor's best class if `objectness * class score` clears
/// `threshold`. Not yet non-maximum suppressed.
pub fn decode(
    output: &[f32],
    config: &Config,
    letterbox: &Letterbox,
    threshold: f32,
) -> Vec<Detection> {
    let per_anchor = config.outputs_per_anchor();
    let mut detections = Vec::new();
    let mut offset = 0;
    let to_frame = [
        1.0 / (letterbox.scale * letterbox.frame_size[0] as f32),
        1.0 / (letterbox.scale * letterbox.frame_size[1] as f32),
    ];
    for ([gw, gh], stride) in config.grids().into_iter().zip(STRIDES) {
        let cells = (gw * gh) as usize;
        let block = &output[offset..offset + cells * per_anchor];
        offset += cells * per_anchor;
        let channel = |c: usize, cell: usize| block[c * cells + cell];
        for cell in 0..cells {
            let objectness = channel(4, cell);
            if objectness < threshold {
                continue;
            }
            let (class, class_score) = (0..config.num_classes as usize)
                .map(|k| (k, channel(5 + k, cell)))
                .fold((0, f32::MIN), |best, c| if c.1 > best.1 { c } else { best });
            let score = objectness * class_score;
            if score < threshold {
                continue;
            }
            let stride = stride as f32;
            let cx = (channel(0, cell) + (cell % gw as usize) as f32) * stride;
            let cy = (channel(1, cell) + (cell / gw as usize) as f32) * stride;
            let w = channel(2, cell).exp() * stride;
            let h = channel(3, cell).exp() * stride;
            let rect = Rect::new(
                ((cx - 0.5 * w) * to_frame[0]).clamp(0.0, 1.0),
                ((cy - 0.5 * h) * to_frame[1]).clamp(0.0, 1.0),
                ((cx + 0.5 * w) * to_frame[0]).clamp(0.0, 1.0),
                ((cy + 0.5 * h) * to_frame[1]).clamp(0.0, 1.0),
            );
            detections.push(Detection {
                rect,
                label: Label(class as u16),
                score,
            });
        }
    }
    detections
}

/// Rearrange the raw output to `[anchors, 5 + classes]`, the layout the
/// official ONNX export emits, for comparisons and tooling.
pub fn to_anchor_major(output: &[f32], config: &Config) -> Vec<f32> {
    let per_anchor = config.outputs_per_anchor();
    let mut out = Vec::with_capacity(output.len());
    let mut offset = 0;
    for [gw, gh] in config.grids() {
        let cells = (gw * gh) as usize;
        let block = &output[offset..offset + cells * per_anchor];
        for cell in 0..cells {
            out.extend((0..per_anchor).map(|c| block[c * cells + cell]));
        }
        offset += cells * per_anchor;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nano_layout_matches_the_official_export() {
        let network = Network::new(Config::nano());
        // 113 convolutions in the official yolox_nano.onnx; spot-check the
        // first (Focus stem), a depthwise one, and the last (level-2
        // objectness).
        assert_eq!(network.convs.len(), 113);
        assert_eq!(network.convs[0].weight_shape(), [16, 12, 3, 3]);
        assert_eq!(network.convs[1].weight_shape(), [16, 1, 3, 3]);
        assert_eq!(network.convs[1].stride, 2);
        assert_eq!(network.convs[82].weight_shape(), [80, 64, 1, 1]);
        assert_eq!(network.convs[112].weight_shape(), [1, 64, 1, 1]);
    }

    #[test]
    fn tiny_layout_matches_the_official_export() {
        let network = Network::new(Config::tiny());
        assert_eq!(network.convs.len(), 83);
        assert_eq!(network.convs[0].weight_shape(), [24, 12, 3, 3]);
        assert!(network.convs.iter().all(|c| !c.depthwise));
    }

    #[test]
    fn anchors_follow_the_input_size() {
        assert_eq!(Config::nano().anchors(), 3549);
        let wide = Config::nano().at_input(416, 320);
        assert_eq!(wide.grids(), [[52, 40], [26, 20], [13, 10]]);
        assert_eq!(wide.anchors(), 52 * 40 + 26 * 20 + 13 * 10);
    }

    #[test]
    fn focus_phases_follow_the_reference_order() {
        let size = [4, 4];
        // top-left, bottom-left, top-right, bottom-right; B, G, R in each.
        assert_eq!(focus_index(size, 0, 0, 0), 0);
        assert_eq!(focus_index(size, 0, 1, 0), 3 * 4);
        assert_eq!(focus_index(size, 1, 0, 0), 6 * 4);
        assert_eq!(focus_index(size, 1, 1, 2), 11 * 4);
        // (3, 2) is top-right phase, G: plane 7, row 1, column 1.
        assert_eq!(focus_index(size, 3, 2, 1), 7 * 4 + 2 + 1);
    }

    #[test]
    fn letterbox_fits_the_camera_frame() {
        let lb = Letterbox::new([1280, 960], [416, 320]);
        assert_eq!(lb.resized(), [416, 312]);
        let square = Letterbox::new([1280, 960], [416, 416]);
        assert_eq!(square.resized(), [416, 312]);
    }

    #[test]
    fn preprocess_pads_and_converts_grey() {
        let config = Config::nano().at_input(64, 64);
        let mut frame = Frame::new(64, 32, crate::frame::Eye::Left);
        frame.luma_mut().fill(200);
        let mut input = Vec::new();
        let lb = preprocess(&frame, &config, &mut input);
        assert_eq!(lb.resized(), [64, 32]);
        // Inside the image: grey stays grey in every channel.
        for c in 0..3 {
            assert!((input[focus_index([64, 64], 10, 10, c)] - 200.0).abs() < 1e-3);
        }
        // Below it: padding.
        assert_eq!(input[focus_index([64, 64], 10, 40, 0)], PAD_VALUE);
    }

    #[test]
    fn decode_places_a_box_in_frame_coordinates() {
        let config = Config::nano().at_input(64, 64);
        let mut output = vec![0.0; config.output_len()];
        let per_anchor = config.outputs_per_anchor();
        // Level 0 is 8×8 cells; light up cell (x=3, y=2) as a "person"
        // 16 pixels square, centred on the cell centre.
        let cells = 64;
        let cell = 2 * 8 + 3;
        let set = |output: &mut Vec<f32>, c: usize, v: f32| output[c * cells + cell] = v;
        set(&mut output, 0, 0.5);
        set(&mut output, 1, 0.5);
        set(&mut output, 2, 2f32.ln());
        set(&mut output, 3, 2f32.ln());
        set(&mut output, 4, 0.9);
        set(&mut output, 5, 0.8);
        assert_eq!(output.len() % per_anchor, 0);
        // A 128×128 frame was scaled by 0.5 into the 64×64 input.
        let lb = Letterbox::new([128, 128], [64, 64]);
        let dets = decode(&output, &config, &lb, 0.5);
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert_eq!(d.label, Label::PERSON);
        assert!((d.score - 0.72).abs() < 1e-6);
        // Centre (28, 20) and size 16 in input pixels → normalised.
        let expect = Rect::new(20.0 / 64.0, 12.0 / 64.0, 36.0 / 64.0, 28.0 / 64.0);
        for (a, b) in d
            .rect
            .min
            .iter()
            .chain(&d.rect.max)
            .zip(expect.min.iter().chain(&expect.max))
        {
            assert!((a - b).abs() < 1e-5, "{:?} vs {:?}", d.rect, expect);
        }
    }
}
