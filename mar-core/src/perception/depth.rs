//! Depth Anything V2 Small: relative depth from the camera image, for the
//! wireframe's contours.
//!
//! Depth Anything V2 (Yang et al., <https://arxiv.org/abs/2406.09414>; the
//! Small model is Apache-2.0) is a DINOv2 ViT-S with a DPT head. Meganeura
//! imports the Transformers export (`onnx-community/depth-anything-v2-small`)
//! after `tools/fetch_depth.sh` fixes its input size, folds its constant
//! subgraphs, and folds the ImageNet normalisation into its patch embedding.
//!
//! # Input and output
//!
//! The frame is stretched over the input, as RGB in `0..=1`, the same as
//! D-FINE takes; both sides must be multiples of the 14-pixel patch. The
//! output is relative inverse depth (larger is nearer, in no particular
//! unit) at the input's resolution, covering the whole camera image, so it
//! lines up pixel for pixel with the boxes. Its contours are of disparity:
//! unevenly spaced in metres, which is fine for the look.

use std::path::Path;
use std::sync::Arc;

use super::dense::DepthMap;
use super::preprocess::{GpuPreprocess, InputSpec};
use crate::frame::Frame;

/// The input Depth Anything takes, at a given `[width, height]`.
pub fn input_spec(size: [u32; 2]) -> InputSpec {
    super::dfine::input_spec(size)
}

/// A compiled Depth Anything ready to run, with its GPU preprocessing.
pub struct DepthAnything {
    spec: InputSpec,
    session: meganeura::Session,
    input_name: String,
    preprocess: Option<GpuPreprocess>,
    depth: Vec<f32>,
}

impl DepthAnything {
    /// Import `model`, an export prepared by `tools/fetch_depth.sh` for an
    /// input of `input_size` (`[width, height]`), and compile it on `gpu`
    /// (a fresh context if `None`).
    ///
    /// The session is not `Sync`; build it on the thread that will run it.
    pub fn new(
        model: &Path,
        input_size: [u32; 2],
        gpu: Option<Arc<blade_graphics::Context>>,
    ) -> Result<Self, String> {
        let onnx = meganeura::load::onnx::load_onnx(model)
            .map_err(|e| format!("{}: {e}", model.display()))?;
        let spec = input_spec(input_size);
        let graph = &onnx.graph;
        let input_len = graph
            .nodes()
            .iter()
            .find_map(|n| match n.op {
                meganeura::graph::Op::Input { .. } => Some(n.ty.shape.iter().product::<usize>()),
                _ => None,
            })
            .ok_or("the model has no input")?;
        if input_len != spec.len() {
            return Err(format!(
                "{} takes {input_len} values, not a {}x{} image: prepared for another size?",
                model.display(),
                input_size[0],
                input_size[1]
            ));
        }
        let pixels = (input_size[0] * input_size[1]) as usize;
        let outputs: Vec<usize> = graph
            .outputs()
            .iter()
            .map(|&id| graph.node(id).ty.shape.iter().product())
            .collect();
        if outputs != [pixels] {
            return Err(format!(
                "expected one depth value per input pixel, found outputs {outputs:?}"
            ));
        }
        let (mut session, _) = meganeura::build(
            graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                gpu,
                ..Default::default()
            },
        );
        for (name, _) in session.plan().param_buffers.clone() {
            let data = onnx
                .weights
                .get(&name)
                .ok_or_else(|| format!("no weight for parameter {name}"))?;
            session.set_parameter(&name, data);
        }
        let input_name = session.plan().input_buffers[0].0.clone();
        let preprocess = Some(GpuPreprocess::new(&session.context()));
        log::info!(
            "Depth Anything: {}x{} input, {} dispatches",
            input_size[0],
            input_size[1],
            session.plan().dispatches.len()
        );
        Ok(Self {
            spec,
            session,
            input_name,
            preprocess,
            depth: vec![0.0; pixels],
        })
    }

    pub fn input_spec(&self) -> &InputSpec {
        &self.spec
    }

    pub fn session(&mut self) -> &mut meganeura::Session {
        &mut self.session
    }

    /// Run on an already preprocessed input and return the raw depth,
    /// row-major at the input's resolution.
    pub fn run(&mut self, input: &[f32]) -> &[f32] {
        self.session.set_input(&self.input_name, input);
        self.session.step();
        self.session.wait_read_output(0, &mut self.depth);
        &self.depth
    }

    /// Preprocess `frame` on the GPU straight into the session's input, run
    /// the network, and return its depth over the whole frame.
    pub fn run_frame(&mut self, frame: &Frame) -> DepthMap {
        let context = self.session.context();
        let input = self
            .session
            .input_buffer(&self.input_name)
            .expect("the input exists");
        // The previous step was waited on when its output was read, so the
        // input buffer is free to overwrite.
        self.preprocess
            .as_mut()
            .expect("alive until drop")
            .run(&context, frame, &self.spec, input);
        self.session.step();
        self.session.wait_read_output(0, &mut self.depth);
        let [width, height] = self.spec.size;
        DepthMap {
            width,
            height,
            values: self.depth.clone(),
        }
    }
}

impl Drop for DepthAnything {
    fn drop(&mut self) {
        if let Some(preprocess) = self.preprocess.take() {
            preprocess.destroy(&self.session.context());
        }
    }
}
