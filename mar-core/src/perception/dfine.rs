//! D-FINE, the quality detector: a real-time DETR, imported from ONNX.
//!
//! D-FINE (Peng et al., <https://arxiv.org/abs/2410.13842>, Apache-2.0)
//! reaches 42.8 COCO AP at its Nano size against YOLOX-Nano's 25.8, at
//! about six times the compute. It is not built by hand: Meganeura imports
//! the Transformers export (`onnx-community/dfine-nano-coco-ONNX`) after
//! `tools/fetch_dfine.sh` fixes its input size and folds the constant
//! subgraphs, so any D-FINE size or fine-tune loads the same way.
//!
//! # Input and output
//!
//! The frame is stretched over the input, not letterboxed, as RGB in
//! `0..=1` without normalisation. A 640×480 input matches the cameras' 4:3.
//! The model returns a fixed set of queries (300), each with sigmoid class
//! logits and a normalised `cx, cy, w, h` box. Queries are already
//! deduplicated by the transformer, so there is no NMS.

use std::path::Path;
use std::sync::Arc;

use super::preprocess::{Fit, GpuPreprocess, InputSpec};
use super::{Detection, Label, Rect};
use crate::frame::Frame;

/// The input D-FINE was trained on, at a given `[width, height]`.
pub fn input_spec(size: [u32; 2]) -> InputSpec {
    InputSpec {
        size,
        letterbox: false,
        pad: 0.0,
        bgr: false,
        value_scale: 1.0 / 255.0,
        focus: false,
    }
}

/// A compiled D-FINE ready to run, with its GPU preprocessing.
pub struct Dfine {
    spec: InputSpec,
    queries: usize,
    classes: usize,
    logits_output: usize,
    session: meganeura::Session,
    input_name: String,
    preprocess: Option<GpuPreprocess>,
    logits: Vec<f32>,
    boxes: Vec<f32>,
    pub score_threshold: f32,
}

impl Dfine {
    /// Import `model`, an export prepared by `tools/fetch_dfine.sh` for an
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
        // Two outputs, which the importer flattens: boxes `[queries, 4]`
        // and logits `[queries, classes]`. Their sizes alone tell them apart,
        // since both share the query count and there are more classes than
        // box coordinates.
        let sizes: Vec<usize> = graph
            .outputs()
            .iter()
            .map(|&id| graph.node(id).ty.shape.iter().product())
            .collect();
        let split = |logits: usize, boxes: usize| {
            let queries = boxes / 4;
            (boxes.is_multiple_of(4)
                && queries > 0
                && logits.is_multiple_of(queries)
                && logits / queries > 4)
                .then_some((queries, logits / queries))
        };
        let (logits_output, (queries, classes)) = match sizes.as_slice() {
            &[a, b] => match (split(a, b), split(b, a)) {
                (Some(shape), None) => (0, shape),
                (None, Some(shape)) => (1, shape),
                _ => {
                    return Err(format!(
                        "cannot tell logits from boxes in outputs {sizes:?}"
                    ));
                }
            },
            _ => {
                return Err(format!(
                    "expected logits and boxes, found outputs {sizes:?}"
                ));
            }
        };

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
            "D-FINE: {queries} queries, {classes} classes, {}x{} input, {} dispatches",
            input_size[0],
            input_size[1],
            session.plan().dispatches.len()
        );
        Ok(Self {
            spec,
            queries,
            classes,
            logits_output,
            session,
            input_name,
            preprocess,
            logits: vec![0.0; queries * classes],
            boxes: vec![0.0; queries * 4],
            score_threshold: 0.4,
        })
    }

    pub fn input_spec(&self) -> &InputSpec {
        &self.spec
    }

    /// Detection slots per frame: the most objects one frame can report.
    pub fn queries(&self) -> usize {
        self.queries
    }

    pub fn session(&mut self) -> &mut meganeura::Session {
        &mut self.session
    }

    fn read_outputs(&mut self) {
        let (logits, boxes) = (self.logits_output, 1 - self.logits_output);
        self.session.wait_read_output(logits, &mut self.logits);
        self.session.read_output_by_index(boxes, &mut self.boxes);
    }

    /// Run on an already preprocessed input (see
    /// [`super::preprocess::preprocess`]) and return the raw logits
    /// `[queries, classes]` and boxes `[queries, 4]`.
    pub fn run(&mut self, input: &[f32]) -> (&[f32], &[f32]) {
        self.session.set_input(&self.input_name, input);
        self.session.step();
        self.read_outputs();
        (&self.logits, &self.boxes)
    }

    /// Preprocess `frame` on the GPU straight into the session's input and
    /// run the network.
    pub fn run_frame(&mut self, frame: &Frame) -> Fit {
        let context = self.session.context();
        let input = self
            .session
            .input_buffer(&self.input_name)
            .expect("the input exists");
        // The previous step was waited on when its outputs were read, so
        // the input buffer is free to overwrite.
        let fit = self
            .preprocess
            .as_mut()
            .expect("alive until drop")
            .run(&context, frame, &self.spec, input);
        self.session.step();
        self.read_outputs();
        fit
    }
}

impl Drop for Dfine {
    fn drop(&mut self) {
        if let Some(preprocess) = self.preprocess.take() {
            preprocess.destroy(&self.session.context());
        }
    }
}

impl super::Detector for Dfine {
    fn detect(&mut self, frame: &Frame) -> Vec<Detection> {
        let fit = self.run_frame(frame);
        let mut detections = decode(
            &self.logits,
            &self.boxes,
            self.classes,
            &fit,
            self.score_threshold,
        );
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        detections
    }
}

/// Turn raw logits and boxes into detections in normalised frame
/// coordinates, keeping each query's best class if its sigmoid score
/// clears `threshold`.
pub fn decode(
    logits: &[f32],
    boxes: &[f32],
    classes: usize,
    fit: &Fit,
    threshold: f32,
) -> Vec<Detection> {
    // Compare logits, not probabilities: the sigmoid is monotonic, so one
    // exponential per query suffices.
    let logit_threshold = (threshold / (1.0 - threshold)).ln();
    let input = fit.resized.map(|v| v as f32);
    logits
        .chunks_exact(classes)
        .zip(boxes.as_chunks::<4>().0)
        .filter_map(|(scores, &[cx, cy, w, h])| {
            let (class, &logit) = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))?;
            if logit < logit_threshold {
                return None;
            }
            // Boxes are normalised to the input; the fit maps input pixels
            // back to the frame.
            let [x0, y0] = fit.to_frame([(cx - 0.5 * w) * input[0], (cy - 0.5 * h) * input[1]]);
            let [x1, y1] = fit.to_frame([(cx + 0.5 * w) * input[0], (cy + 0.5 * h) * input[1]]);
            Some(Detection {
                rect: Rect::new(
                    x0.clamp(0.0, 1.0),
                    y0.clamp(0.0, 1.0),
                    x1.clamp(0.0, 1.0),
                    y1.clamp(0.0, 1.0),
                ),
                label: Label(class as u16),
                score: 1.0 / (1.0 + (-logit).exp()),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_keeps_confident_queries_with_their_best_class() {
        let classes = 3;
        // Query 0: class 2 at logit 2 (p 0.88). Query 1: nothing confident.
        let logits = [-5.0, -1.0, 2.0, -3.0, -4.0, -2.0];
        let boxes = [0.5, 0.5, 0.2, 0.4, 0.1, 0.1, 0.1, 0.1];
        let fit = Fit::new([1280, 960], &input_spec([640, 480]));
        let dets = decode(&logits, &boxes, classes, &fit, 0.5);
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert_eq!(d.label, Label(2));
        assert!((d.score - 0.880_797).abs() < 1e-5);
        let expect = Rect::new(0.4, 0.3, 0.6, 0.7);
        for (a, b) in d
            .rect
            .min
            .iter()
            .chain(&d.rect.max)
            .zip(expect.min.iter().chain(&expect.max))
        {
            assert!((a - b).abs() < 1e-5, "{:?}", d.rect);
        }
    }

    #[test]
    fn input_is_stretched_rgb_in_unit_range() {
        let spec = input_spec([640, 480]);
        assert!(!spec.letterbox && !spec.bgr && !spec.focus);
        assert_eq!(spec.value_scale, 1.0 / 255.0);
        assert_eq!(Fit::new([1280, 960], &spec).resized, [640, 480]);
    }
}
