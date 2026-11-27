//! What the HUD knows about the world.
//!
//! Perception runs asynchronously from rendering, on whatever cadence the
//! models sustain, and publishes [`Observation`]s. The HUD only ever reads
//! the most recent one and interpolates; nothing on the display path waits
//! for a network.
//!
//! The core is a single object detector: every frame it yields boxes with
//! a [`labels::Label`] (a COCO class name) and a score. Tracking turns those
//! into persistent targets. Depth comes from the platform, not a model.
//! Anything further (pose, OCR, speech) is an optional extension, scheduled
//! around the detector. See `docs/technology.md`, "Model plan".
//!
//! All coordinates are normalised image coordinates of the source camera
//! frame: `x` right and `y` down, both in `0..=1`.

pub mod dense;
pub mod depth;
pub mod dfine;
pub mod environment;
pub mod labels;
pub mod preprocess;
pub mod track;
pub mod worker;
pub mod yolox;

pub use labels::{Category, Label};

/// An axis-aligned box in normalised image coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Rect {
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self {
            min: [x0.min(x1), y0.min(y1)],
            max: [x0.max(x1), y0.max(y1)],
        }
    }

    pub fn width(&self) -> f32 {
        self.max[0] - self.min[0]
    }

    pub fn height(&self) -> f32 {
        self.max[1] - self.min[1]
    }

    pub fn area(&self) -> f32 {
        self.width().max(0.0) * self.height().max(0.0)
    }

    pub fn center(&self) -> [f32; 2] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
        ]
    }

    /// Intersection over union, `0..=1`.
    pub fn iou(&self, other: &Rect) -> f32 {
        let ix = (self.max[0].min(other.max[0]) - self.min[0].max(other.min[0])).max(0.0);
        let iy = (self.max[1].min(other.max[1]) - self.min[1].max(other.min[1])).max(0.0);
        let inter = ix * iy;
        let union = self.area() + other.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }

    /// Linear interpolation towards `other`, for smoothing and for drawing
    /// between observations.
    pub fn lerp(&self, other: &Rect, t: f32) -> Rect {
        let mix = |a: f32, b: f32| a + (b - a) * t;
        Rect {
            min: [
                mix(self.min[0], other.min[0]),
                mix(self.min[1], other.min[1]),
            ],
            max: [
                mix(self.max[0], other.max[0]),
                mix(self.max[1], other.max[1]),
            ],
        }
    }
}

/// One detector output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub rect: Rect,
    pub label: Label,
    /// Confidence, `0..=1`.
    pub score: f32,
}

/// Turns a camera frame into detections. Implemented by Meganeura-backed
/// models; the trait keeps the tracker and HUD testable without a GPU.
pub trait Detector {
    fn detect(&mut self, frame: &crate::frame::Frame) -> Vec<Detection>;
}

/// Class-aware non-maximum suppression: keep the best-scoring box of each
/// overlapping same-label cluster, as YOLOX's reference postprocess does.
/// Leaves `detections` sorted by descending score.
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f32) {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());
    for det in detections.drain(..) {
        let suppressed = kept
            .iter()
            .any(|k| k.label == det.label && k.rect.iou(&det.rect) > iou_threshold);
        if !suppressed {
            kept.push(det);
        }
    }
    *detections = kept;
}

/// Everything perception knows at one instant.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// Capture time of the frame this was computed from, nanoseconds.
    pub timestamp_ns: i64,
    /// Every live track, confirmed or not; the HUD decides what to show.
    pub tracks: Vec<track::Track>,
    /// Wall time from submission to result, for the data-scroll telemetry.
    pub latency_ms: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x: f32, label: u16, score: f32) -> Detection {
        Detection {
            rect: Rect::new(x, 0.0, x + 0.2, 0.5),
            label: Label(label),
            score,
        }
    }

    #[test]
    fn nms_keeps_the_best_of_each_cluster_per_label() {
        let mut dets = vec![
            det(0.10, 0, 0.6),
            det(0.11, 0, 0.9),
            det(0.12, 2, 0.5), // same place, different label: kept
            det(0.60, 0, 0.4), // elsewhere: kept
        ];
        nms(&mut dets, 0.45);
        let kept: Vec<_> = dets.iter().map(|d| (d.label.0, d.score)).collect();
        assert_eq!(kept, [(0, 0.9), (2, 0.5), (0, 0.4)]);
    }
}
