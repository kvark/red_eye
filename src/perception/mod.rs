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

pub mod labels;
pub mod track;

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

/// Everything perception knows at one instant.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// Capture time of the frame this was computed from, nanoseconds.
    pub timestamp_ns: i64,
    pub tracks: Vec<track::Track>,
}
