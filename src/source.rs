//! Where frames come from.
//!
//! The XR loop pulls from a [`FrameSource`] and does not care whether the
//! pixels came from the passthrough camera, a desktop image, or
//! [`TestPattern`]. A missing camera permission should degrade to the test
//! pattern rather than to a black screen.

use crate::frame::{Eye, Frame};

pub trait FrameSource {
    /// The newest frame, or `None` if nothing new arrived since the last
    /// call. Callers keep showing the previous frame in that case.
    fn next_frame(&mut self) -> Option<&Frame>;
}

/// A still image, delivered once.
pub struct StillImage {
    frame: Frame,
    delivered: bool,
}

impl StillImage {
    pub fn new(frame: Frame) -> Self {
        Self {
            frame,
            delivered: false,
        }
    }
}

impl FrameSource for StillImage {
    fn next_frame(&mut self) -> Option<&Frame> {
        if self.delivered {
            return None;
        }
        self.delivered = true;
        Some(&self.frame)
    }
}

/// A moving synthetic scene with a full luminance range: a gradient,
/// drifting discs at several brightness levels, a high-contrast
/// checkerboard, and a bright "window" to exercise the highlight end of the
/// grade.
pub struct TestPattern {
    frame: Frame,
    tick: u32,
}

impl TestPattern {
    /// Quest 3 passthrough camera resolution.
    pub const CAMERA_SIZE: [u32; 2] = [1280, 960];

    pub fn new(width: u32, height: u32) -> Self {
        Self {
            frame: Frame::new(width, height, Eye::Left),
            tick: 0,
        }
    }

    fn render(&mut self) {
        let (w, h) = (self.frame.width as usize, self.frame.height as usize);
        let t = self.tick as f32 / 60.0;
        let discs = [
            (0.30 + 0.12 * t.cos(), 0.55 + 0.10 * t.sin(), 0.14, 70.0),
            (0.62 + 0.10 * (t * 0.7).cos(), 0.40, 0.11, 150.0),
            (0.80, 0.70 + 0.08 * (t * 1.3).sin(), 0.09, 220.0),
        ];
        let aspect = w as f32 / h as f32;
        let luma = self.frame.luma_mut();
        for y in 0..h {
            let v = y as f32 / h as f32;
            for x in 0..w {
                let u = x as f32 / w as f32;
                let mut value = 20.0 + 90.0 * v;
                for &(cx, cy, r, level) in &discs {
                    let (dx, dy) = ((u - cx) * aspect, v - cy);
                    if dx * dx + dy * dy < r * r {
                        value = level;
                    }
                }
                if u < 0.15 && v < 0.2 && ((x / 16) + (y / 16)) % 2 == 0 {
                    value = 235.0;
                }
                if (0.42..0.58).contains(&u) && (0.06..0.22).contains(&v) {
                    value = 250.0;
                }
                luma[y * w + x] = value as u8;
            }
        }
        self.frame.timestamp_ns = (t as f64 * 1e9) as i64;
    }
}

impl FrameSource for TestPattern {
    fn next_frame(&mut self) -> Option<&Frame> {
        self.render();
        self.tick += 1;
        Some(&self.frame)
    }
}
