//! Per-pixel perception: instance masks and depth, in camera-image
//! coordinates.
//!
//! Both are plain row-major grids at whatever resolution their model
//! produces, spanning the whole camera image, so the HUD can sample them with
//! the same normalised coordinates it uses for boxes. A model that works on a
//! crop pastes its result into the full-image grid.

/// Instance ids per pixel: 0 is background; 1..=255 name instances.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub ids: Vec<u8>,
}

impl Mask {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            ids: vec![0; (width * height) as usize],
        }
    }

    /// Mark every pixel whose centre `inside` accepts, in normalised image
    /// coordinates, with `id`.
    pub fn paint(&mut self, id: u8, mut inside: impl FnMut(f32, f32) -> bool) {
        for y in 0..self.height {
            for x in 0..self.width {
                let u = (x as f32 + 0.5) / self.width as f32;
                let v = (y as f32 + 0.5) / self.height as f32;
                if inside(u, v) {
                    self.ids[(y * self.width + x) as usize] = id;
                }
            }
        }
    }
}

/// Depth per pixel, in whatever unit the source gives: metres from the
/// Depth API, relative inverse depth from a monocular model. Contours only
/// need it to be smooth and monotonic within an object.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

impl DepthMap {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            values: vec![0.0; (width * height) as usize],
        }
    }

    /// Fill from a function of normalised image coordinates.
    pub fn fill(&mut self, mut depth: impl FnMut(f32, f32) -> f32) {
        for y in 0..self.height {
            for x in 0..self.width {
                let u = (x as f32 + 0.5) / self.width as f32;
                let v = (y as f32 + 0.5) / self.height as f32;
                self.values[(y * self.width + x) as usize] = depth(u, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paint_and_fill_use_pixel_centres() {
        let mut mask = Mask::new(4, 2);
        mask.paint(3, |u, _| u > 0.5);
        assert_eq!(mask.ids, [0, 0, 3, 3, 0, 0, 3, 3]);
        let mut depth = DepthMap::new(2, 2);
        depth.fill(|u, v| u + 10.0 * v);
        // Pixel centres sit at 0.25 and 0.75 on each axis.
        assert_eq!(depth.values, [2.75, 3.25, 7.75, 8.25]);
    }
}
