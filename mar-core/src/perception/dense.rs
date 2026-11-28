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

impl DepthMap {
    /// The value at normalised image coordinates, from the nearest pixel.
    pub fn sample(&self, u: f32, v: f32) -> f32 {
        let x = ((u * self.width as f32) as u32).min(self.width - 1);
        let y = ((v * self.height as f32) as u32).min(self.height - 1);
        self.values[(y * self.width + x) as usize]
    }
}

/// Silhouettes of boxed objects, cut out of their boxes by depth.
///
/// A detector gives boxes, which hold background as well as the object.
/// Depth separates them: within each box, the pixels nearer than the box's
/// median (larger, for inverse depth) are taken as the object, which holds
/// whenever the object fills a good part of its box and stands in front of
/// what is behind it. Boxes are painted largest first, so a small object in
/// front of a large one keeps its own id. Ids are each box's `id`.
pub fn silhouettes(boxes: &[(u8, super::Rect)], depth: &DepthMap, width: u32, height: u32) -> Mask {
    let mut mask = Mask::new(width, height);
    let mut order: Vec<&(u8, super::Rect)> = boxes.iter().collect();
    let area = |r: &super::Rect| (r.max[0] - r.min[0]) * (r.max[1] - r.min[1]);
    order.sort_by(|a, b| area(&b.1).total_cmp(&area(&a.1)));
    let inside = |r: &super::Rect, u: f32, v: f32| {
        u >= r.min[0] && u < r.max[0] && v >= r.min[1] && v < r.max[1]
    };
    for &&(id, rect) in &order {
        let mut values = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let (u, v) = (
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                );
                if inside(&rect, u, v) {
                    values.push(depth.sample(u, v));
                }
            }
        }
        if values.is_empty() {
            continue;
        }
        values.sort_by(f32::total_cmp);
        let median = values[values.len() / 2];
        mask.paint(id, |u, v| {
            inside(&rect, u, v) && depth.sample(u, v) >= median
        });
    }
    mask
}

/// The range of `depth` over the pixels `mask` marks, or `None` where it
/// marks none: the span the wireframe's contours divide.
pub fn masked_range(mask: &Mask, depth: &DepthMap) -> Option<(f32, f32)> {
    let mut range: Option<(f32, f32)> = None;
    for y in 0..mask.height {
        for x in 0..mask.width {
            if mask.ids[(y * mask.width + x) as usize] == 0 {
                continue;
            }
            let u = (x as f32 + 0.5) / mask.width as f32;
            let v = (y as f32 + 0.5) / mask.height as f32;
            let d = depth.sample(u, v);
            range = Some(range.map_or((d, d), |(lo, hi)| (lo.min(d), hi.max(d))));
        }
    }
    range
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::Rect;

    #[test]
    fn silhouettes_keep_the_near_part_of_each_box() {
        // Background at depth 1; a nearer object (depth 5) in the left half
        // of the image, which the box over the whole width half covers.
        let mut depth = DepthMap::new(8, 8);
        depth.fill(|u, v| if u < 0.5 && v > 0.25 { 5.0 } else { 1.0 });
        let mask = silhouettes(&[(7, Rect::new(0.0, 0.25, 1.0, 1.0))], &depth, 8, 8);
        for y in 0..8 {
            for x in 0..8 {
                let expected = if x < 4 && y >= 2 { 7 } else { 0 };
                assert_eq!(mask.ids[y * 8 + x], expected, "pixel {x},{y}");
            }
        }
        assert_eq!(masked_range(&mask, &depth), Some((5.0, 5.0)));
    }

    #[test]
    fn smaller_boxes_paint_over_larger_ones() {
        let mut depth = DepthMap::new(4, 4);
        depth.fill(|_, _| 2.0);
        let mask = silhouettes(
            &[
                (1, Rect::new(0.5, 0.5, 1.0, 1.0)),
                (2, Rect::new(0.0, 0.0, 1.0, 1.0)),
            ],
            &depth,
            4,
            4,
        );
        assert_eq!(mask.ids[0], 2);
        assert_eq!(mask.ids[15], 1);
        assert_eq!(masked_range(&Mask::new(2, 2), &depth), None);
    }

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
