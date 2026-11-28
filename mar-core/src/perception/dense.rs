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
/// The object is the surface at the box's centre, where the detector puts
/// it: grown from there across pixels whose depth changes smoothly, and
/// stopped where depth jumps, at the object's edge. Depth Anything's edges
/// are sharp, so this follows the object however much of its box it fills,
/// and leaves out whatever else stands in the box. Cutting at the box's
/// median depth instead lost the far half of anything filling its box, and
/// kept anything near at the box's sides.
///
/// Boxes are painted largest first, so a small object in front of a large
/// one keeps its own id. Ids are each box's `id`.
pub fn silhouettes(boxes: &[(u8, super::Rect)], depth: &DepthMap, width: u32, height: u32) -> Mask {
    let mut mask = Mask::new(width, height);
    let mut order: Vec<&(u8, super::Rect)> = boxes.iter().collect();
    let area = |r: &super::Rect| (r.max[0] - r.min[0]) * (r.max[1] - r.min[1]);
    order.sort_by(|a, b| area(&b.1).total_cmp(&area(&a.1)));
    for &&(id, rect) in &order {
        for index in silhouette(rect, depth, width, height) {
            mask.ids[index] = id;
        }
    }
    mask
}

/// The largest step in depth between neighbouring pixels that is still the
/// same surface, as a share of the box's spread of depth: well above a
/// surface's own slope from pixel to pixel, well below the jump at an edge.
const EDGE: f32 = 0.05;

/// How far past its seeds toward its surroundings (the box's border) an
/// object may reach, as a share of the way: a surface leaning back, but
/// not the floor it stands on running off behind it, which is how growth
/// otherwise leaks from an object into its support and on to what stands
/// behind.
const REACH: f32 = 0.6;

/// One box's object, as indices into a `width` × `height` grid; see
/// [`silhouettes`].
fn silhouette(rect: super::Rect, depth: &DepthMap, width: u32, height: u32) -> Vec<usize> {
    // The box's pixels on the grid, by their centres.
    let span = |min: f32, max: f32, n: u32| {
        let first = (min * n as f32 - 0.5).ceil().clamp(0.0, n as f32) as usize;
        let end = (max * n as f32 - 0.5).ceil().clamp(0.0, n as f32) as usize;
        (first, end.max(first))
    };
    let (x0, x1) = span(rect.min[0], rect.max[0], width);
    let (y0, y1) = span(rect.min[1], rect.max[1], height);
    let (w, h) = (x1 - x0, y1 - y0);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let values: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| {
            depth.sample(
                ((x0 + x) as f32 + 0.5) / width as f32,
                ((y0 + y) as f32 + 0.5) / height as f32,
            )
        })
        .collect();
    let quantile = |sorted: &[f32], q: f32| sorted[((sorted.len() - 1) as f32 * q) as usize];
    let mut sorted = values.clone();
    sorted.sort_by(f32::total_cmp);
    let (far, near) = (quantile(&sorted, 0.05), quantile(&sorted, 0.95));
    // A box of one flat depth is all object.
    let step = (EDGE * (near - far)).max(1e-6);

    // Seeds: the nearer half of the box's middle third. Objects stand in
    // front of what is behind them, and a see-through one (a bicycle) shows
    // background at its centre.
    let middle = |n: usize| (n / 3, (2 * n).div_ceil(3).max(n / 3 + 1));
    let ((mx0, mx1), (my0, my1)) = (middle(w), middle(h));
    let mut central: Vec<(usize, f32)> = (my0..my1)
        .flat_map(|y| (mx0..mx1).map(move |x| y * w + x))
        .map(|i| (i, values[i]))
        .collect();
    central.sort_by(|a, b| a.1.total_cmp(&b.1));
    let seeds = &central[central.len() / 2..];
    let seed_level = seeds[seeds.len() / 2].1;
    // What surrounds the object: the median of the box's border.
    let mut border: Vec<f32> = (0..w * h)
        .filter(|&i| {
            let (x, y) = (i % w, i / w);
            x == 0 || y == 0 || x + 1 == w || y + 1 == h
        })
        .map(|i| values[i])
        .collect();
    border.sort_by(f32::total_cmp);
    let surroundings = quantile(&border, 0.5).min(seed_level);
    let floor = seed_level - REACH * (seed_level - surroundings);

    let mut object = vec![false; w * h];
    let mut queue: std::collections::VecDeque<usize> = seeds.iter().map(|&(i, _)| i).collect();
    for &(i, _) in seeds {
        object[i] = true;
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        let neighbours = [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ];
        for n in neighbours.into_iter().flatten() {
            if !object[n] && values[n] >= floor && (values[n] - values[i]).abs() <= step {
                object[n] = true;
                queue.push_back(n);
            }
        }
    }

    // Holes: background the box's border cannot reach is inside the object.
    let mut outside = vec![false; w * h];
    let mut queue: std::collections::VecDeque<usize> = (0..w * h)
        .filter(|&i| {
            let (x, y) = (i % w, i / w);
            (x == 0 || y == 0 || x + 1 == w || y + 1 == h) && !object[i]
        })
        .collect();
    for &i in &queue {
        outside[i] = true;
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = (i % w, i / w);
        let neighbours = [
            (x > 0).then(|| i - 1),
            (x + 1 < w).then(|| i + 1),
            (y > 0).then(|| i - w),
            (y + 1 < h).then(|| i + w),
        ];
        for n in neighbours.into_iter().flatten() {
            if !object[n] && !outside[n] {
                outside[n] = true;
                queue.push_back(n);
            }
        }
    }
    (0..w * h)
        .filter(|&i| !outside[i])
        .map(|i| (y0 + i / w) * width as usize + x0 + i % w)
        .collect()
}

/// The median of `depth` over the pixels `mask` marks, or `None` where it
/// marks none: an object's typical value, unmoved by its edges.
pub fn masked_median(mask: &Mask, depth: &DepthMap) -> Option<f32> {
    let mut values: Vec<f32> = (0..mask.ids.len())
        .filter(|&i| mask.ids[i] != 0)
        .map(|i| {
            let (x, y) = (i as u32 % mask.width, i as u32 / mask.width);
            depth.sample(
                (x as f32 + 0.5) / mask.width as f32,
                (y as f32 + 0.5) / mask.height as f32,
            )
        })
        .collect();
    if values.is_empty() {
        return None;
    }
    let middle = values.len() / 2;
    Some(*values.select_nth_unstable_by(middle, f32::total_cmp).1)
}

/// The range of `depth` over the pixels `mask` marks, or `None` where it
/// marks none: the span the wireframe's lines divide.
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

    /// A dome of an object filling most of its box, against a far
    /// background, with a nearer bar at the box's side that is not part of
    /// it: what a toy in a tray looks like.
    fn toy_in_a_tray(u: f32, v: f32) -> f32 {
        let d = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
        if d < 0.35 {
            // 5 at the middle, 3 at the rim: its far half is far from the
            // box's median.
            3.0 + 2.0 * (1.0 - (d / 0.35).powi(2))
        } else if (0.1..0.14).contains(&u) {
            6.0
        } else {
            1.0
        }
    }

    #[test]
    fn a_silhouette_is_the_whole_object_and_nothing_beside_it() {
        let mut depth = DepthMap::new(96, 96);
        depth.fill(toy_in_a_tray);
        let mask = silhouettes(&[(1, Rect::new(0.08, 0.1, 0.9, 0.9))], &depth, 96, 96);
        let (mut object, mut found, mut stray) = (0, 0, 0);
        for y in 0..96 {
            for x in 0..96 {
                let (u, v) = ((x as f32 + 0.5) / 96.0, (y as f32 + 0.5) / 96.0);
                let inside = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt() < 0.35;
                let marked = mask.ids[y * 96 + x] == 1;
                object += inside as u32;
                found += (inside && marked) as u32;
                stray += (!inside && marked) as u32;
            }
        }
        // Bilinear sampling softens the rim by a pixel or so.
        assert!(found as f32 > 0.97 * object as f32, "{found} of {object}");
        assert!(
            (stray as f32) < 0.03 * object as f32,
            "{stray} stray pixels"
        );
    }

    #[test]
    fn holes_in_an_object_are_filled() {
        // A ring of near depth around a far centre: a frame, say.
        let mut depth = DepthMap::new(64, 64);
        depth.fill(|u, v| {
            let d = ((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt();
            if (0.1..0.3).contains(&d) { 4.0 } else { 1.0 }
        });
        let mask = silhouettes(&[(1, Rect::new(0.1, 0.1, 0.9, 0.9))], &depth, 64, 64);
        assert_eq!(mask.ids[32 * 64 + 32], 1, "the hole is part of the object");
        assert_eq!(mask.ids[8 * 64 + 8], 0, "outside the ring is not");
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
