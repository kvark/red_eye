//! Horizontal slices through an object: the look of the films' traced
//! objects, an outline with the edges of horizontal cuts across it.
//!
//! Cut an object with evenly spaced horizontal planes and look at it: each
//! cut shows as the curve where its plane meets the surface facing you,
//! bending with the object's shape. Those curves are the lines of constant
//! height in the world, so drawing them is contouring each pixel's height
//! ([`heights`]) inside the object's mask, which the wireframe pass does
//! for any field.
//!
//! A pixel's height comes from its camera ray and its depth. Depth
//! Anything has the detail but no scale; the headset's environment depth
//! has metres but soft edges. [`calibrate`] fits the first to the second
//! over the object, as the affine map of inverse depth that monocular
//! models get right up to.

use super::Rect;
use super::dense::DepthMap;
use super::environment::EnvironmentDepth;
use crate::projection::{CameraModel, Pose, rotate};

/// The fit of relative inverse depth `r` to metric camera depth `s`:
/// `1 / s = scale * r + offset`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseFit {
    pub scale: f32,
    pub offset: f32,
}

impl InverseFit {
    /// Camera depth, in metres, of relative inverse depth `r`; `None` where
    /// the fit puts it behind the camera or absurdly far.
    pub fn depth(&self, r: f32) -> Option<f32> {
        let inverse = self.scale * r + self.offset;
        (inverse > 1.0 / 100.0).then(|| 1.0 / inverse)
    }
}

/// Least-squares [`InverseFit`] of `(r, s)` pairs, refitted once without
/// the fifth that fits worst: the two depths disagree at edges, where one
/// sees the object and the other the background. `None` with too few pairs
/// or a fit that does not make nearer things nearer.
pub fn fit_inverse(pairs: &[(f32, f32)]) -> Option<InverseFit> {
    fn solve(pairs: &[(f32, f32)]) -> Option<InverseFit> {
        let n = pairs.len() as f32;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for &(r, s) in pairs {
            let y = 1.0 / s;
            sx += r;
            sy += y;
            sxx += r * r;
            sxy += r * y;
        }
        let det = n * sxx - sx * sx;
        if det.abs() < 1e-9 {
            // No spread in r: a flat object, at the mean depth.
            return Some(InverseFit {
                scale: 0.0,
                offset: sy / n,
            });
        }
        let scale = (n * sxy - sx * sy) / det;
        Some(InverseFit {
            scale,
            offset: (sy - scale * sx) / n,
        })
    }
    if pairs.len() < 8 {
        return None;
    }
    let first = solve(pairs)?;
    let residual = |&(r, s): &(f32, f32)| (first.scale * r + first.offset - 1.0 / s).abs();
    let mut ranked: Vec<(f32, f32)> = pairs.to_vec();
    ranked.sort_by(|a, b| residual(a).total_cmp(&residual(b)));
    ranked.truncate(pairs.len() - pairs.len() / 5);
    let fit = solve(&ranked)?;
    (fit.scale >= 0.0).then_some(fit)
}

/// Fit `relative` to the headset's depth over a `columns` × `rows` grid
/// across the whole camera image.
///
/// The whole image, not the object: a monocular model's depth is right up
/// to one affine map of inverse depth per image, and the room's spread of
/// depths pins that map down where one object's narrow range does not.
/// Fitted over a single chair, the scale swung tenfold between frames.
pub fn calibrate(
    environment: &EnvironmentDepth,
    camera: &CameraModel,
    head: &Pose,
    relative: &DepthMap,
    columns: u32,
    rows: u32,
) -> Option<InverseFit> {
    let origin = head.apply(camera.position);
    let pairs: Vec<(f32, f32)> = (0..rows)
        .flat_map(|y| (0..columns).map(move |x| (x, y)))
        .filter_map(|(x, y)| {
            let uv = [
                (x as f32 + 0.5) / columns as f32,
                (y as f32 + 0.5) / rows as f32,
            ];
            let direction = rotate(head.orientation, camera.ray(uv));
            let s = environment
                .along(0, origin, direction)
                .or_else(|| environment.along(1, origin, direction))?;
            Some((relative.sample(uv[0], uv[1]), s))
        })
        .collect();
    fit_inverse(&pairs)
}

impl InverseFit {
    /// The same variation of depth, moved so that relative depth `r` is at
    /// camera depth `s`: the shape from the fit over the whole image, the
    /// distance from the headset's range for the object itself, which is
    /// surer than a fit dominated by the rest of the room.
    pub fn anchored(&self, r: f32, s: f32) -> InverseFit {
        InverseFit {
            scale: self.scale,
            offset: 1.0 / s.max(1e-3) - self.scale * r,
        }
    }

    /// `self` moved `t` of the way to `other`: successive fits smoothed,
    /// so the slices do not jump between depth frames.
    pub fn blend(&self, other: &InverseFit, t: f32) -> InverseFit {
        InverseFit {
            scale: self.scale + t * (other.scale - self.scale),
            offset: self.offset + t * (other.offset - self.offset),
        }
    }
}

/// The height in the world (the base space's `y`, which points up) of what
/// each pixel of a `width` × `height` grid over the camera image shows,
/// within `region` (normalised image coordinates; elsewhere the field is
/// left at 0). `depth` gives each pixel's camera depth in metres from its
/// relative inverse depth in `relative`.
pub fn heights(
    camera: &CameraModel,
    head: &Pose,
    relative: &DepthMap,
    depth: impl Fn(f32) -> Option<f32>,
    region: Rect,
    width: u32,
    height: u32,
) -> DepthMap {
    let mut field = DepthMap::new(width, height);
    let origin = head.apply(camera.position);
    let x0 = ((region.min[0] * width as f32).floor().max(0.0)) as u32;
    let x1 = ((region.max[0] * width as f32).ceil() as u32).min(width);
    let y0 = ((region.min[1] * height as f32).floor().max(0.0)) as u32;
    let y1 = ((region.max[1] * height as f32).ceil() as u32).min(height);
    // Each pixel's own height, or the nearest known one along its row, so
    // the field has no holes for the line widths to trip over.
    let mut last = origin[1];
    for y in y0..y1 {
        for x in x0..x1 {
            let uv = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];
            if let Some(s) = depth(relative.sample(uv[0], uv[1])) {
                let direction = rotate(head.orientation, camera.ray(uv));
                last = origin[1] + s * direction[1];
            }
            field.values[(y * width + x) as usize] = last;
        }
    }
    field
}

/// Spacing of slices across an object of `extent` metres from top to
/// bottom: about a dozen, but neither finer than 2 cm, which the depth
/// cannot resolve, nor coarser than 15 cm, which reads as a few stray
/// lines rather than a shape.
pub fn spacing(extent: f32) -> f32 {
    (extent / 12.0).clamp(0.02, 0.15)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A half-turn about x: Android camera axes (y down, z ahead) onto
    /// OpenXR's (y up, -z ahead).
    const FLIP: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

    fn camera() -> CameraModel {
        CameraModel {
            focal: [400.0, 400.0],
            principal: [320.0, 240.0],
            size: [640, 480],
            orientation: FLIP,
            position: [0.0; 3],
        }
    }

    #[test]
    fn the_fit_recovers_scale_and_shrugs_off_edge_outliers() {
        // r = 2 / s exactly, i.e. 1/s = 0.5 r, plus a few wild pairs.
        let mut pairs: Vec<(f32, f32)> = (1..=20)
            .map(|i| {
                let s = 0.5 + 0.1 * i as f32;
                (2.0 / s, s)
            })
            .collect();
        pairs.extend([(4.0, 6.0), (0.5, 0.4), (3.0, 9.0)]);
        let fit = fit_inverse(&pairs).unwrap();
        assert!((fit.scale - 0.5).abs() < 0.02, "{fit:?}");
        assert!(fit.offset.abs() < 0.02, "{fit:?}");
        assert!((fit.depth(2.0 / 1.5).unwrap() - 1.5).abs() < 0.05);
        assert!(fit_inverse(&pairs[..5]).is_none(), "too few to trust");
    }

    #[test]
    fn a_flat_wall_slices_into_level_lines() {
        // Every pixel at camera depth 2 m, the head level at 1.6 m: each
        // row of the image is one height, falling towards the bottom.
        let head = Pose {
            orientation: [0.0, 0.0, 0.0, 1.0],
            position: [0.0, 1.6, 0.0],
        };
        let relative = DepthMap::new(8, 6);
        let region = Rect::new(0.0, 0.0, 1.0, 1.0);
        let field = heights(&camera(), &head, &relative, |_| Some(2.0), region, 64, 48);
        let at = |x: u32, y: u32| field.values[(y * 64 + x) as usize];
        // Level along each row: horizontal slices of a wall are straight.
        assert!((at(0, 10) - at(63, 10)).abs() < 1e-5);
        // The middle row is at eye height; the bottom row is lower.
        assert!((at(32, 24) - 1.6).abs() < 0.03, "{}", at(32, 24));
        assert!(at(32, 47) < at(32, 24) - 0.5);
    }

    #[test]
    fn a_tilted_head_still_slices_level() {
        // The head rolled 20 degrees: the slices must stay level in the
        // world, so they cross the image's rows.
        let roll = 20.0f32.to_radians();
        let head = Pose {
            orientation: [0.0, 0.0, (roll / 2.0).sin(), (roll / 2.0).cos()],
            position: [0.0, 1.6, 0.0],
        };
        let relative = DepthMap::new(8, 6);
        let region = Rect::new(0.0, 0.0, 1.0, 1.0);
        let field = heights(&camera(), &head, &relative, |_| Some(2.0), region, 64, 48);
        let at = |x: u32, y: u32| field.values[(y * 64 + x) as usize];
        assert!(
            (at(0, 24) - at(63, 24)).abs() > 0.2,
            "rows are no longer level"
        );
    }

    #[test]
    fn anchoring_keeps_the_shape_and_moves_the_distance() {
        let fit = InverseFit {
            scale: 0.5,
            offset: 0.1,
        };
        let anchored = fit.anchored(2.0, 0.8);
        assert!((anchored.depth(2.0).unwrap() - 0.8).abs() < 1e-5);
        // The same change of inverse depth per unit of r.
        let step = |f: &InverseFit| 1.0 / f.depth(3.0).unwrap() - 1.0 / f.depth(2.0).unwrap();
        assert!((step(&anchored) - step(&fit)).abs() < 1e-5);
    }

    #[test]
    fn slices_space_with_the_object() {
        assert_eq!(spacing(1.2), 0.1);
        assert_eq!(spacing(0.05), 0.02);
        assert_eq!(spacing(10.0), 0.15);
    }
}
