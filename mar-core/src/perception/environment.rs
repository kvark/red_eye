//! Metric depth from the headset: `XR_META_environment_depth`.
//!
//! The Quest senses the depth of its surroundings and hands it out as a
//! small depth image per eye, projected for views the runtime chooses near
//! the eyes. Depth Anything gives our depth its shape, but only relative,
//! and only in the camera image; this gives distances in metres and where
//! things are in the world, at no inference cost.
//!
//! [`Resample`] copies the image to the CPU through a fixed grid each frame;
//! [`EnvironmentDepth`] answers how far along a ray the first surface is,
//! and [`object_depth`] how far an object in the camera image is.

use std::mem;

use blade_graphics as gpu;

use crate::perception::Rect;
use crate::perception::dense::DepthMap;
use crate::projection::{CameraModel, Pose, Vec3, rotate};

/// The grid each eye's depth is resampled to. The Quest's own image is
/// about this size; ranging objects needs no more.
pub const GRID: [u32; 2] = [160, 160];

/// A view the depth was projected for: its pose in the XR base space, and
/// its frustum as `[left, right, up, down]` angles in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthView {
    pub pose: Pose,
    pub fov: [f32; 4],
}

impl DepthView {
    /// Where `point` (base space) lands in this view's image, in normalised
    /// coordinates (0..1, y down), with its distance ahead of the view.
    fn project(&self, point: Vec3) -> Option<([f32; 2], f32)> {
        let p = self.pose.inverse_apply(point);
        if p[2] >= -1e-4 {
            return None;
        }
        let [l, r, u, d] = self.fov.map(f32::tan);
        let (tx, ty) = (p[0] / -p[2], p[1] / -p[2]);
        let uv = [(tx - l) / (r - l), (u - ty) / (u - d)];
        Some((uv, -p[2]))
    }
}

/// One frame of the environment's depth on the CPU.
#[derive(Clone, Debug)]
pub struct EnvironmentDepth {
    /// Width and height of each layer.
    pub size: [u32; 2],
    /// The stored depth values, 0..1, layer by layer, rows top to bottom.
    pub raw: Vec<f32>,
    pub views: [DepthView; 2],
    /// The projection's planes; `far` may be infinite.
    pub near: f32,
    pub far: f32,
}

impl EnvironmentDepth {
    /// The distance ahead of the view, in metres, that a stored value
    /// stands for; `None` where nothing was sensed.
    ///
    /// The image is stored like an OpenGL depth buffer: the value maps to
    /// normalised device depth `2 d - 1` of a perspective projection between
    /// `near` and `far`, as Meta's own samples undo it.
    pub fn linear(&self, raw: f32) -> Option<f32> {
        if !(0.0..1.0).contains(&raw) {
            return None;
        }
        let ndc = 2.0 * raw - 1.0;
        let (n, f) = (self.near, self.far);
        let z = if f.is_finite() {
            2.0 * f * n / ((f + n) - ndc * (f - n))
        } else {
            2.0 * n / (1.0 - ndc)
        };
        (z.is_finite() && z > 0.0).then_some(z)
    }

    /// The sensed distance ahead of view `layer` where `point` lands in its
    /// image, with the point's own distance ahead of it.
    pub fn sample(&self, layer: usize, point: Vec3) -> Option<(f32, f32)> {
        let (uv, ahead) = self.views[layer].project(point)?;
        if !(0.0..1.0).contains(&uv[0]) || !(0.0..1.0).contains(&uv[1]) {
            return None;
        }
        let [w, h] = self.size;
        let x = (uv[0] * w as f32) as usize;
        let y = (uv[1] * h as f32) as usize;
        let index = (layer * h as usize + y) * w as usize + x;
        Some((self.linear(*self.raw.get(index)?)?, ahead))
    }

    /// How far along `direction` from `origin` (both in base space) the
    /// first sensed surface is, in units of `direction`.
    ///
    /// The views are not where the ray starts, so the surface is found by
    /// iteration: guess a distance, look up the depth where that point
    /// lands, and move to where the ray reaches that depth.
    pub fn along(&self, layer: usize, origin: Vec3, direction: Vec3) -> Option<f32> {
        let view = &self.views[layer];
        let o = view.pose.inverse_apply(origin);
        let d = rotate(
            crate::projection::conjugate(view.pose.orientation),
            direction,
        );
        if d[2] > -1e-6 {
            return None;
        }
        let mut s = 1.5;
        for _ in 0..8 {
            let point = [0, 1, 2].map(|i| origin[i] + s * direction[i]);
            let (sensed, _) = self.sample(layer, point)?;
            let next = ((-sensed - o[2]) / d[2]).max(0.05);
            if (next - s).abs() < 0.005 * s {
                return Some(next);
            }
            s = next;
        }
        Some(s)
    }
}

/// How far an object is: the median depth along the camera's axis over
/// the middle of its box, with the point that median comes from, in base
/// space.
///
/// The middle half of the box, because its edges are mostly background;
/// the median, because what remains still mixes object and background.
/// `head` is the head's pose in base space, which `camera` is relative to.
pub fn object_depth(
    depth: &EnvironmentDepth,
    camera: &CameraModel,
    head: &Pose,
    rect: Rect,
) -> Option<(f32, Vec3)> {
    const STEPS: usize = 7;
    let origin = head.apply(camera.position);
    let mut found: Vec<(f32, Vec3)> = Vec::with_capacity(STEPS * STEPS);
    for j in 0..STEPS {
        for i in 0..STEPS {
            let t = [i, j].map(|k| 0.25 + 0.5 * (k as f32 + 0.5) / STEPS as f32);
            let uv = [0, 1].map(|a| rect.min[a] + t[a] * (rect.max[a] - rect.min[a]));
            let direction = rotate(head.orientation, camera.ray(uv));
            // The left view first: the camera is the left one.
            let s = depth
                .along(0, origin, direction)
                .or_else(|| depth.along(1, origin, direction));
            if let Some(s) = s {
                found.push((s, [0, 1, 2].map(|k| origin[k] + s * direction[k])));
            }
        }
    }
    // Too few hits say more about the sensor than the object.
    if found.len() < STEPS * STEPS / 4 {
        return None;
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    Some(found[found.len() / 2])
}

impl EnvironmentDepth {
    /// The same depth with each layer's rows in the opposite order: the
    /// other reading of the image, for [`agreement`] to judge.
    pub fn flipped(&self) -> Self {
        let [w, h] = self.size.map(|v| v as usize);
        let mut raw = Vec::with_capacity(self.raw.len());
        for layer in self.raw.chunks(w * h) {
            for row in layer.chunks(w).rev() {
                raw.extend_from_slice(row);
            }
        }
        Self {
            raw,
            ..self.clone()
        }
    }
}

/// How well the headset's depth and the camera's relative depth agree,
/// as the rank correlation of their inverse depths along camera rays over
/// the image: near 1 when they see the same scene the same way round.
///
/// Monocular depth has no scale but orders surfaces well, which is all a
/// rank correlation asks of it. `relative` is inverse depth over the
/// camera image, as Depth Anything gives it.
pub fn agreement(
    depth: &EnvironmentDepth,
    camera: &CameraModel,
    head: &Pose,
    relative: &DepthMap,
) -> Option<f32> {
    const COLUMNS: usize = 16;
    const ROWS: usize = 12;
    let origin = head.apply(camera.position);
    let mut pairs = Vec::with_capacity(COLUMNS * ROWS);
    for j in 0..ROWS {
        for i in 0..COLUMNS {
            let uv = [
                (i as f32 + 0.5) / COLUMNS as f32,
                (j as f32 + 0.5) / ROWS as f32,
            ];
            let direction = rotate(head.orientation, camera.ray(uv));
            if let Some(s) = depth.along(0, origin, direction) {
                pairs.push((1.0 / s, relative.sample(uv[0], uv[1])));
            }
        }
    }
    if pairs.len() < COLUMNS * ROWS / 4 {
        return None;
    }
    let ranks = |key: fn(&(f32, f32)) -> f32| {
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by(|&a, &b| key(&pairs[a]).total_cmp(&key(&pairs[b])));
        let mut rank = vec![0.0f32; pairs.len()];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r as f32;
        }
        rank
    };
    let a = ranks(|p| p.0);
    let b = ranks(|p| p.1);
    let mean = (pairs.len() - 1) as f32 / 2.0;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        ab += (x - mean) * (y - mean);
        aa += (x - mean) * (x - mean);
        bb += (y - mean) * (y - mean);
    }
    Some(ab / (aa * bb).sqrt().max(f32::EPSILON))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    size: [u32; 4],
}

#[derive(blade_macros::ShaderData)]
struct Data {
    params: Params,
    depth_texture: gpu::TextureView,
    output: gpu::BufferPiece,
}

/// What a recorded resample was of, kept until its result is read.
struct Pending {
    views: [DepthView; 2],
    near: f32,
    far: f32,
}

/// Resamples depth images on the GPU into a buffer the CPU reads.
pub struct Resample {
    pipeline: gpu::ComputePipeline,
    buffer: gpu::Buffer,
    pending: Option<Pending>,
}

impl Resample {
    pub fn new(context: &gpu::Context) -> Self {
        let shader = context.create_shader(gpu::ShaderDesc {
            source: include_str!("../shaders/environment_depth.wgsl"),
            naga_module: None,
        });
        let layout = <Data as gpu::ShaderData>::layout();
        let pipeline = context.create_compute_pipeline(gpu::ComputePipelineDesc {
            name: "environment-depth",
            data_layouts: &[&layout],
            compute: shader.at("main"),
            full_subgroup_size: None,
        });
        let buffer = context.create_buffer(gpu::BufferDesc {
            name: "environment-depth",
            size: (2 * GRID[0] * GRID[1]) as u64 * mem::size_of::<f32>() as u64,
            memory: gpu::Memory::Shared,
        });
        Self {
            pipeline,
            buffer,
            pending: None,
        }
    }

    /// Record resampling the two layers of `texture`, a 2D-array view of a
    /// depth texture readable by shaders, projected for `views`.
    pub fn record(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        texture: gpu::TextureView,
        views: [DepthView; 2],
        near: f32,
        far: f32,
    ) {
        {
            let mut pass = encoder.compute("environment-depth");
            let mut pc = pass.with(&self.pipeline);
            pc.bind(
                0,
                &Data {
                    params: Params {
                        size: [GRID[0], GRID[1], 2, 0],
                    },
                    depth_texture: texture,
                    output: self.buffer.into(),
                },
            );
            pc.dispatch([GRID[0].div_ceil(8), GRID[1].div_ceil(8), 2]);
        }
        self.pending = Some(Pending { views, near, far });
    }

    /// The last recorded resample, once the submission that ran it has
    /// completed; `None` if nothing was recorded since the last call.
    pub fn read(&mut self) -> Option<EnvironmentDepth> {
        let Pending { views, near, far } = self.pending.take()?;
        let len = (2 * GRID[0] * GRID[1]) as usize;
        let raw =
            unsafe { std::slice::from_raw_parts(self.buffer.data() as *const f32, len) }.to_vec();
        Some(EnvironmentDepth {
            size: GRID,
            raw,
            views,
            near,
            far,
        })
    }

    pub fn destroy(mut self, context: &gpu::Context) {
        context.destroy_buffer(self.buffer);
        context.destroy_compute_pipeline(&mut self.pipeline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::multiply;
    use std::f32::consts::FRAC_PI_4;

    const NEAR: f32 = 0.1;

    /// The stored value of a surface `z` metres ahead, infinite far plane.
    fn store(z: f32) -> f32 {
        let ndc = 1.0 - 2.0 * NEAR / z;
        0.5 * (ndc + 1.0)
    }

    /// A symmetric 90° view at `position`, looking down -z, seeing a wall
    /// at `wall` metres ahead everywhere.
    fn wall(position: Vec3, wall: f32) -> EnvironmentDepth {
        let view = DepthView {
            pose: Pose {
                orientation: [0.0, 0.0, 0.0, 1.0],
                position,
            },
            fov: [-FRAC_PI_4, FRAC_PI_4, FRAC_PI_4, -FRAC_PI_4],
        };
        EnvironmentDepth {
            size: [32, 32],
            raw: vec![store(wall); 2 * 32 * 32],
            views: [view; 2],
            near: NEAR,
            far: f32::INFINITY,
        }
    }

    #[test]
    fn stored_depth_linearises() {
        let depth = wall([0.0; 3], 2.0);
        for z in [0.3, 1.0, 2.5, 7.0] {
            let got = depth.linear(store(z)).unwrap();
            assert!((got - z).abs() < 1e-3 * z, "{z}: {got}");
        }
        assert_eq!(depth.linear(1.0), None);
        let finite = EnvironmentDepth { far: 10.0, ..depth };
        // The far plane stores 1, so just under it is just under 10 m.
        assert!((finite.linear(0.99999).unwrap() - 10.0).abs() < 0.1);
    }

    #[test]
    fn a_ray_from_elsewhere_finds_the_wall() {
        // The view 5 cm left of the ray's origin; the wall 2 m ahead of it.
        let depth = wall([-0.05, 0.0, 0.0], 2.0);
        // Straight ahead: two metres.
        let s = depth.along(0, [0.0; 3], [0.0, 0.0, -1.0]).unwrap();
        assert!((s - 2.0).abs() < 0.01, "{s}");
        // Slanted, with unit depth per step: still at depth 2.
        let s = depth.along(0, [0.0; 3], [0.3, -0.2, -1.0]).unwrap();
        assert!((s - 2.0).abs() < 0.01, "{s}");
        // Away from the view: nothing.
        assert_eq!(depth.along(0, [0.0; 3], [0.0, 0.0, 1.0]), None);
    }

    #[test]
    fn rows_run_top_to_bottom() {
        // Near below the horizon, far above it, as a floor and a sky would.
        let mut depth = wall([0.0; 3], 5.0);
        for y in 16..32 {
            for x in 0..32 {
                depth.raw[y * 32 + x] = store(1.0);
            }
        }
        let below = depth.along(0, [0.0; 3], [0.0, -0.5, -1.0]).unwrap();
        let above = depth.along(0, [0.0; 3], [0.0, 0.5, -1.0]).unwrap();
        assert!((below - 1.0).abs() < 0.01 && (above - 5.0).abs() < 0.05);
    }

    #[test]
    fn only_the_right_row_order_agrees_with_the_camera() {
        // A camera at the eye, looking ahead: a floor below the horizon
        // nearing toward the bottom, a wall above it at 4 m.
        let flip = [1.0, 0.0, 0.0, 0.0];
        let camera = CameraModel {
            focal: [400.0, 400.0],
            principal: [320.0, 240.0],
            size: [640, 480],
            orientation: flip,
            position: [0.0; 3],
        };
        let head = Pose::IDENTITY;
        let mut depth = wall([0.0; 3], 4.0);
        let [w, h] = depth.size.map(|v| v as usize);
        for layer in 0..2 {
            for y in 0..h {
                // The ray's slope below the horizon, at this row.
                let ty = (1.0 - 2.0 * (y as f32 + 0.5) / h as f32) * 1.0;
                let z = if ty < 0.0 { (1.5 / -ty).min(4.0) } else { 4.0 };
                for x in 0..w {
                    depth.raw[(layer * h + y) * w + x] = store(z);
                }
            }
        }
        // What the camera sees: the same scene, as inverse depth.
        let mut relative = DepthMap::new(64, 48);
        for y in 0..48 {
            let ty = (240.0 - (y as f32 + 0.5) * 10.0) / 400.0;
            let z = if ty < 0.0 { (1.5 / -ty).min(4.0) } else { 4.0 };
            for x in 0..64 {
                relative.values[y * 64 + x] = 1.0 / z;
            }
        }
        let right = agreement(&depth, &camera, &head, &relative).unwrap();
        let wrong = agreement(&depth.flipped(), &camera, &head, &relative).unwrap();
        assert!(right > 0.9 && wrong < 0.5, "{right} vs {wrong}");
        // Flipping twice changes nothing.
        assert_eq!(depth.flipped().flipped().raw, depth.raw);
    }

    #[test]
    fn objects_range_through_the_camera() {
        // A camera 8 cm ahead of the head, pitched down 6°, with Android
        // axes; the head turned 30° left in the world, a metre up.
        let flip = [1.0, 0.0, 0.0, 0.0];
        let pitch = 6.0f32.to_radians();
        let tilt = [(pitch / 2.0).sin(), 0.0, 0.0, (pitch / 2.0).cos()];
        let camera = CameraModel {
            focal: [867.0, 867.0],
            principal: [640.0, 480.0],
            size: [1280, 960],
            orientation: multiply(tilt, flip),
            position: [-0.03, -0.01, -0.08],
        };
        let yaw = 30.0f32.to_radians();
        let head = Pose {
            orientation: [0.0, (yaw / 2.0).sin(), 0.0, (yaw / 2.0).cos()],
            position: [0.0, 1.0, 0.0],
        };
        // The eyes' views where the head is, seeing a wall 3 m ahead.
        let mut depth = wall([0.0; 3], 3.0);
        for view in &mut depth.views {
            view.pose = Pose {
                orientation: head.orientation,
                position: head.apply([-0.032, 0.0, 0.0]),
            };
        }
        let rect = Rect::new(0.4, 0.4, 0.6, 0.6);
        let (s, point) = object_depth(&depth, &camera, &head, rect).unwrap();
        // The wall is 3 m ahead of the eyes; the camera is 8 cm nearer it
        // and looks 6° down, so its depth there is a touch less.
        assert!((s - 2.93).abs() < 0.03, "{s}");
        let ahead = -head.inverse_apply(point)[2];
        assert!((ahead - 3.0).abs() < 0.01, "{ahead}");
    }
}
