//! From the camera image to an eye: where a pixel the camera saw belongs on
//! the display.
//!
//! The camera and the eyes see the world from different places, through
//! different fields of view: on a Quest 3S the left camera sits about 3 cm
//! left of, 1 cm below and 8 cm ahead of the head's centre, pitched a few
//! degrees, and covers about 73° × 58°, narrower than an eye. Stretching
//! the camera image over the eye's view, as the app first did, puts
//! everything drawn in camera coordinates (boxes, the HUD, the wireframe)
//! off what it describes once the compositor draws the world itself.
//!
//! [`CameraModel`] holds the camera's intrinsics and pose relative to the
//! head. [`placement`] turns it into the projective [`Placement`] that
//! carries the camera image onto one eye, through a plane at a chosen
//! distance in front of the camera: exact for things at that distance, off
//! by the parallax of the few centimetres between camera and eye elsewhere.

use crate::render::Placement;

/// A quaternion `x, y, z, w`.
pub type Quat = [f32; 4];
pub type Vec3 = [f32; 3];

/// Rotate `v` by the unit quaternion `q`.
pub fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let cross = |a: Vec3, b: Vec3| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = cross(u, v).map(|c| 2.0 * c);
    let ut = cross(u, t);
    [0, 1, 2].map(|i| v[i] + w * t[i] + ut[i])
}

/// The inverse of a unit quaternion.
pub fn conjugate(q: Quat) -> Quat {
    [-q[0], -q[1], -q[2], q[3]]
}

/// `a` then `b`... as a product `a * b`: rotating by it rotates by `b`
/// first, then by `a`.
pub fn multiply(a: Quat, b: Quat) -> Quat {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

/// A rigid pose: orientation and position, of something in a parent frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub orientation: Quat,
    pub position: Vec3,
}

impl Pose {
    pub const IDENTITY: Self = Self {
        orientation: [0.0, 0.0, 0.0, 1.0],
        position: [0.0; 3],
    };

    /// `self` expressed in the frame of `parent`, both given in the same
    /// frame.
    pub fn relative_to(&self, parent: &Pose) -> Pose {
        let inverse = conjugate(parent.orientation);
        let offset = [0, 1, 2].map(|i| self.position[i] - parent.position[i]);
        Pose {
            orientation: multiply(inverse, self.orientation),
            position: rotate(inverse, offset),
        }
    }

    /// A point given in this pose's frame, in its parent's.
    pub fn apply(&self, point: Vec3) -> Vec3 {
        let r = rotate(self.orientation, point);
        [0, 1, 2].map(|i| r[i] + self.position[i])
    }

    /// A point given in the parent's frame, in this pose's.
    pub fn inverse_apply(&self, point: Vec3) -> Vec3 {
        let offset = [0, 1, 2].map(|i| point[i] - self.position[i]);
        rotate(conjugate(self.orientation), offset)
    }
}

/// A pinhole camera in the stream's pixels, and its pose in the head's
/// frame (OpenXR axes: x right, y up, -z ahead). Its own axes are
/// Android's: x right, y down, z ahead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraModel {
    /// Focal lengths and principal point, in stream pixels.
    pub focal: [f32; 2],
    pub principal: [f32; 2],
    /// The stream's `[width, height]`.
    pub size: [u32; 2],
    /// Rotates camera axes into the head's.
    pub orientation: Quat,
    /// The camera's position in the head's frame, in metres.
    pub position: Vec3,
}

impl CameraModel {
    /// Where a point in the head's frame shows in the camera image, in
    /// normalised image coordinates (0..1, y down), or `None` behind the
    /// camera.
    pub fn project(&self, point: Vec3) -> Option<[f32; 2]> {
        let offset = [0, 1, 2].map(|i| point[i] - self.position[i]);
        let p = rotate(conjugate(self.orientation), offset);
        if p[2] <= 0.0 {
            return None;
        }
        let [w, h] = self.size.map(|s| s as f32);
        Some([
            (self.focal[0] * p[0] / p[2] + self.principal[0]) / w,
            (self.focal[1] * p[1] / p[2] + self.principal[1]) / h,
        ])
    }

    /// The ray through `uv` (normalised image coordinates, y down), in the
    /// head's frame, scaled to unit depth along the camera's axis: the
    /// point at camera depth `s` is `position + s * ray`.
    pub fn ray(&self, uv: [f32; 2]) -> Vec3 {
        let [w, h] = self.size.map(|s| s as f32);
        let local = [
            (uv[0] * w - self.principal[0]) / self.focal[0],
            (uv[1] * h - self.principal[1]) / self.focal[1],
            1.0,
        ];
        rotate(self.orientation, local)
    }

    /// Where the head's line of sight, at `distance` metres, shows in the
    /// camera image: the wearer's focus, on a headset without eye tracking.
    pub fn focus(&self, distance: f32) -> Option<[f32; 2]> {
        self.project([0.0, 0.0, -distance])
    }

    /// The model of a stream of `stream` pixels cut from a sensor of
    /// `sensor` pixels, from `LENS_INTRINSIC_CALIBRATION` (`fx, fy, cx, cy,
    /// skew`, in sensor pixels) and `LENS_POSE_ROTATION` /
    /// `LENS_POSE_TRANSLATION`.
    ///
    /// The stream keeps the sensor's centre and the largest region of its
    /// own aspect ratio, scaled: a 1280×960 stream of the Quest's 1728×1728
    /// sensor is its full width and middle 1296 rows, at 0.74.
    pub fn from_sensor(
        intrinsics: [f32; 5],
        sensor: [u32; 2],
        stream: [u32; 2],
        orientation: Quat,
        position: Vec3,
    ) -> Self {
        let [fx, fy, cx, cy, _skew] = intrinsics;
        let ratio = [0, 1].map(|i| stream[i] as f32 / sensor[i] as f32);
        let largest = ratio[0].max(ratio[1]);
        // The share of each sensor axis the stream covers.
        let cover = ratio.map(|r| r / largest);
        let offset = [0, 1].map(|i| sensor[i] as f32 * (1.0 - cover[i]) * 0.5);
        let scale = largest;
        Self {
            focal: [fx * scale, fy * scale],
            principal: [(cx - offset[0]) * scale, (cy - offset[1]) * scale],
            size: stream,
            orientation,
            position,
        }
    }
}

/// A 3×3 matrix by rows.
type Mat3 = [[f32; 3]; 3];

fn mat_mul(a: Mat3, b: Mat3) -> Mat3 {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

/// The matrix of rotating by `q`.
fn rotation_matrix(q: Quat) -> Mat3 {
    let columns = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(|e| rotate(q, e));
    std::array::from_fn(|r| std::array::from_fn(|c| columns[c][r]))
}

/// The projective map carrying the camera image onto an eye.
///
/// `eye` is the eye's pose in the head's frame, `fov` its frustum as
/// `[left, right, up, down]` angles in radians (OpenXR's), and `distance`
/// the depth, along the camera's axis, of the plane the image is placed on.
pub fn placement(camera: &CameraModel, eye: &Pose, fov: [f32; 4], distance: f32) -> Placement {
    let [w, h] = camera.size.map(|s| s as f32);
    let [fx, fy] = camera.focal;
    let [cx, cy] = camera.principal;
    // Local (x right, y up, -1..1) to the camera ray through that pixel,
    // scaled to the plane: pixel (u, v) = ((x + 1) w / 2, (1 - y) h / 2).
    let ray: Mat3 = [
        [w / (2.0 * fx), 0.0, (w / 2.0 - cx) / fx],
        [0.0, -h / (2.0 * fy), (h / 2.0 - cy) / fy],
        [0.0, 0.0, 1.0],
    ]
    .map(|row| row.map(|v| v * distance));
    // Into the head's frame, then the eye's: rotations, and the offset
    // between the two origins on the constant column.
    let head = mat_mul(rotation_matrix(camera.orientation), ray);
    let inverse = conjugate(eye.orientation);
    let mut in_eye = mat_mul(rotation_matrix(inverse), head);
    let offset = rotate(
        inverse,
        [0, 1, 2].map(|i| camera.position[i] - eye.position[i]),
    );
    for (row, o) in in_eye.iter_mut().zip(offset) {
        row[2] += o;
    }
    // Onto the frustum: clip x = (2 x + (r + l) z) / (r - l), and so on,
    // with w = -z, so the divided result spans -1..1 across the frustum.
    let [l, r, u, d] = fov.map(f32::tan);
    let project: Mat3 = [
        [2.0 / (r - l), 0.0, (r + l) / (r - l)],
        [0.0, 2.0 / (u - d), (u + d) / (u - d)],
        [0.0, 0.0, -1.0],
    ];
    let m = mat_mul(project, in_eye);
    Placement {
        rows: m.map(|row| [row[0], row[1], row[2], 0.0]),
    }
}

/// Where `rect` (normalised image coordinates) from a frame the camera
/// took with the head at `from` shows in a frame taken with the head at
/// `to` (both in the base space), for what stands `distance` metres along
/// the camera's axis: the bounding box of its corners carried across.
///
/// Results from different frames meet this way, such as a track's box
/// from the newest frame cut out of a depth map half a second older.
pub fn reproject_rect(
    camera: &CameraModel,
    from: &Pose,
    to: &Pose,
    rect: crate::perception::Rect,
    distance: f32,
) -> crate::perception::Rect {
    let origin = from.apply(camera.position);
    let corners = [
        [rect.min[0], rect.min[1]],
        [rect.max[0], rect.min[1]],
        [rect.min[0], rect.max[1]],
        [rect.max[0], rect.max[1]],
    ];
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for corner in corners {
        let direction = rotate(from.orientation, camera.ray(corner));
        let point = [0, 1, 2].map(|i| origin[i] + distance * direction[i]);
        // Behind the camera now: keep the corner where it was.
        let uv = camera.project(to.inverse_apply(point)).unwrap_or(corner);
        for a in 0..2 {
            min[a] = min[a].min(uv[a]);
            max[a] = max[a].max(uv[a]);
        }
    }
    crate::perception::Rect { min, max }
}

/// The head's pose from its two eyes' (both in the same frame): midway
/// between them, turned as the first. Quest eyes look parallel.
pub fn head_from_eyes(left: &Pose, right: &Pose) -> Pose {
    Pose {
        orientation: left.orientation,
        position: [0, 1, 2].map(|i| 0.5 * (left.position[i] + right.position[i])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A half-turn about x: Android camera axes (y down, z ahead) onto
    /// OpenXR's (y up, -z ahead).
    const FLIP: Quat = [1.0, 0.0, 0.0, 0.0];

    fn close(a: [f32; 2], b: [f32; 2], tolerance: f32) -> bool {
        (a[0] - b[0]).abs() < tolerance && (a[1] - b[1]).abs() < tolerance
    }

    #[test]
    fn quest_sensor_intrinsics_scale_to_the_stream() {
        let camera = CameraModel::from_sensor(
            [1170.4446, 1170.4446, 864.60406, 868.108, 0.0],
            [1728, 1728],
            [1280, 960],
            FLIP,
            [0.0; 3],
        );
        let k = 1280.0 / 1728.0;
        assert!((camera.focal[0] - 1170.4446 * k).abs() < 1e-3);
        assert!((camera.principal[0] - 864.60406 * k).abs() < 1e-3);
        // The middle 1296 rows: 216 cut from the top.
        assert!((camera.principal[1] - (868.108 - 216.0) * k).abs() < 1e-3);
    }

    #[test]
    fn a_camera_at_the_eye_with_its_frustum_maps_corners_to_corners() {
        let camera = CameraModel {
            focal: [500.0, 500.0],
            principal: [320.0, 240.0],
            size: [640, 480],
            orientation: FLIP,
            position: [0.0; 3],
        };
        // The camera's own frustum: tan = 320 / 500 across, 240 / 500 down.
        let (tx, ty) = (320.0f32 / 500.0, 240.0f32 / 500.0);
        let fov = [-tx.atan(), tx.atan(), ty.atan(), -ty.atan()];
        let placed = placement(&camera, &Pose::IDENTITY, fov, 2.0);
        for corner in [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [0.0, 0.0],
        ] {
            let got = placed.map(corner);
            assert!(close(got, corner, 1e-5), "{corner:?} -> {got:?}");
        }
    }

    #[test]
    fn points_on_the_plane_land_where_the_eye_sees_them() {
        // A camera off to the side, below, ahead, and pitched down 6°.
        let pitch = 6.0f32.to_radians();
        let tilt = [(pitch / 2.0).sin(), 0.0, 0.0, (pitch / 2.0).cos()];
        let camera = CameraModel {
            focal: [867.0, 866.0],
            principal: [640.4, 483.0],
            size: [1280, 960],
            orientation: multiply(tilt, FLIP),
            position: [-0.03, -0.012, -0.075],
        };
        let eye = Pose {
            orientation: [0.0, 0.0, 0.0, 1.0],
            position: [-0.032, 0.0, 0.0],
        };
        let fov = [-0.9f32, 0.8, 0.85, -0.95];
        let distance = 1.5;
        let placed = placement(&camera, &eye, fov, distance);
        for pixel in [[100.0, 80.0], [640.0, 480.0], [1200.0, 900.0]] {
            // The point that pixel shows on the plane, in the head's frame.
            let ray = [
                (pixel[0] - camera.principal[0]) / camera.focal[0],
                (pixel[1] - camera.principal[1]) / camera.focal[1],
                1.0,
            ];
            let in_head = rotate(camera.orientation, ray.map(|v| v * distance));
            let p = [0, 1, 2].map(|i| in_head[i] + camera.position[i] - eye.position[i]);
            // Straight into the eye's frustum.
            let [l, r, u, d] = fov.map(f32::tan);
            let (tx, ty) = (p[0] / -p[2], p[1] / -p[2]);
            let expected = [
                (2.0 * tx - (r + l)) / (r - l),
                (2.0 * ty - (u + d)) / (u - d),
            ];
            let local = [
                2.0 * pixel[0] / camera.size[0] as f32 - 1.0,
                1.0 - 2.0 * pixel[1] / camera.size[1] as f32,
            ];
            let got = placed.map(local);
            assert!(
                close(got, expected, 1e-4),
                "{pixel:?}: {got:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn the_line_of_sight_projects_into_the_image() {
        let camera = CameraModel {
            focal: [500.0, 500.0],
            principal: [320.0, 240.0],
            size: [640, 480],
            orientation: FLIP,
            position: [0.0; 3],
        };
        let focus = camera.focus(2.0).unwrap();
        assert!(close(focus, [0.5, 0.5], 1e-6), "{focus:?}");
        // Up and to the right of ahead: right of and above the centre.
        let p = camera.project([0.5, 0.5, -2.0]).unwrap();
        assert!(p[0] > 0.5 && p[1] < 0.5, "{p:?}");
        assert_eq!(camera.project([0.0, 0.0, 1.0]), None);
        // A ray and its projection agree.
        let uv = [0.3, 0.7];
        let r = camera.ray(uv);
        let back = camera.project(r.map(|v| v * 2.5)).unwrap();
        assert!(close(back, uv, 1e-6), "{back:?}");
    }

    #[test]
    fn rects_follow_the_head_between_frames() {
        let camera = CameraModel {
            focal: [500.0, 500.0],
            principal: [320.0, 240.0],
            size: [640, 480],
            orientation: FLIP,
            position: [0.0; 3],
        };
        let rect = crate::perception::Rect::new(0.4, 0.4, 0.6, 0.6);
        let still = Pose::IDENTITY;
        let same = reproject_rect(&camera, &still, &still, rect, 2.0);
        assert!(close(same.min, rect.min, 1e-5) && close(same.max, rect.max, 1e-5));
        // The head turned 5 degrees left since: what was ahead is now to
        // the right in the image.
        let yaw = 5.0f32.to_radians();
        let turned = Pose {
            orientation: [0.0, (yaw / 2.0).sin(), 0.0, (yaw / 2.0).cos()],
            position: [0.0; 3],
        };
        let moved = reproject_rect(&camera, &still, &turned, rect, 2.0);
        let shift = 500.0 * yaw.tan() / 640.0;
        assert!(
            (moved.min[0] - (rect.min[0] + shift)).abs() < 0.01,
            "{moved:?}"
        );
        assert!((moved.min[1] - rect.min[1]).abs() < 0.01, "{moved:?}");
    }

    #[test]
    fn eyes_and_head_poses_compose() {
        let left = Pose {
            orientation: [0.0, 0.0, 0.0, 1.0],
            position: [0.968, 1.5, 0.0],
        };
        let right = Pose {
            position: [1.032, 1.5, 0.0],
            ..left
        };
        let head = head_from_eyes(&left, &right);
        let eye = left.relative_to(&head);
        assert!(close(
            [eye.position[0], eye.position[1]],
            [-0.032, 0.0],
            1e-6
        ));
        // Rotating by a quaternion and back is the identity.
        let q = [0.1f32, 0.2, 0.3, 0.927];
        let n = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let q = q.map(|v| v / n);
        let v = rotate(conjugate(q), rotate(q, [0.3, -0.4, 0.5]));
        assert!(close([v[0], v[1]], [0.3, -0.4], 1e-6) && (v[2] - 0.5).abs() < 1e-6);
    }
}
