//! The RedEye XR application.
//!
//! The left passthrough camera, graded red, drawn to both eyes at display
//! rate, with detections outlined over it and the HUD on top. Detection runs on a worker
//! thread sharing the GPU context and never holds up a frame. The OpenXR
//! plumbing is taken from DinoVision, where it was proven on a Quest 3.
//!
//! This is the *self-rendered* world view. The target architecture lets the
//! compositor draw passthrough with the grade as its colour map and keeps
//! our layer for the HUD only; that needs Blade to submit an
//! `XR_FB_passthrough` layer beneath ours. See `docs/technology.md`.
//!
//! ```text
//! cargo apk run --manifest-path android-xr/Cargo.toml --release --no-logcat
//! adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA
//! adb logcat -v time | grep -E "red_eye|RustStdoutStderr"
//! ```
//!
//! Detection needs a model on the device; without one the app runs
//! without it. D-FINE is preferred, YOLOX-Nano is the light fallback:
//!
//! ```text
//! adb shell mkdir -p /data/local/tmp/red_eye
//! adb push ref/dfine_nano_640x480/model.onnx /data/local/tmp/red_eye/dfine_nano.onnx
//! adb push ref/yolox_nano.safetensors /data/local/tmp/red_eye/
//! ```

#![cfg(target_os = "android")]

mod camera;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use blade_graphics as gpu;
use log::{info, warn};
use openxr as xr;
use red_eye::frame::Eye;
use red_eye::grade::Grade;
use red_eye::hud::canvas::Canvas;
use red_eye::hud::{Hud, Telemetry};
use red_eye::perception::Detector;
use red_eye::perception::Label;
use red_eye::perception::dense::{self, DepthMap, Mask};
use red_eye::perception::depth::DepthAnything;
use red_eye::perception::dfine::Dfine;
use red_eye::perception::environment::{self, DepthView, EnvironmentDepth, Resample};
use red_eye::perception::track::TrackerConfig;
use red_eye::perception::track::{self as track, Track};
use red_eye::perception::worker::Background;
use red_eye::perception::worker::Worker;
use red_eye::perception::yolox::{Config, Yolox};
use red_eye::projection::{self, CameraModel, Pose};
use red_eye::render::{BoxInstance, BoxPass, GradePass, Placement, WireframePass, WireframeStyle};
use red_eye::source::{FrameSource, TestPattern};

const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const CAPTURE_SIZE: (i32, i32) = (1280, 960);

/// The distance, in metres, at which overlays drawn in camera coordinates
/// line up exactly with the world: the plane the camera image is placed on.
/// Things nearer or farther are off by the parallax between camera and eye,
/// a degree or two for the right eye at arm's length. With the headset's
/// depth the plane moves to the outlined object; this is where it starts,
/// and where it stays without one.
const PLACEMENT_DISTANCE: f32 = 1.5;

/// The placement plane's limits: the depth sensor's useful range.
const PLACEMENT_RANGE: std::ops::RangeInclusive<f32> = 0.3..=6.0;

/// Share of a new measurement in each smoothed distance: the depth is
/// noisy at object edges, and a plane that jumps shakes every overlay.
const RANGE_SMOOTHING: f32 = 0.3;

/// With this file present, the first environment depth frame is logged,
/// for checking its encoding on the host (`tools/env_depth.py`).
const DUMP_DEPTH_PATH: &str = "/data/local/tmp/red_eye/dump_depth";

/// Detector models, pushed with `adb` rather than bundled while the model
/// choice is still moving. D-FINE (`tools/fetch_dfine.sh`, prepared at
/// [`DFINE_SIZE`]) is preferred when present: it is far more accurate, at
/// about six times YOLOX-Nano's compute. YOLOX-Nano
/// (`tools/fetch_yolox.sh`) is the light fallback.
const DFINE_PATH: &str = "/data/local/tmp/red_eye/dfine_nano.onnx";
const DFINE_SIZE: [u32; 2] = [640, 480];
const YOLOX_PATH: &str = "/data/local/tmp/red_eye/yolox_nano.safetensors";

/// Depth Anything V2 Small, prepared by `tools/fetch_depth.sh 224 168`. The
/// wireframe needs coarse depth only, so the smallest input that keeps the
/// cameras' 4:3 in 14-pixel patches.
const DEPTH_PATH: &str = "/data/local/tmp/red_eye/depth_small.onnx";
const DEPTH_SIZE: [u32; 2] = [224, 168];

/// Least time between depth frames: it shares the GPU with detection and
/// the display, and the wireframe tolerates stale depth.
const DEPTH_INTERVAL: Duration = Duration::from_millis(500);

/// Resolution of the silhouette mask the wireframe outlines.
const MASK_SIZE: [u32; 2] = [320, 240];

/// Contour lines across the depth range of what is outlined.
const CONTOURS: f32 = 8.0;

/// Upper bound on detection rate: the 24 Hz target. Submitting faster would
/// only take GPU time from the display.
const DETECT_INTERVAL: Duration = Duration::from_millis(41);

/// Meganeura splits the network into this many queue submissions so the
/// renderer's submissions can interleave. Two was best in DinoVision's
/// co-tenancy sweep; re-measure for YOLOX.
const SUBMISSION_CHUNKS: usize = 2;

/// Frames a track must be seen in before it is outlined.
const CONFIRM_HITS: u32 = 2;

/// Outline thickness in eye-buffer pixels.
const BOX_THICKNESS: f32 = 3.0;

/// The class to look for, by COCO name, until voice commands set it:
///
/// ```text
/// adb shell "echo motorcycle > /data/local/tmp/red_eye/target"
/// adb shell rm /data/local/tmp/red_eye/target       # stand down
/// ```
///
/// With a target, the HUD brackets the match and the debug boxes go away.
const TARGET_PATH: &str = "/data/local/tmp/red_eye/target";
const TARGET_POLL: Duration = Duration::from_secs(1);

/// The HUD canvas covers the camera image at its own resolution, laid out
/// at one pixel per point.
const HUD_SIZE: [u32; 2] = [CAPTURE_SIZE.0 as u32, CAPTURE_SIZE.1 as u32];

fn read_target() -> Option<Label> {
    let name = std::fs::read_to_string(TARGET_PATH).ok()?;
    let label = Label::from_name(&name);
    if label.is_none() && !name.trim().is_empty() {
        warn!("{TARGET_PATH}: {:?} is not a COCO class", name.trim());
    }
    label
}

fn spawn_detector(context: &Arc<gpu::Context>) -> Option<Worker> {
    let dfine = std::path::PathBuf::from(DFINE_PATH);
    let yolox = std::path::PathBuf::from(YOLOX_PATH);
    if !dfine.exists() && !yolox.exists() {
        warn!("no detector at {DFINE_PATH} or {YOLOX_PATH}; running without detection");
        return None;
    }
    let context = Arc::clone(context);
    Some(Worker::spawn(
        move || {
            let model: Box<dyn Detector> = if dfine.exists() {
                let mut model = Dfine::new(&dfine, DFINE_SIZE, Some(context))?;
                model.session().set_submission_chunks(SUBMISSION_CHUNKS);
                info!(
                    "detector ready: D-FINE-N at {}x{}",
                    DFINE_SIZE[0], DFINE_SIZE[1]
                );
                Box::new(model)
            } else {
                let config = Config::nano().at_input(416, 320);
                let mut model = Yolox::new(config, &yolox, Some(context))?;
                model.session().set_submission_chunks(SUBMISSION_CHUNKS);
                info!("detector ready: YOLOX-Nano at 416x320");
                Box::new(model)
            };
            Ok(model)
        },
        TrackerConfig::default(),
    ))
}

/// The depth model on a thread of its own, if it was pushed to the device.
fn spawn_depth(context: &Arc<gpu::Context>) -> Option<Background<DepthMap>> {
    let path = std::path::PathBuf::from(DEPTH_PATH);
    if !path.exists() {
        warn!("no depth model at {DEPTH_PATH}; no wireframe");
        return None;
    }
    let context = Arc::clone(context);
    Some(Background::spawn("depth", move || {
        let mut model = DepthAnything::new(&path, DEPTH_SIZE, Some(context))?;
        model.session().set_submission_chunks(SUBMISSION_CHUNKS);
        info!(
            "depth ready: Depth Anything V2 Small at {}x{}",
            DEPTH_SIZE[0], DEPTH_SIZE[1]
        );
        Ok(move |frame: &red_eye::frame::Frame| model.run_frame(frame))
    }))
}

/// Place the camera image in an eye whose frustum is `[left, right, up,
/// down]` radians: span the frustum symmetrically about the eye's axis.
///
/// Headset frusta are asymmetric and mirrored between the eyes, so filling
/// the eye buffer edge to edge would put the image centre at a different
/// angle in each eye, and the two would not fuse. See DinoVision's
/// `render::EyeTransform` for the measurements behind this.
fn eye_placement(fov: [f32; 4]) -> Placement {
    let [left, right, up, down] = fov;
    let (tan_l, tan_r) = (left.tan(), right.tan());
    let (tan_u, tan_d) = (up.tan(), down.tan());
    let (span_x, span_y) = (tan_r - tan_l, tan_u - tan_d);
    Placement::affine([
        1.0,
        1.0,
        -(tan_r + tan_l) / span_x,
        -(tan_u + tan_d) / span_y,
    ])
}

/// The camera's pinhole model in the head's frame, from what Camera2
/// reports, or `None` when it reports too little to place by.
fn camera_model(calibration: &camera::Calibration) -> Option<CameraModel> {
    let intrinsics = calibration.intrinsics?;
    let array = calibration
        .pre_correction_array
        .or(calibration.active_array)?;
    // Android's rotation turns the sensor's frame (x right, y up, z toward
    // the wearer, which is OpenXR's view space) into the camera's; we need
    // the camera's axes in the head's, its inverse.
    let rotation = calibration.rotation?;
    // The gyroscope is the headset's own frame; any other reference would
    // put the pose somewhere we cannot relate to the eyes.
    if calibration.pose_reference != Some(1) {
        warn!(
            "camera pose is relative to {:?}, not the gyroscope; overlays may be off",
            calibration.pose_reference
        );
    }
    if calibration
        .distortion
        .is_some_and(|d| d.iter().any(|&k| k != 0.0))
    {
        warn!("camera reports lens distortion, which placement ignores");
    }
    let model = CameraModel::from_sensor(
        intrinsics,
        [array[2] as u32, array[3] as u32],
        [CAPTURE_SIZE.0 as u32, CAPTURE_SIZE.1 as u32],
        projection::conjugate(rotation),
        calibration.translation.unwrap_or_default(),
    );
    let axis = projection::rotate(model.orientation, [0.0, 0.0, 1.0]);
    info!(
        "camera model: focal {:?}, principal {:?}, at {:?}, looking {:.1}° down, {:.1}° right",
        model.focal,
        model.principal,
        model.position,
        (-axis[1]).asin().to_degrees(),
        axis[0].atan2(-axis[2]).to_degrees(),
    );
    Some(model)
}

/// Log `depth` and the head pose it goes with, for `tools/env_depth.py`:
/// the app cannot write where `adb` reads, but its log reaches the host.
/// Every fourth sample of every fourth row, as 16-bit hex.
fn dump_depth(depth: &EnvironmentDepth, head: &Pose) {
    info!(
        "envdump size {} {} near {} far {} head {:?} {:?}",
        depth.size[0] / 4,
        depth.size[1] / 4,
        depth.near,
        depth.far,
        head.orientation,
        head.position
    );
    for view in &depth.views {
        info!(
            "envdump view {:?} {:?} {:?}",
            view.pose.orientation, view.pose.position, view.fov
        );
    }
    let [w, h] = depth.size.map(|v| v as usize);
    for layer in 0..2 {
        for y in (0..h).step_by(4) {
            let row: String = (0..w)
                .step_by(4)
                .map(|x| {
                    let raw = depth.raw[(layer * h + y) * w + x];
                    format!("{:04x}", (raw.clamp(0.0, 1.0) * 65535.0).round() as u16)
                })
                .collect();
            info!("envdump row {layer} {} {row}", y / 4);
        }
    }
}

/// Debugging switch for the passthrough colour map's encoding, which the
/// extension does not document: with this file present the map is
/// converted from sRGB-encoded values to linear ones.
const LINEAR_LUT_PATH: &str = "/data/local/tmp/red_eye/lut_linear";

/// The system's passthrough, recoloured by the grade: stereo-correct,
/// reprojected with the head at display rate, and free for our GPU. `None`
/// where the runtime lacks `XR_FB_passthrough`; the app then draws the
/// left camera itself, as before.
fn create_passthrough(context: &gpu::Context, grade: &Grade) -> Option<gpu::xr::Passthrough> {
    let mut lut = grade.mono_to_rgba_lut();
    let linear = std::path::Path::new(LINEAR_LUT_PATH).exists();
    if linear {
        let to_linear = |c: f32| {
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        for entry in lut.iter_mut() {
            for c in &mut entry[..3] {
                *c = to_linear(*c);
            }
        }
    }
    let style = gpu::xr::PassthroughStyle {
        opacity: 1.0,
        edge_color: [0.0; 4],
        color_map: gpu::xr::PassthroughColorMap::MonoToRgba(Box::new(lut)),
    };
    let passthrough = context.create_xr_passthrough(&style);
    match passthrough {
        Some(_) => info!(
            "compositor passthrough, grade LUT {}",
            if linear { "linearized" } else { "as encoded" }
        ),
        None => warn!("no compositor passthrough; drawing the camera image"),
    }
    passthrough
}

fn is_srgb(format: gpu::TextureFormat) -> bool {
    matches!(
        format,
        gpu::TextureFormat::Rgba8UnormSrgb | gpu::TextureFormat::Bgra8UnormSrgb
    )
}

/// Where this frame renders: a projection layer under the passthrough's,
/// or the single layer `present` ends the frame with.
enum Output {
    Layered {
        frame: gpu::xr::Frame,
        image: gpu::xr::Image,
    },
    Presented(gpu::Frame),
}

struct App {
    surface: gpu::xr::Surface,
    passthrough: Option<gpu::xr::Passthrough>,
    source: Box<dyn FrameSource>,
    /// The open camera's model, for placing what it saw in each eye.
    camera_model: Option<CameraModel>,
    /// The headset's sensed depth, when the runtime offers it and
    /// USE_SCENE is granted.
    environment: Option<gpu::xr::EnvironmentDepth>,
    resample: Resample,
    /// The head's pose in base space when the resample being read was
    /// recorded, and that resample's result.
    resample_head: Pose,
    environment_depth: Option<(EnvironmentDepth, Pose)>,
    dumped_depth: bool,
    /// Camera depth of the plane overlays are placed on, following the
    /// outlined object.
    placement_distance: f32,
    camera_active: bool,
    grade_pass: GradePass,
    grade: Grade,
    linear_target: bool,
    box_pass: BoxPass,
    wireframe: WireframePass,
    depth: Option<Background<DepthMap>>,
    depth_map: Option<DepthMap>,
    depth_generation: u64,
    last_depth: Instant,
    depths: u64,
    depth_hz: f32,
    /// The wireframe's style for the mask last uploaded; `None` until both
    /// depth and something to outline exist.
    wireframe_style: Option<WireframeStyle>,
    hud_canvas: Canvas,
    hud: Hud,
    tracks: Vec<Track>,
    last_target_poll: Instant,
    detect_hz: f32,
    render_hz: f32,
    /// Seconds of rates measured since the last log line.
    reports: u32,
    detector: Option<Worker>,
    last_detect: Instant,
    shown_generation: u64,
    detections: u64,
    detect_latency_ms: f32,
    start: Instant,
    /// The previous submission. The grade pass reuses one staging buffer,
    /// so its upload must not be overwritten while still in flight.
    last_submit: Option<gpu::SyncPoint>,
    frames: u64,
    last_report: Instant,
}

impl App {
    fn new(context: &Arc<gpu::Context>) -> Self {
        let surface = context
            .create_xr_surface()
            .expect("unable to create XR surface");
        let format = surface.format();
        info!("XR swapchain format {format:?}");
        let (source, camera_model) = Self::open_camera();
        let grade = Grade::default();
        let passthrough = create_passthrough(context, &grade);
        let environment = context.create_xr_environment_depth();
        match environment {
            Some(ref environment) => {
                if let Err(e) = environment.start() {
                    warn!("environment depth would not start: {e}");
                }
                // Hands are not objects to range.
                if let Err(e) = environment.set_hand_removal(true) {
                    warn!("environment depth hand removal: {e}");
                }
                info!("environment depth started");
            }
            None => warn!(
                "no environment depth; grant it with: adb shell pm grant \
                 rust.red_eye_xr com.oculus.permission.USE_SCENE"
            ),
        }
        Self {
            passthrough,
            environment,
            resample: Resample::new(context),
            resample_head: Pose::IDENTITY,
            environment_depth: None,
            dumped_depth: false,
            placement_distance: PLACEMENT_DISTANCE,
            grade_pass: GradePass::new(
                context,
                format,
                [CAPTURE_SIZE.0 as u32, CAPTURE_SIZE.1 as u32],
            ),
            linear_target: is_srgb(format),
            box_pass: BoxPass::new(context, format),
            wireframe: WireframePass::new(context, format),
            depth: spawn_depth(context),
            depth_map: None,
            depth_generation: 0,
            last_depth: Instant::now(),
            depths: 0,
            depth_hz: 0.0,
            wireframe_style: None,
            hud_canvas: Canvas::new(context, HUD_SIZE, 1.0, format),
            hud: Hud::default(),
            tracks: Vec::new(),
            last_target_poll: Instant::now() - TARGET_POLL,
            detect_hz: 0.0,
            render_hz: 0.0,
            reports: 0,
            detector: spawn_detector(context),
            last_detect: Instant::now(),
            shown_generation: 0,
            detections: 0,
            detect_latency_ms: 0.0,
            surface,
            source,
            camera_model,
            camera_active: true,
            grade,
            start: Instant::now(),
            last_submit: None,
            frames: 0,
            last_report: Instant::now(),
        }
    }

    /// The left passthrough camera, or the test pattern if it will not
    /// open: a missing permission should look like "no camera", not like a
    /// crash or a black screen.
    fn open_camera() -> (Box<dyn FrameSource>, Option<CameraModel>) {
        match camera::PassthroughCamera::new(Eye::Left, CAPTURE_SIZE) {
            Ok(camera) => {
                let model = camera_model(camera.calibration());
                (Box::new(camera), model)
            }
            Err(e) => {
                warn!(
                    "camera unavailable ({e}); showing the test pattern. Grant it with: \
                     adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA"
                );
                (
                    Box::new(TestPattern::new(
                        CAPTURE_SIZE.0 as u32,
                        CAPTURE_SIZE.1 as u32,
                    )),
                    None,
                )
            }
        }
    }

    /// Hold the camera only while focused. Android revokes it from
    /// background apps anyway, and a headset camera should not keep
    /// streaming to an app the wearer has left.
    fn set_camera_active(&mut self, active: bool) {
        if active == self.camera_active {
            return;
        }
        self.camera_active = active;
        // Passthrough's cameras stop with ours: nothing to show unfocused.
        if let Some(ref passthrough) = self.passthrough {
            let result = if active {
                passthrough.resume()
            } else {
                passthrough.pause()
            };
            if let Err(e) = result {
                warn!(
                    "passthrough {}: {e}",
                    if active { "resume" } else { "pause" }
                );
            }
        }
        if let Some(ref environment) = self.environment {
            let result = if active {
                environment.start()
            } else {
                environment.stop()
            };
            if let Err(e) = result {
                warn!("environment depth: {e}");
            }
        }
        if active {
            (self.source, self.camera_model) = Self::open_camera();
        } else {
            info!("releasing the camera on focus loss");
            self.source = Box::new(TestPattern::new(
                CAPTURE_SIZE.0 as u32,
                CAPTURE_SIZE.1 as u32,
            ));
        }
    }

    fn render(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        let output = if self.passthrough.is_some() {
            let Some(frame) = context.xr_begin_frame() else {
                return;
            };
            let Some(image) = self.surface.acquire_image() else {
                // The frame has begun, so it ends, with nothing in it.
                let _ = context.xr_end_frame(frame, gpu::xr::BlendMode::Opaque, &[]);
                return;
            };
            Output::Layered { frame, image }
        } else {
            let Some(frame) = self.surface.acquire_frame(context) else {
                return;
            };
            Output::Presented(frame)
        };
        if let Some(sync) = self.last_submit.take() {
            let _ = context.wait_for(&sync, !0);
        }
        // The previous frame's resample has run, so its result is ready.
        let mut environment_changed = false;
        if let Some(depth) = self.resample.read() {
            // Once per run: the app cannot remove the flag.
            if !self.dumped_depth && std::path::Path::new(DUMP_DEPTH_PATH).exists() {
                dump_depth(&depth, &self.resample_head);
                self.dumped_depth = true;
            }
            self.environment_depth = Some((depth, self.resample_head));
            environment_changed = true;
        }
        let layered = matches!(output, Output::Layered { .. });
        let (texture, views) = match output {
            Output::Layered {
                ref frame,
                ref image,
            } => (
                image.texture(),
                frame
                    .views()
                    .iter()
                    .enumerate()
                    .map(|(eye, view)| (*view, image.layer_view(eye as u32)))
                    .collect::<Vec<_>>(),
            ),
            Output::Presented(ref frame) => (
                frame.texture(),
                (0..frame.xr_view_count())
                    .map(|eye| (frame.xr_view(eye), frame.xr_texture_view(eye)))
                    .collect(),
            ),
        };

        // The eyes' poses relative to the head, which is where the camera's
        // calibration places it.
        let pose = |v: &gpu::xr::View| Pose {
            orientation: v.pose.orientation,
            position: v.pose.position,
        };
        let head = match views.as_slice() {
            [(left, _), (right, _), ..] => projection::head_from_eyes(&pose(left), &pose(right)),
            [(only, _)] => pose(only),
            [] => Pose::IDENTITY,
        };

        encoder.start();
        encoder.init_texture(texture);
        // Resample the headset's depth for the CPU, read next frame.
        if let (Some(environment), Output::Layered { frame, .. }) = (&self.environment, &output)
            && let Some(image) = environment.acquire(context, frame)
        {
            let view = |v: &gpu::xr::View| DepthView {
                pose: pose(v),
                fov: [
                    v.fov.angle_left,
                    v.fov.angle_right,
                    v.fov.angle_up,
                    v.fov.angle_down,
                ],
            };
            encoder.begin_xr_depth(&image);
            self.resample.record(
                encoder,
                image.view,
                image.views.each_ref().map(view),
                image.near,
                image.far,
            );
            encoder.end_xr_depth(&image);
            self.resample_head = head;
        }
        if let Some(camera_frame) = self.source.next_frame() {
            // The compositor draws the world under passthrough; the camera
            // image only feeds detection then.
            if !layered {
                self.grade_pass.upload(context, encoder, camera_frame);
            }
            // Offer the frame for detection; a busy worker drops it.
            if let Some(worker) = &self.detector
                && self.last_detect.elapsed() >= DETECT_INTERVAL
                && worker.submit(camera_frame)
            {
                self.last_detect = Instant::now();
            }
            if let Some(depth) = &self.depth
                && self.last_depth.elapsed() >= DEPTH_INTERVAL
                && depth.submit(camera_frame)
            {
                self.last_depth = Instant::now();
            }
        }
        let mut tracks_changed = false;
        // The previous submission has completed (waited on above), so the
        // box buffer is free to rewrite.
        if let Some(worker) = &self.detector
            && worker.generation() != self.shown_generation
            && let Some(observation) = worker.latest()
        {
            self.shown_generation = worker.generation();
            self.detections += 1;
            self.detect_latency_ms = observation.latency_ms;
            self.tracks = observation.tracks;
            // Debug boxes only while scanning everything; a named target
            // gets the HUD's brackets instead.
            let boxes: Vec<_> = if self.hud.targeting.target().is_some() {
                Vec::new()
            } else {
                self.tracks
                    .iter()
                    .filter(|t| t.is_confirmed(CONFIRM_HITS))
                    .map(|t| {
                        BoxInstance::new(&t.rect, red_eye::hud::outline_color(t.label.category()))
                    })
                    .collect()
            };
            self.box_pass.set_boxes(context, &boxes);
            tracks_changed = true;
        }
        if let Some(depth) = &self.depth
            && depth.generation() != self.depth_generation
            && let Some(map) = depth.latest()
        {
            self.depth_generation = depth.generation();
            self.depths += 1;
            self.depth_map = Some(map);
            tracks_changed = true;
        }
        if tracks_changed || environment_changed {
            self.update_ranges();
        }
        // The previous submission has completed, so the wireframe's staging
        // buffers are free to rewrite.
        if tracks_changed {
            self.update_wireframe(context, encoder);
        }

        let time = self.start.elapsed().as_secs_f32();
        if self.last_target_poll.elapsed() >= TARGET_POLL {
            self.last_target_poll = Instant::now();
            self.hud.set_target(read_target(), time);
        }
        // Lay out the HUD once; each eye composites it.
        let telemetry = Telemetry {
            render_hz: self.render_hz,
            detect_hz: self.detect_hz,
            latency_ms: self.detect_latency_ms,
        };
        let (hud, tracks) = (&mut self.hud, &self.tracks);
        self.hud_canvas.render(context, encoder, time as f64, |ui| {
            hud.draw(ui, time, tracks, &telemetry)
        });
        // Under passthrough, clear to nothing so the world shows through.
        let clear = if layered {
            gpu::TextureColor::TransparentBlack
        } else {
            gpu::TextureColor::OpaqueBlack
        };
        for &(ref xr_view, view) in &views {
            let fov = [
                xr_view.fov.angle_left,
                xr_view.fov.angle_right,
                xr_view.fov.angle_up,
                xr_view.fov.angle_down,
            ];
            let eye = pose(xr_view).relative_to(&head);
            let mut pass = encoder.render(
                "eye",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view,
                        init_op: gpu::InitOp::Clear(clear),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            let placement = match self.camera_model {
                Some(ref camera) => {
                    projection::placement(camera, &eye, fov, self.placement_distance)
                }
                None => eye_placement(fov),
            };
            if !layered {
                self.grade_pass
                    .draw(&mut pass, &self.grade, placement, time, self.linear_target);
            }
            let extent = self.surface.extent();
            self.box_pass.draw(
                &mut pass,
                placement,
                [extent.width, extent.height],
                BOX_THICKNESS,
            );
            if let Some(ref style) = self.wireframe_style {
                self.wireframe
                    .draw(&mut pass, placement, style, self.linear_target);
            }
            // Placed like the camera image, so track rectangles line up.
            self.hud_canvas
                .composite(&mut pass, placement, self.linear_target, 1.0);
        }

        match output {
            Output::Layered { frame, image } => {
                encoder.end_xr_image(&image);
                let sync = context.submit(encoder);
                self.hud_canvas.after_submit(&sync);
                // The image goes back to the runtime once it is rendered,
                // before the frame that shows it ends.
                let _ = context.wait_for(&sync, !0);
                self.surface.release_image();
                let passthrough = self.passthrough.as_ref().expect("layered output");
                let layers = [
                    passthrough.layer(),
                    gpu::xr::Layer::Projection(gpu::xr::ProjectionLayer {
                        surface: &self.surface,
                        views: None,
                        space: None,
                        flags: gpu::xr::LayerFlags::BLEND_SOURCE_ALPHA,
                    }),
                ];
                if let Err(e) = context.xr_end_frame(frame, gpu::xr::BlendMode::Opaque, &layers) {
                    warn!("XR frame end failed: {e}");
                }
            }
            Output::Presented(frame) => {
                encoder.present(frame);
                let sync = context.submit(encoder);
                self.hud_canvas.after_submit(&sync);
                self.last_submit = Some(sync);
            }
        }
        self.frames += 1;

        // Rates refresh every second for the HUD; the log gets every fifth.
        if self.last_report.elapsed() >= Duration::from_secs(1) {
            let secs = self.last_report.elapsed().as_secs_f64();
            self.render_hz = (self.frames as f64 / secs) as f32;
            self.detect_hz = (self.detections as f64 / secs) as f32;
            self.depth_hz = (self.depths as f64 / secs) as f32;
            self.depths = 0;
            self.reports += 1;
            if self.reports.is_multiple_of(5) {
                // What the headset senses straight ahead, a sanity check.
                let ahead = self.environment_depth.as_ref().and_then(|(depth, head)| {
                    let direction = projection::rotate(head.orientation, [0.0, 0.0, -1.0]);
                    depth.along(0, head.position, direction)
                });
                // Which way up the depth image is, judged by the camera.
                if let (Some((depth, head)), Some(camera), Some(relative)) =
                    (&self.environment_depth, &self.camera_model, &self.depth_map)
                {
                    let agree =
                        |d: &EnvironmentDepth| environment::agreement(d, camera, head, relative);
                    info!(
                        "environment depth agrees with the camera: rows top-down {:.2?},                          bottom-up {:.2?}",
                        agree(depth),
                        agree(&depth.flipped())
                    );
                }
                info!(
                    "render {:.1} Hz | detect {:.1} Hz, latency {:.1} ms | depth {:.1} Hz | \
                     ahead {ahead:.2?} m, plane {:.2} m, ranges {:.1?} | {:?}",
                    self.render_hz,
                    self.detect_hz,
                    self.detect_latency_ms,
                    self.depth_hz,
                    self.placement_distance,
                    self.hud.ranges.values().collect::<Vec<_>>(),
                    self.hud.targeting.phase(),
                );
            }
            self.frames = 0;
            self.detections = 0;
            self.last_report = Instant::now();
        }
    }

    /// The object the HUD is about: the matched target when there is one,
    /// else the one the wearer faces.
    fn outlined(&self) -> Option<&Track> {
        let target = match self.hud.targeting.phase() {
            red_eye::hud::targeting::Phase::Match { track, .. } => {
                self.tracks.iter().find(|t| t.id == track)
            }
            _ => None,
        };
        // Without eye tracking, the wearer's focus is where the head points.
        let focus = self
            .camera_model
            .and_then(|camera| camera.focus(self.placement_distance))
            .unwrap_or([0.5, 0.5]);
        target.or_else(|| track::focused(&self.tracks, focus, CONFIRM_HITS))
    }

    /// Range every confirmed track through the headset's depth, and move
    /// the placement plane to the outlined object.
    fn update_ranges(&mut self) {
        let (Some((depth, head)), Some(camera)) = (&self.environment_depth, &self.camera_model)
        else {
            return;
        };
        let mut ranges = std::collections::HashMap::new();
        let mut placement = None;
        let outlined = self.outlined().map(|t| t.id);
        for t in self.tracks.iter().filter(|t| t.is_confirmed(CONFIRM_HITS)) {
            let Some((camera_depth, point)) =
                environment::object_depth(depth, camera, head, t.rect)
            else {
                continue;
            };
            let offset = [0, 1, 2].map(|i| point[i] - head.position[i]);
            let range = offset.iter().map(|v| v * v).sum::<f32>().sqrt();
            let smoothed = match self.hud.ranges.get(&t.id) {
                Some(&old) => old + RANGE_SMOOTHING * (range - old),
                None => range,
            };
            ranges.insert(t.id, smoothed);
            if Some(t.id) == outlined {
                placement = Some(camera_depth);
            }
        }
        self.hud.ranges = ranges;
        if let Some(target) = placement {
            let target = target.clamp(*PLACEMENT_RANGE.start(), *PLACEMENT_RANGE.end());
            self.placement_distance += RANGE_SMOOTHING * (target - self.placement_distance);
        }
    }

    /// Outline one object: the matched target when there is one, else the
    /// one the wearer faces. Contours on everything at once read as noise,
    /// and the wireframe is the HUD saying "this one". Its silhouette is cut
    /// from its box by depth, and the contours spaced over its depth range.
    fn update_wireframe(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        let Some(ref depth) = self.depth_map else {
            return;
        };
        let chosen = self.outlined();
        let boxes: Vec<(u8, red_eye::perception::Rect)> =
            chosen.map(|t| (1, t.rect)).into_iter().collect();
        let highlight = 1;
        let mask: Mask = dense::silhouettes(&boxes, depth, MASK_SIZE[0], MASK_SIZE[1]);
        let Some((lo, hi)) = dense::masked_range(&mask, depth) else {
            self.wireframe_style = None;
            return;
        };
        self.wireframe.upload(context, encoder, &mask, depth);
        let time = self.start.elapsed().as_secs_f32();
        self.wireframe_style = Some(WireframeStyle {
            spacing: ((hi - lo) / CONTOURS).max(1e-3),
            // A slow drift, for the scanning look.
            phase: (time * 0.25).fract(),
            highlight,
            ..WireframeStyle::default()
        });
    }

    fn destroy(mut self, context: &gpu::Context) {
        // Join the worker first: it holds the context and may be mid-submit.
        drop(self.detector.take());
        drop(self.depth.take());
        if let Some(sync) = self.last_submit.take() {
            let _ = context.wait_for(&sync, !0);
        }
        // Its layer belongs to the session, which ends after this.
        drop(self.passthrough.take());
        if let Some(mut environment) = self.environment.take() {
            let _ = environment.stop();
            context.destroy_xr_environment_depth(&mut environment);
        }
        self.resample.destroy(context);
        self.grade_pass.destroy(context);
        self.box_pass.destroy(context);
        self.wireframe.destroy(context);
        self.hud_canvas.destroy(context);
        context.destroy_xr_surface(&mut self.surface);
    }
}

fn spawn_event_pump() -> Arc<AtomicBool> {
    let should_exit = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&should_exit);
    std::thread::spawn(move || {
        while let Some(event) = ndk_glue::poll_events() {
            if matches!(event, ndk_glue::Event::Destroy) {
                flag.store(true, Ordering::Relaxed);
                break;
            }
        }
    });
    should_exit
}

#[ndk_glue::main]
pub fn main() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("red_eye"),
    );
    std::panic::set_hook(Box::new(|info| log::error!("panic: {info}")));
    info!("=== red_eye starting ===");

    let entry = unsafe { xr::Entry::load().expect("no OpenXR loader") };
    entry.initialize_android_loader().unwrap();

    let available = entry.enumerate_extensions().unwrap();
    assert!(
        available.khr_vulkan_enable2,
        "runtime lacks XR_KHR_vulkan_enable2"
    );
    info!(
        "XR_FB_passthrough {}, XR_META_environment_depth {}",
        available.fb_passthrough, available.meta_environment_depth
    );
    let mut extensions = xr::ExtensionSet::default();
    extensions.khr_vulkan_enable2 = true;
    extensions.khr_android_create_instance = true;
    extensions.fb_passthrough = available.fb_passthrough;
    extensions.meta_environment_depth = available.meta_environment_depth;

    let xr_instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "RedEye",
                application_version: 0,
                engine_name: "Blade",
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            &extensions,
            &[],
        )
        .unwrap();
    let system = xr_instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .unwrap();

    // One context, shared by the renderer and (soon) by Meganeura.
    let context = red_eye::init_context(Some(gpu::xr::Desc {
        instance: xr_instance.clone(),
        system_id: system,
    }))
    .expect("failed to initialize GPU context");

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: "red_eye",
        buffer_count: 2,
        manual_barriers: false,
    });

    let mut app: Option<App> = None;
    let should_exit = spawn_event_pump();
    let mut events = xr::EventDataBuffer::new();

    'main: loop {
        if should_exit.load(Ordering::Relaxed) {
            break 'main;
        }

        while let Some(event) = xr_instance.poll_event(&mut events).unwrap() {
            use xr::Event::*;
            match event {
                SessionStateChanged(e) => {
                    info!("XR session state: {:?}", e.state());
                    match e.state() {
                        xr::SessionState::READY => {
                            if app.is_none() {
                                app = Some(App::new(&context));
                            }
                            context.xr_session().unwrap().begin(VIEW_TYPE).unwrap();
                        }
                        xr::SessionState::STOPPING => {
                            context.xr_session().unwrap().end().unwrap();
                            if let Some(app) = app.take() {
                                app.destroy(&context);
                            }
                        }
                        xr::SessionState::VISIBLE | xr::SessionState::SYNCHRONIZED => {
                            if let Some(app) = app.as_mut() {
                                app.set_camera_active(false);
                            }
                        }
                        xr::SessionState::FOCUSED => {
                            if let Some(app) = app.as_mut() {
                                app.set_camera_active(true);
                            }
                        }
                        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => break 'main,
                        _ => {}
                    }
                }
                InstanceLossPending(_) => break 'main,
                _ => {}
            }
        }

        match &mut app {
            Some(app) => app.render(&context, &mut encoder),
            // Not in session yet; idle rather than spin.
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }

    if let Some(app) = app.take() {
        app.destroy(&context);
    }
    context.destroy_command_encoder(&mut encoder);
    info!("=== red_eye stopped ===");
}
