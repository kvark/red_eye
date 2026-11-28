//! The XR app every MAR vision runs in.
//!
//! A vision ([`Vision`]) decides how the world looks and what is drawn over
//! it; the shell does everything else, the same for every vision:
//!
//! - the OpenXR session and its frames, with the system's passthrough under
//!   our layer (or, without `XR_FB_passthrough`, an opaque layer the vision
//!   draws the camera into);
//! - the left passthrough camera and its calibration, which places
//!   camera-image overlays in each eye ([`mar_core::projection`]);
//! - the headset's environment depth, read back each frame;
//! - detection and tracking, monocular depth and voice commands, each on a
//!   thread of its own sharing the GPU context, never holding up a frame;
//! - focus: camera, microphone, passthrough and depth run only while the
//!   app has it;
//! - rates for the HUD and a periodic log line.
//!
//! What a vision gets to see each frame is [`Senses`]. Models and switches
//! live in the vision's data directory, pushed with `adb`; anything missing
//! is logged and done without.

#![cfg(target_os = "android")]

mod camera;
mod listen;
mod microphone;

pub use listen::Heard;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use blade_graphics as gpu;
use log::{info, warn};
use mar_core::frame::{Eye, Frame};
use mar_core::perception::Detector;
use mar_core::perception::dense::DepthMap;
use mar_core::perception::depth::DepthAnything;
use mar_core::perception::dfine::Dfine;
use mar_core::perception::environment::{DepthView, EnvironmentDepth, Resample};
use mar_core::perception::track::{Track, TrackerConfig};
use mar_core::perception::worker::{Background, Worker};
use mar_core::perception::yolox::{Config as YoloxConfig, Yolox};
use mar_core::projection::{self, CameraModel, Pose};
use mar_core::render::Placement;
use mar_core::source::{FrameSource, TestPattern};
use openxr as xr;

const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

/// The camera stream: what the Quest 3/3S passthrough cameras offer.
pub const CAPTURE_SIZE: [u32; 2] = [1280, 960];

/// D-FINE's input, as `tools/fetch_dfine.sh` prepares it.
const DFINE_SIZE: [u32; 2] = [640, 480];

/// Depth Anything V2 Small's input, as `tools/fetch_depth.sh 224 168`
/// prepares it: overlays need coarse depth only, so the smallest that keeps
/// the cameras' 4:3 in 14-pixel patches.
const DEPTH_SIZE: [u32; 2] = [224, 168];

/// Least time between depth frames: it shares the GPU with detection and
/// the display, and its users tolerate stale depth. Every 250 ms halved the
/// display rate on a Quest 3S.
const DEPTH_INTERVAL: Duration = Duration::from_millis(500);

/// Upper bound on detection rate: the 24 Hz target. Submitting faster would
/// only take GPU time from the display.
const DETECT_INTERVAL: Duration = Duration::from_millis(41);

/// Meganeura splits each network into this many queue submissions so the
/// renderer's submissions can interleave. Two was best in DinoVision's
/// co-tenancy sweep.
const SUBMISSION_CHUNKS: usize = 2;

/// Who a vision is, and where its files are.
pub struct Config {
    /// Shown in the log and given to OpenXR.
    pub name: &'static str,
    /// The Android log tag.
    pub log_tag: &'static str,
    /// The Android package, for the `adb` hints in the log.
    pub package: &'static str,
    /// Where models and switches are pushed with `adb`:
    /// `dfine_nano.onnx` or `yolox_nano.safetensors` for detection,
    /// `depth_small.onnx` for depth, `moonshine/` for voice.
    pub data_dir: &'static str,
}

/// Rates the shell measures, refreshed every second.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rates {
    pub render_hz: f32,
    pub detect_hz: f32,
    pub depth_hz: f32,
    /// Capture to tracks, of the latest detection.
    pub detect_latency_ms: f32,
}

/// What the shell perceived, as of this frame.
pub struct Senses<'a> {
    /// Seconds since the vision started.
    pub time: f32,
    /// Whether the compositor draws the world under our layer. Without it
    /// the vision draws the camera image itself.
    pub passthrough: bool,
    /// The newest camera frame, when one arrived since the last frame.
    pub frame: Option<&'a Frame>,
    /// The open camera's model, when Camera2 reports its calibration.
    pub camera: Option<&'a CameraModel>,
    /// The head's pose in the XR base space, whose `y` points up, as of
    /// this frame.
    pub head: Pose,
    /// The latest tracks, and whether they changed this frame.
    pub tracks: &'a [Track],
    pub tracks_changed: bool,
    /// Monocular relative depth over the camera image, and whether it
    /// changed this frame.
    pub depth: Option<&'a DepthMap>,
    pub depth_changed: bool,
    /// The headset's metric depth with the head pose it goes with, and
    /// whether it changed this frame.
    pub environment: Option<&'a (EnvironmentDepth, Pose)>,
    pub environment_changed: bool,
    /// A voice command heard since the last frame.
    pub heard: Option<Heard>,
    /// Someone is speaking now.
    pub speaking: bool,
    pub rates: Rates,
}

/// One eye being drawn.
pub struct EyeView {
    /// Carries the camera image, and anything drawn in its coordinates,
    /// onto this eye.
    pub placement: Placement,
    /// The eye buffer's size in pixels.
    pub extent: [u32; 2],
    /// The target encodes sRGB itself, so shaders write linear values.
    pub linear_target: bool,
    /// See [`Senses::passthrough`].
    pub passthrough: bool,
}

/// A look at the world: one MAR app.
pub trait Vision: Sized {
    fn config() -> Config;

    /// Called once the XR session is ready; `format` is the eye buffers'.
    fn new(context: &Arc<gpu::Context>, format: gpu::TextureFormat) -> Self;

    /// How the compositor's passthrough shows the world.
    fn passthrough_style(&self) -> gpu::xr::PassthroughStyle;

    /// A new passthrough style, when the vision wants one: polled after
    /// each [`Vision::update`]. Restyling sends a new colour map to the
    /// compositor, so it should be occasional.
    fn restyle_passthrough(&mut self) -> Option<gpu::xr::PassthroughStyle> {
        None
    }

    /// React to what was sensed and record this frame's uploads. The
    /// previous frame's GPU work has finished, so its buffers are free.
    fn update(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        senses: &Senses,
    );

    /// Depth along the camera's axis, in metres, of the plane camera-image
    /// overlays are placed on: they line up exactly with things at that
    /// depth.
    fn placement_distance(&self) -> f32 {
        1.5
    }

    /// Draw into one eye; called for each after [`Vision::update`].
    fn draw(&mut self, pass: &mut gpu::RenderCommandEncoder, eye: &EyeView);

    /// The frame's work was submitted.
    fn after_submit(&mut self, _sync: &gpu::SyncPoint) {}

    /// Appended to the shell's periodic log line.
    fn status(&self) -> String {
        String::new()
    }

    fn destroy(self, context: &gpu::Context);
}

fn spawn_detector(dir: &Path, context: &Arc<gpu::Context>) -> Option<Worker> {
    let dfine = dir.join("dfine_nano.onnx");
    let yolox = dir.join("yolox_nano.safetensors");
    if !dfine.exists() && !yolox.exists() {
        warn!(
            "no detector at {} or {}; running without detection",
            dfine.display(),
            yolox.display()
        );
        return None;
    }
    let context = Arc::clone(context);
    Some(Worker::spawn(
        move || {
            // D-FINE is far more accurate, at about six times YOLOX-Nano's
            // compute; YOLOX-Nano is the light fallback.
            let model: Box<dyn Detector> = if dfine.exists() {
                let mut model = Dfine::new(&dfine, DFINE_SIZE, Some(context))?;
                model.session().set_submission_chunks(SUBMISSION_CHUNKS);
                info!(
                    "detector ready: D-FINE-N at {}x{}",
                    DFINE_SIZE[0], DFINE_SIZE[1]
                );
                Box::new(model)
            } else {
                let config = YoloxConfig::nano().at_input(416, 320);
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
fn spawn_depth(dir: &Path, context: &Arc<gpu::Context>) -> Option<Background<DepthMap>> {
    let path = dir.join("depth_small.onnx");
    if !path.exists() {
        warn!("no depth model at {}; no monocular depth", path.display());
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
        Ok(move |frame: &Frame| model.run_frame(frame))
    }))
}

/// Place the camera image in an eye whose frustum is `[left, right, up,
/// down]` radians, for a camera without calibration: span the frustum
/// symmetrically about the eye's axis.
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
        CAPTURE_SIZE,
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

/// The system's passthrough in the vision's style: stereo-correct,
/// reprojected with the head at display rate, and free for our GPU. `None`
/// where the runtime lacks `XR_FB_passthrough`; the vision then draws the
/// camera itself.
///
/// The extension does not say how its colour map is encoded; with
/// `lut_linear` in the data directory, a mono-to-RGBA map is converted from
/// sRGB-encoded values to linear ones, to compare the two in the headset.
fn create_passthrough(
    context: &gpu::Context,
    style: gpu::xr::PassthroughStyle,
    dir: &Path,
) -> Option<gpu::xr::Passthrough> {
    let linear = dir.join("lut_linear").exists();
    let passthrough = context.create_xr_passthrough(&encode_style(style, linear));
    match passthrough {
        Some(_) => info!(
            "compositor passthrough, colour map {}",
            if linear { "linearized" } else { "as encoded" }
        ),
        None => warn!("no compositor passthrough; the vision draws the camera image"),
    }
    passthrough
}

/// `style` as the compositor takes it: a mono-to-RGBA map converted to
/// linear values when `linear`.
fn encode_style(mut style: gpu::xr::PassthroughStyle, linear: bool) -> gpu::xr::PassthroughStyle {
    if let (true, gpu::xr::PassthroughColorMap::MonoToRgba(lut)) = (linear, &mut style.color_map) {
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
    style
}

fn is_srgb(format: gpu::TextureFormat) -> bool {
    matches!(
        format,
        gpu::TextureFormat::Rgba8UnormSrgb | gpu::TextureFormat::Bgra8UnormSrgb
    )
}

/// Where this frame renders: a projection layer over the passthrough's, or
/// the single layer `present` ends the frame with.
enum Output {
    Layered {
        frame: gpu::xr::Frame,
        image: gpu::xr::Image,
    },
    Presented(gpu::Frame),
}

/// A vision in the shell, for the length of an XR session.
struct Shell<V> {
    vision: V,
    data_dir: PathBuf,
    package: &'static str,
    surface: gpu::xr::Surface,
    passthrough: Option<gpu::xr::Passthrough>,
    source: Box<dyn FrameSource>,
    /// The open camera's model, for placing what it saw in each eye.
    camera_model: Option<CameraModel>,
    camera_active: bool,
    linear_target: bool,
    /// The headset's sensed depth, when the runtime offers it and
    /// USE_SCENE is granted.
    environment: Option<gpu::xr::EnvironmentDepth>,
    resample: Resample,
    /// The head's pose in base space when the resample being read was
    /// recorded, and that resample's result.
    resample_head: Pose,
    environment_depth: Option<(EnvironmentDepth, Pose)>,
    dumped_depth: bool,
    detector: Option<Worker>,
    last_detect: Instant,
    shown_generation: u64,
    tracks: Vec<Track>,
    detections: u64,
    depth: Option<Background<DepthMap>>,
    depth_map: Option<DepthMap>,
    depth_generation: u64,
    last_depth: Instant,
    depths: u64,
    /// Voice commands, if the model was pushed.
    listener: Option<listen::Listener>,
    heard_generation: u64,
    rates: Rates,
    /// Seconds of rates measured since the last log line.
    reports: u32,
    start: Instant,
    /// The previous submission, waited on before its buffers are reused.
    last_submit: Option<gpu::SyncPoint>,
    frames: u64,
    /// Time in the vision's update since the last report: total and worst,
    /// in milliseconds. The update runs on the display thread.
    update_ms: (f32, f32),
    last_report: Instant,
}

impl<V: Vision> Shell<V> {
    fn new(context: &Arc<gpu::Context>, config: &Config) -> Self {
        let surface = context
            .create_xr_surface()
            .expect("unable to create XR surface");
        let format = surface.format();
        info!("XR swapchain format {format:?}");
        let data_dir = PathBuf::from(config.data_dir);
        let vision = V::new(context, format);
        let passthrough = create_passthrough(context, vision.passthrough_style(), &data_dir);
        let environment = context.create_xr_environment_depth();
        match environment {
            Some(ref environment) => {
                if let Err(e) = environment.start() {
                    warn!("environment depth would not start: {e}");
                }
                // Hands are not part of the scene.
                if let Err(e) = environment.set_hand_removal(true) {
                    warn!("environment depth hand removal: {e}");
                }
                info!("environment depth started");
            }
            None => warn!(
                "no environment depth; grant it with: adb shell pm grant {} \
                 com.oculus.permission.USE_SCENE",
                config.package
            ),
        }
        let (source, camera_model) = Self::open_camera(config.package);
        Self {
            vision,
            surface,
            passthrough,
            source,
            camera_model,
            camera_active: true,
            linear_target: is_srgb(format),
            environment,
            resample: Resample::new(context),
            resample_head: Pose::IDENTITY,
            environment_depth: None,
            dumped_depth: false,
            detector: spawn_detector(&data_dir, context),
            last_detect: Instant::now(),
            shown_generation: 0,
            tracks: Vec::new(),
            detections: 0,
            depth: spawn_depth(&data_dir, context),
            depth_map: None,
            depth_generation: 0,
            last_depth: Instant::now(),
            depths: 0,
            listener: listen::Listener::spawn(&data_dir.join("moonshine"), context),
            heard_generation: 0,
            rates: Rates::default(),
            reports: 0,
            start: Instant::now(),
            last_submit: None,
            frames: 0,
            update_ms: (0.0, 0.0),
            last_report: Instant::now(),
            data_dir,
            package: config.package,
        }
    }

    /// The left passthrough camera, or the test pattern if it will not
    /// open: a missing permission should look like "no camera", not like a
    /// crash or a black screen.
    fn open_camera(package: &str) -> (Box<dyn FrameSource>, Option<CameraModel>) {
        let size = (CAPTURE_SIZE[0] as i32, CAPTURE_SIZE[1] as i32);
        match camera::PassthroughCamera::new(Eye::Left, size) {
            Ok(camera) => {
                let model = camera_model(camera.calibration());
                (Box::new(camera), model)
            }
            Err(e) => {
                warn!(
                    "camera unavailable ({e}); showing the test pattern. Grant it with: \
                     adb shell pm grant {package} horizonos.permission.HEADSET_CAMERA"
                );
                (
                    Box::new(TestPattern::new(CAPTURE_SIZE[0], CAPTURE_SIZE[1])),
                    None,
                )
            }
        }
    }

    /// Sense only while focused. Android revokes the camera from background
    /// apps anyway, and a headset should not keep watching or listening for
    /// an app the wearer has left.
    fn set_active(&mut self, active: bool) {
        if active == self.camera_active {
            return;
        }
        self.camera_active = active;
        if let Some(ref listener) = self.listener {
            listener.set_listening(active);
        }
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
            (self.source, self.camera_model) = Self::open_camera(self.package);
        } else {
            info!("releasing the camera on focus loss");
            self.source = Box::new(TestPattern::new(CAPTURE_SIZE[0], CAPTURE_SIZE[1]));
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
            if !self.dumped_depth && self.data_dir.join("dump_depth").exists() {
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

        let camera_frame = self.source.next_frame();
        if let Some(camera_frame) = camera_frame {
            // Offer the frame to the models; a busy one drops it.
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
        if let Some(worker) = &self.detector
            && worker.generation() != self.shown_generation
            && let Some(observation) = worker.latest()
        {
            self.shown_generation = worker.generation();
            self.detections += 1;
            self.rates.detect_latency_ms = observation.latency_ms;
            self.tracks = observation.tracks;
            tracks_changed = true;
        }
        let mut depth_changed = false;
        if let Some(depth) = &self.depth
            && depth.generation() != self.depth_generation
            && let Some(map) = depth.latest()
        {
            self.depth_generation = depth.generation();
            self.depths += 1;
            self.depth_map = Some(map);
            depth_changed = true;
        }
        let (mut heard, mut speaking) = (None, false);
        if let Some(ref listener) = self.listener {
            speaking = listener.speaking();
            if listener.generation() != self.heard_generation {
                self.heard_generation = listener.generation();
                heard = listener.latest();
            }
        }

        let senses = Senses {
            time: self.start.elapsed().as_secs_f32(),
            passthrough: layered,
            frame: camera_frame,
            camera: self.camera_model.as_ref(),
            head,
            tracks: &self.tracks,
            tracks_changed,
            depth: self.depth_map.as_ref(),
            depth_changed,
            environment: self.environment_depth.as_ref(),
            environment_changed,
            heard,
            speaking,
            rates: self.rates,
        };
        let started = Instant::now();
        self.vision.update(context, encoder, &senses);
        let ms = started.elapsed().as_secs_f32() * 1000.0;
        self.update_ms = (self.update_ms.0 + ms, self.update_ms.1.max(ms));
        if let Some(style) = self.vision.restyle_passthrough()
            && let Some(ref passthrough) = self.passthrough
        {
            let linear = self.data_dir.join("lut_linear").exists();
            if let Err(e) = passthrough.set_style(&encode_style(style, linear)) {
                warn!("passthrough restyle: {e}");
            }
        }

        // Under passthrough, clear to nothing so the world shows through.
        let clear = if layered {
            gpu::TextureColor::TransparentBlack
        } else {
            gpu::TextureColor::OpaqueBlack
        };
        let extent = self.surface.extent();
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
                    projection::placement(camera, &eye, fov, self.vision.placement_distance())
                }
                None => eye_placement(fov),
            };
            self.vision.draw(
                &mut pass,
                &EyeView {
                    placement,
                    extent: [extent.width, extent.height],
                    linear_target: self.linear_target,
                    passthrough: layered,
                },
            );
        }

        match output {
            Output::Layered { frame, image } => {
                encoder.end_xr_image(&image);
                let sync = context.submit(encoder);
                self.vision.after_submit(&sync);
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
                self.vision.after_submit(&sync);
                self.last_submit = Some(sync);
            }
        }
        self.frames += 1;
        self.report();
    }

    /// Rates refresh every second for the HUD; the log gets every fifth.
    fn report(&mut self) {
        if self.last_report.elapsed() < Duration::from_secs(1) {
            return;
        }
        let secs = self.last_report.elapsed().as_secs_f64();
        self.rates.render_hz = (self.frames as f64 / secs) as f32;
        self.rates.detect_hz = (self.detections as f64 / secs) as f32;
        self.rates.depth_hz = (self.depths as f64 / secs) as f32;
        let frames = self.frames.max(1) as f32;
        let (update_total, update_worst) = std::mem::take(&mut self.update_ms);
        (self.frames, self.detections, self.depths) = (0, 0, 0);
        self.last_report = Instant::now();
        self.reports += 1;
        if !self.reports.is_multiple_of(5) {
            return;
        }
        // What the headset senses straight ahead, a sanity check.
        let ahead = self.environment_depth.as_ref().and_then(|(depth, head)| {
            let direction = projection::rotate(head.orientation, [0.0, 0.0, -1.0]);
            depth.along(0, head.position, direction)
        });
        let r = self.rates;
        info!(
            "render {:.1} Hz | detect {:.1} Hz, latency {:.1} ms | depth {:.1} Hz | \
             ahead {ahead:.2?} m | update {:.1} ms, worst {:.1} | {}",
            r.render_hz,
            r.detect_hz,
            r.detect_latency_ms,
            r.depth_hz,
            update_total / frames,
            update_worst,
            self.vision.status(),
        );
    }

    fn destroy(mut self, context: &gpu::Context) {
        // Join the workers first: they hold the context and may be
        // mid-submit.
        drop(self.detector.take());
        drop(self.depth.take());
        drop(self.listener.take());
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
        self.vision.destroy(context);
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

/// Run `V` until the activity is destroyed: the app's whole life. Call it
/// from the `#[ndk_glue::main]` function of the vision's crate.
pub fn run<V: Vision>() {
    let config = V::config();
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag(config.log_tag),
    );
    std::panic::set_hook(Box::new(|info| log::error!("panic: {info}")));
    info!("=== {} starting ===", config.name);

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
                application_name: config.name,
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

    // One context, shared by the renderer and Meganeura.
    let context = mar_core::init_context(Some(gpu::xr::Desc {
        instance: xr_instance.clone(),
        system_id: system,
    }))
    .expect("failed to initialize GPU context");

    let mut encoder = context.create_command_encoder(gpu::CommandEncoderDesc {
        name: config.log_tag,
        buffer_count: 2,
        manual_barriers: false,
    });

    let mut shell: Option<Shell<V>> = None;
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
                            if shell.is_none() {
                                shell = Some(Shell::new(&context, &config));
                            }
                            context.xr_session().unwrap().begin(VIEW_TYPE).unwrap();
                        }
                        xr::SessionState::STOPPING => {
                            context.xr_session().unwrap().end().unwrap();
                            if let Some(shell) = shell.take() {
                                shell.destroy(&context);
                            }
                        }
                        xr::SessionState::VISIBLE | xr::SessionState::SYNCHRONIZED => {
                            if let Some(shell) = shell.as_mut() {
                                shell.set_active(false);
                            }
                        }
                        xr::SessionState::FOCUSED => {
                            if let Some(shell) = shell.as_mut() {
                                shell.set_active(true);
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

        match &mut shell {
            Some(shell) => shell.render(&context, &mut encoder),
            // Not in session yet; idle rather than spin.
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }

    if let Some(shell) = shell.take() {
        shell.destroy(&context);
    }
    context.destroy_command_encoder(&mut encoder);
    info!("=== {} stopped ===", config.name);
}
