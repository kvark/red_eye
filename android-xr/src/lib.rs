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
use red_eye::perception::dfine::Dfine;
use red_eye::perception::track::Track;
use red_eye::perception::track::TrackerConfig;
use red_eye::perception::worker::Worker;
use red_eye::perception::yolox::{Config, Yolox};
use red_eye::render::{BoxInstance, BoxPass, GradePass, Placement};
use red_eye::source::{FrameSource, TestPattern};

const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const CAPTURE_SIZE: (i32, i32) = (1280, 960);

/// Detector models, pushed with `adb` rather than bundled while the model
/// choice is still moving. D-FINE (`tools/fetch_dfine.sh`, prepared at
/// [`DFINE_SIZE`]) is preferred when present: it is far more accurate, at
/// about six times YOLOX-Nano's compute. YOLOX-Nano
/// (`tools/fetch_yolox.sh`) is the light fallback.
const DFINE_PATH: &str = "/data/local/tmp/red_eye/dfine_nano.onnx";
const DFINE_SIZE: [u32; 2] = [640, 480];
const YOLOX_PATH: &str = "/data/local/tmp/red_eye/yolox_nano.safetensors";

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
    Placement {
        transform: [
            1.0,
            1.0,
            -(tan_r + tan_l) / span_x,
            -(tan_u + tan_d) / span_y,
        ],
    }
}

fn is_srgb(format: gpu::TextureFormat) -> bool {
    matches!(
        format,
        gpu::TextureFormat::Rgba8UnormSrgb | gpu::TextureFormat::Bgra8UnormSrgb
    )
}

struct App {
    surface: gpu::XrSurface,
    source: Box<dyn FrameSource>,
    camera_active: bool,
    grade_pass: GradePass,
    grade: Grade,
    linear_target: bool,
    box_pass: BoxPass,
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
        let source = Self::open_camera();
        Self {
            grade_pass: GradePass::new(
                context,
                format,
                [CAPTURE_SIZE.0 as u32, CAPTURE_SIZE.1 as u32],
            ),
            linear_target: is_srgb(format),
            box_pass: BoxPass::new(context, format),
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
            camera_active: true,
            grade: Grade::default(),
            start: Instant::now(),
            last_submit: None,
            frames: 0,
            last_report: Instant::now(),
        }
    }

    /// The left passthrough camera, or the test pattern if it will not
    /// open: a missing permission should look like "no camera", not like a
    /// crash or a black screen.
    fn open_camera() -> Box<dyn FrameSource> {
        match camera::PassthroughCamera::new(Eye::Left, CAPTURE_SIZE) {
            Ok(camera) => Box::new(camera),
            Err(e) => {
                warn!(
                    "camera unavailable ({e}); showing the test pattern. Grant it with: \
                     adb shell pm grant rust.red_eye_xr horizonos.permission.HEADSET_CAMERA"
                );
                Box::new(TestPattern::new(
                    CAPTURE_SIZE.0 as u32,
                    CAPTURE_SIZE.1 as u32,
                ))
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
        self.source = if active {
            Self::open_camera()
        } else {
            info!("releasing the camera on focus loss");
            Box::new(TestPattern::new(
                CAPTURE_SIZE.0 as u32,
                CAPTURE_SIZE.1 as u32,
            ))
        };
    }

    fn render(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        let Some(frame) = self.surface.acquire_frame(context) else {
            return;
        };
        if let Some(sync) = self.last_submit.take() {
            let _ = context.wait_for(&sync, !0);
        }

        encoder.start();
        encoder.init_texture(frame.texture());
        if let Some(camera_frame) = self.source.next_frame() {
            self.grade_pass.upload(context, encoder, camera_frame);
            // Offer the frame for detection; a busy worker drops it.
            if let Some(worker) = &self.detector
                && self.last_detect.elapsed() >= DETECT_INTERVAL
                && worker.submit(camera_frame)
            {
                self.last_detect = Instant::now();
            }
        }
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
        for eye in 0..frame.xr_view_count() {
            let view = frame.xr_view(eye);
            let fov = [
                view.fov.angle_left,
                view.fov.angle_right,
                view.fov.angle_up,
                view.fov.angle_down,
            ];
            let mut pass = encoder.render(
                "eye",
                gpu::RenderTargetSet {
                    colors: &[gpu::RenderTarget {
                        view: frame.xr_texture_view(eye),
                        init_op: gpu::InitOp::Clear(gpu::TextureColor::OpaqueBlack),
                        finish_op: gpu::FinishOp::Store,
                    }],
                    depth_stencil: None,
                },
            );
            let placement = eye_placement(fov);
            self.grade_pass
                .draw(&mut pass, &self.grade, placement, time, self.linear_target);
            let extent = self.surface.extent();
            self.box_pass.draw(
                &mut pass,
                placement,
                [extent.width, extent.height],
                BOX_THICKNESS,
            );
            // Placed like the camera image, so track rectangles line up.
            self.hud_canvas
                .composite(&mut pass, placement, self.linear_target, 1.0);
        }

        encoder.present(frame);
        let sync = context.submit(encoder);
        self.hud_canvas.after_submit(&sync);
        self.last_submit = Some(sync);
        self.frames += 1;

        // Rates refresh every second for the HUD; the log gets every fifth.
        if self.last_report.elapsed() >= Duration::from_secs(1) {
            let secs = self.last_report.elapsed().as_secs_f64();
            self.render_hz = (self.frames as f64 / secs) as f32;
            self.detect_hz = (self.detections as f64 / secs) as f32;
            self.reports += 1;
            if self.reports.is_multiple_of(5) {
                info!(
                    "render {:.1} Hz | detect {:.1} Hz, latency {:.1} ms | {:?}",
                    self.render_hz,
                    self.detect_hz,
                    self.detect_latency_ms,
                    self.hud.targeting.phase(),
                );
            }
            self.frames = 0;
            self.detections = 0;
            self.last_report = Instant::now();
        }
    }

    fn destroy(mut self, context: &gpu::Context) {
        // Join the worker first: it holds the context and may be mid-submit.
        drop(self.detector.take());
        if let Some(sync) = self.last_submit.take() {
            let _ = context.wait_for(&sync, !0);
        }
        self.grade_pass.destroy(context);
        self.box_pass.destroy(context);
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
    let context = red_eye::init_context(Some(gpu::XrDesc {
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
