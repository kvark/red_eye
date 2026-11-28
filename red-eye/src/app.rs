//! RedEye in the MAR shell: the red grade, the T-800 HUD, targeting by
//! voice, and objects outlined, ranged and placed at their depth.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use blade_graphics as gpu;
use log::warn;
use mar_core::grade::Grade;
use mar_core::hud::canvas::Canvas;
use mar_core::perception::dense::{self, Mask};
use mar_core::perception::environment;
use mar_core::perception::track::{self, Track};
use mar_core::perception::{Label, Rect};
use mar_core::render::{BoxInstance, BoxPass, GradePass, WireframePass, WireframeStyle};
use mar_core::voice::Command;
use mar_shell::{CAPTURE_SIZE, Config, EyeView, Senses, Vision};

use crate::hud::targeting::Phase;
use crate::hud::{Hud, Telemetry};

const DATA_DIR: &str = "/data/local/tmp/red_eye";

/// Where overlays are placed until the headset's depth ranges the outlined
/// object: the plane the camera image is put on, in metres along the
/// camera's axis. Things nearer or farther are off by the parallax between
/// camera and eye, a degree or two for the right eye at arm's length.
const PLACEMENT_DISTANCE: f32 = 1.5;

/// The placement plane's limits: the depth sensor's useful range.
const PLACEMENT_RANGE: std::ops::RangeInclusive<f32> = 0.3..=6.0;

/// Share of a new measurement in each smoothed distance: the depth is
/// noisy at object edges, and a plane that jumps shakes every overlay.
const RANGE_SMOOTHING: f32 = 0.3;

/// Resolution of the silhouette mask the wireframe outlines.
const MASK_SIZE: [u32; 2] = [320, 240];

/// Contour lines across the depth range of what is outlined.
const CONTOURS: f32 = 8.0;

/// Frames a track must be seen in before it is outlined.
const CONFIRM_HITS: u32 = 2;

/// Outline thickness in eye-buffer pixels.
const BOX_THICKNESS: f32 = 3.0;

/// The class to look for, by COCO name, beside voice commands:
///
/// ```text
/// adb shell "echo motorcycle > /data/local/tmp/red_eye/target"
/// adb shell rm /data/local/tmp/red_eye/target       # stand down
/// ```
///
/// With a target, the HUD brackets the match and the debug boxes go away.
const TARGET_FILE: &str = "target";
const TARGET_POLL: Duration = Duration::from_secs(1);

/// The target file's contents, when they differ from `last`: the file is
/// a fallback beside voice, so it only names a target when it changes, and
/// does not undo what was said since.
fn read_target(path: &Path, last: &mut Option<String>) -> Option<Option<Label>> {
    let name = std::fs::read_to_string(path).unwrap_or_default();
    if last.as_deref() == Some(name.as_str()) {
        return None;
    }
    let label = Label::from_name(&name);
    if label.is_none() && !name.trim().is_empty() {
        warn!("{}: {:?} is not a COCO class", path.display(), name.trim());
    }
    *last = Some(name);
    Some(label)
}

pub struct RedEye {
    grade: Grade,
    grade_pass: GradePass,
    box_pass: BoxPass,
    wireframe: WireframePass,
    /// The wireframe's style for the mask last uploaded; `None` until both
    /// depth and something to outline exist.
    wireframe_style: Option<WireframeStyle>,
    hud_canvas: Canvas,
    hud: Hud,
    /// Camera depth of the plane overlays are placed on, following the
    /// outlined object.
    placement_distance: f32,
    target_path: PathBuf,
    /// What the target file last said.
    target_file: Option<String>,
    last_target_poll: Instant,
    /// The HUD's clock, as of the last update.
    time: f32,
}

impl RedEye {
    /// The object the HUD is about: the matched target when there is one,
    /// else the one the wearer faces.
    fn outlined<'a>(&self, senses: &Senses<'a>) -> Option<&'a Track> {
        let tracks = senses.tracks;
        if let Phase::Match { track, .. } = self.hud.targeting.phase()
            && let Some(target) = tracks.iter().find(|t| t.id == track)
        {
            return Some(target);
        }
        // Without eye tracking, the wearer's focus is where the head points.
        let focus = senses
            .camera
            .and_then(|camera| camera.focus(self.placement_distance))
            .unwrap_or([0.5, 0.5]);
        track::focused(tracks, focus, CONFIRM_HITS)
    }

    /// Range every confirmed track through the headset's depth, and move
    /// the placement plane to the outlined object.
    fn update_ranges(&mut self, senses: &Senses) {
        let (Some((depth, head)), Some(camera)) = (senses.environment, senses.camera) else {
            return;
        };
        let mut ranges = HashMap::new();
        let mut placement = None;
        let outlined = self.outlined(senses).map(|t| t.id);
        for t in senses
            .tracks
            .iter()
            .filter(|t| t.is_confirmed(CONFIRM_HITS))
        {
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
    fn update_wireframe(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        senses: &Senses,
    ) {
        let Some(depth) = senses.depth else {
            return;
        };
        let boxes: Vec<(u8, Rect)> = self
            .outlined(senses)
            .map(|t| (1, t.rect))
            .into_iter()
            .collect();
        let mask: Mask = dense::silhouettes(&boxes, depth, MASK_SIZE[0], MASK_SIZE[1]);
        let Some((lo, hi)) = dense::masked_range(&mask, depth) else {
            self.wireframe_style = None;
            return;
        };
        self.wireframe.upload(context, encoder, &mask, depth);
        self.wireframe_style = Some(WireframeStyle {
            spacing: ((hi - lo) / CONTOURS).max(1e-3),
            // A slow drift, for the scanning look.
            phase: (senses.time * 0.25).fract(),
            highlight: 1,
            ..WireframeStyle::default()
        });
    }
}

impl Vision for RedEye {
    fn config() -> Config {
        Config {
            name: "RedEye",
            log_tag: "red_eye",
            package: "rust.red_eye_xr",
            data_dir: DATA_DIR,
        }
    }

    fn new(context: &Arc<gpu::Context>, format: gpu::TextureFormat) -> Self {
        Self {
            grade: Grade::default(),
            grade_pass: GradePass::new(context, format, CAPTURE_SIZE),
            box_pass: BoxPass::new(context, format),
            wireframe: WireframePass::new(context, format),
            wireframe_style: None,
            // The HUD canvas covers the camera image at its own resolution,
            // laid out at one pixel per point.
            hud_canvas: Canvas::new(context, CAPTURE_SIZE, 1.0, format),
            hud: Hud::default(),
            placement_distance: PLACEMENT_DISTANCE,
            target_path: Path::new(DATA_DIR).join(TARGET_FILE),
            target_file: None,
            last_target_poll: Instant::now() - TARGET_POLL,
            time: 0.0,
        }
    }

    /// The world graded red by the compositor, through the grade's
    /// mono-to-RGBA map.
    fn passthrough_style(&self) -> gpu::xr::PassthroughStyle {
        gpu::xr::PassthroughStyle {
            opacity: 1.0,
            edge_color: [0.0; 4],
            color_map: gpu::xr::PassthroughColorMap::MonoToRgba(Box::new(
                self.grade.mono_to_rgba_lut(),
            )),
        }
    }

    fn update(
        &mut self,
        context: &gpu::Context,
        encoder: &mut gpu::CommandEncoder,
        senses: &Senses,
    ) {
        let time = senses.time;
        self.time = time;
        // Without the compositor's passthrough, RedEye draws the camera.
        if !senses.passthrough
            && let Some(frame) = senses.frame
        {
            self.grade_pass.upload(context, encoder, frame);
        }
        if senses.tracks_changed {
            // Debug boxes only while scanning everything; a named target
            // gets the HUD's brackets instead.
            let boxes: Vec<_> = if self.hud.targeting.target().is_some() {
                Vec::new()
            } else {
                senses
                    .tracks
                    .iter()
                    .filter(|t| t.is_confirmed(CONFIRM_HITS))
                    .map(|t| {
                        BoxInstance::new(&t.rect, crate::hud::outline_color(t.label.category()))
                    })
                    .collect()
            };
            self.box_pass.set_boxes(context, &boxes);
        }
        if senses.tracks_changed || senses.environment_changed {
            self.update_ranges(senses);
        }
        if senses.tracks_changed || senses.depth_changed {
            self.update_wireframe(context, encoder, senses);
        }

        if self.last_target_poll.elapsed() >= TARGET_POLL {
            self.last_target_poll = Instant::now();
            if let Some(target) = read_target(&self.target_path, &mut self.target_file) {
                self.hud.set_target(target, time);
            }
        }
        self.hud.voice.speaking = senses.speaking;
        if let Some(ref heard) = senses.heard {
            if heard.accepted {
                match heard.recognition.command {
                    Command::Track(label) => self.hud.set_target(Some(label), time),
                    Command::Cancel => self.hud.set_target(None, time),
                }
            }
            self.hud.voice.heard = Some((heard.recognition.phrase.clone(), heard.accepted, time));
        }

        // Lay out the HUD once; each eye composites it.
        let rates = senses.rates;
        let telemetry = Telemetry {
            render_hz: rates.render_hz,
            detect_hz: rates.detect_hz,
            latency_ms: rates.detect_latency_ms,
        };
        let (hud, tracks) = (&mut self.hud, senses.tracks);
        self.hud_canvas.render(context, encoder, time as f64, |ui| {
            hud.draw(ui, time, tracks, &telemetry)
        });
    }

    fn placement_distance(&self) -> f32 {
        self.placement_distance
    }

    fn draw(&mut self, pass: &mut gpu::RenderCommandEncoder, eye: &EyeView) {
        if !eye.passthrough {
            self.grade_pass.draw(
                pass,
                &self.grade,
                eye.placement,
                self.time,
                eye.linear_target,
            );
        }
        self.box_pass
            .draw(pass, eye.placement, eye.extent, BOX_THICKNESS);
        if let Some(ref style) = self.wireframe_style {
            self.wireframe
                .draw(pass, eye.placement, style, eye.linear_target);
        }
        // Placed like the camera image, so track rectangles line up.
        self.hud_canvas
            .composite(pass, eye.placement, eye.linear_target, 1.0);
    }

    fn after_submit(&mut self, sync: &gpu::SyncPoint) {
        self.hud_canvas.after_submit(sync);
    }

    fn status(&self) -> String {
        format!(
            "plane {:.2} m, ranges {:.1?} | {:?}",
            self.placement_distance,
            self.hud.ranges.values().collect::<Vec<_>>(),
            self.hud.targeting.phase()
        )
    }

    fn destroy(self, context: &gpu::Context) {
        self.grade_pass.destroy(context);
        self.box_pass.destroy(context);
        self.wireframe.destroy(context);
        self.hud_canvas.destroy(context);
    }
}

#[ndk_glue::main]
pub fn main() {
    mar_shell::run::<RedEye>();
}
