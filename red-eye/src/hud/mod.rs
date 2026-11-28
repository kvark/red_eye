//! RedEye's HUD: the films' visual grammar as time-driven state.
//!
//! Each [`Mechanic`] is one formalised behaviour from `docs/concept.md`.
//! The HUD owns per-target state (keyed by
//! [`mar_core::perception::track::TrackId`]) and per-mechanic timers, reads
//! the latest tracks, and lays itself out on a
//! [`mar_core::hud::canvas::Canvas`].
//!
//! Layout follows Terminator 2's grammar: labelled number blocks in the
//! corners, a left-aligned status stack mid-right, one large state word
//! with a block cursor bottom-centre. In a headset that frame is *body*
//! locked, lazily following the head, with text kept within about ±25° of
//! centre; a rigidly head-locked HUD is uncomfortable.

pub mod targeting;
pub mod view;

pub use view::{Hud, Telemetry, Voice};

use blade_graphics as gpu;
use mar_core::hud::canvas::Canvas;
use mar_core::perception::track::Track;
use mar_core::render::{Overlay, Placement};

/// The mechanics catalogued in `docs/concept.md`, in its order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mechanic {
    /// M1: the whole view mapped luminance to red. [`mar_core::grade`].
    ColorGrade,
    /// M2: code listings and labelled number blocks, always running.
    DataScroll,
    /// M3: reticle that follows a tracked person, with motion prediction.
    TargetLock,
    /// M4: snap-zoom freeze-frame inset with "TARGET ACQUIRED".
    AcquisitionInset,
    /// M5: body measurement and "MATCH" against the wearer.
    SizeAssessment,
    /// M6: traced outline and spec table for a vehicle or object.
    ObjectIdentification,
    /// M7: count people, trajectories, casualty readout.
    ThreatAssessment,
    /// M8: "POSSIBLE RESPONSE:" menu when spoken to.
    ResponseMenu,
    /// M9: text in view transcribed beneath itself.
    TextReading,
    /// M10: magnified inset with a landmark mesh of the face.
    ExpressionAnalysis,
    /// M11: compass rose, "IMAGE ENHANCE" mosaic resolve.
    NightVision,
    /// M12: model/version block and diagnostics on start.
    Boot,
    /// M13: scanlines, jitter, "REROUTE / ALTERNATE POWER".
    Damage,
    /// M14: speech level and pitch readouts.
    AudioAnalysis,
}

/// Outline colour per category for the box view: white for people, as T2
/// traces its subjects; the films' schematic cyan for vehicles; dimmer
/// tones for the rest, so people read first.
pub fn outline_color(category: mar_core::perception::Category) -> [f32; 4] {
    use mar_core::perception::Category;
    match category {
        Category::Person => [1.0, 1.0, 1.0, 1.0],
        Category::Vehicle => [0.45, 0.95, 1.0, 1.0],
        Category::Animal => [1.0, 0.85, 0.55, 0.9],
        Category::Object => [1.0, 0.6, 0.55, 0.7],
    }
}

/// One HUD frame as a [`mar_core::render::Overlay`]: lays out `hud` on the
/// canvas before the pass and composites it within.
pub struct HudOverlay<'a> {
    pub canvas: &'a mut Canvas,
    pub hud: &'a mut Hud,
    /// Seconds, the HUD's clock.
    pub now: f32,
    pub tracks: &'a [Track],
    pub telemetry: Telemetry,
}

impl Overlay for HudOverlay<'_> {
    fn prepare(&mut self, context: &gpu::Context, encoder: &mut gpu::CommandEncoder) {
        let (hud, now, tracks, telemetry) =
            (&mut *self.hud, self.now, self.tracks, &self.telemetry);
        self.canvas.render(context, encoder, now as f64, |ui| {
            hud.draw(ui, now, tracks, telemetry)
        });
    }

    fn draw(
        &self,
        pass: &mut gpu::RenderCommandEncoder,
        placement: Placement,
        linear_target: bool,
    ) {
        self.canvas.composite(pass, placement, linear_target, 1.0);
    }

    fn submitted(&mut self, sync_point: &gpu::SyncPoint) {
        self.canvas.after_submit(sync_point);
    }
}
