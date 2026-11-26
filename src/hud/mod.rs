//! The HUD: the films' visual grammar as time-driven state.
//!
//! Each [`Mechanic`] is one formalised behaviour from `docs/concept.md`.
//! The HUD owns per-target state (keyed by
//! [`crate::perception::track::TrackId`]) and per-mechanic timers, reads the
//! latest [`crate::perception::Observation`], and emits draw primitives.
//! Drawing stays in [`crate::render`].
//!
//! Layout follows Terminator 2's grammar: labelled number blocks in the
//! corners, a left-aligned status stack mid-right, one large state word
//! with a block cursor bottom-centre. In a headset that frame is *body*
//! locked, lazily following the head, with text kept within about ±25° of
//! centre; a rigidly head-locked HUD is uncomfortable.

pub mod text;

/// The mechanics catalogued in `docs/concept.md`, in its order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mechanic {
    /// M1: the whole view mapped luminance to red. [`crate::grade`].
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
pub fn outline_color(category: crate::perception::Category) -> [f32; 4] {
    use crate::perception::Category;
    match category {
        Category::Person => [1.0, 1.0, 1.0, 1.0],
        Category::Vehicle => [0.45, 0.95, 1.0, 1.0],
        Category::Animal => [1.0, 0.85, 0.55, 0.9],
        Category::Object => [1.0, 0.6, 0.55, 0.7],
    }
}
