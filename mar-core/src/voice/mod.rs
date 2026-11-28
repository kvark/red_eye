//! Voice commands: "track the motorcycle", "cancel".
//!
//! The microphone stream is cut into utterances by [`vad`], each is heard
//! by Moonshine-tiny ([`moonshine`]), and decoding is constrained to the
//! command [`grammar`], whose confidence rejects speech that is no command.
//! Audio is processed as it arrives and never stored.

pub mod grammar;
pub mod moonshine;
pub mod vad;

pub use grammar::{Command, Grammar, Recognition};
pub use moonshine::Moonshine;
pub use vad::{Vad, VadConfig};

/// Least confidence (mean log-probability per token) for a command to be
/// acted on. Commands spoken clearly score near 0; other speech, forced
/// into the grammar, scores far lower.
pub const MIN_CONFIDENCE: f32 = -1.5;
