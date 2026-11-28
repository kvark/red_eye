//! RedEye: T-800 style augmented vision, a MAR app.
//!
//! The world through the system's passthrough, graded red; objects
//! detected, tracked, ranged and outlined with depth contours; a HUD in
//! Terminator 2's visual grammar; targets named by voice. The XR app around
//! it is `mar-shell`; everything it perceives with is in `mar-core`.
//!
//! ```text
//! cargo apk run --manifest-path red-eye/Cargo.toml --release --no-logcat
//! tools/quest_setup.sh        # permission grants, and the device kept awake
//! adb logcat -v time | grep -E "red_eye|RustStdoutStderr"
//! ```
//!
//! Models are pushed to `/data/local/tmp/red_eye/` (see `AGENTS.md`); the
//! app runs without any of them, with less to show.

pub mod hud;

#[cfg(target_os = "android")]
mod app;
