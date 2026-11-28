//! MAR core: what every vision shares.
//!
//! MAR (movie augmented reality) puts visions from films over the world on
//! a Meta Quest: perception models run on the headset through Meganeura,
//! and overlays are drawn with Blade over the system's passthrough.
//! Everything is local; nothing leaves the device. Each vision is an app of
//! its own on `mar-shell`, the XR app they all run in; this crate is
//! everything they share, and builds and tests on a desktop.
//!
//! # Layout
//!
//! * [`frame`] — the camera frame type (YUV420, as Camera2 delivers it).
//! * [`source`] — frame sources: test pattern and still image; the
//!   passthrough camera is in `mar-shell`.
//! * [`grade`] — a colour grade of the camera's luminance, shared by the
//!   shader and the passthrough colour map.
//! * [`projection`] — the camera's model, and placing what it saw in an eye.
//! * [`render`] — Blade passes: the graded camera, boxes, depth contours.
//! * [`perception`] — detectors, tracking, depth (monocular and the
//!   headset's own), masks.
//! * [`voice`] — voice commands, from utterance to command.
//! * [`hud`] — building blocks for a HUD: the canvas, text, effects.
//!
//! # Sharing one GPU context
//!
//! The renderer and Meganeura run on a single `blade_graphics::Context`,
//! created by [`init_context`] and handed to Meganeura through
//! `SessionConfig::gpu`. That is why this crate pins the blade-graphics
//! revision Meganeura uses: two copies would make the `Arc<Context>` types
//! incompatible.

pub mod frame;
pub mod grade;
pub mod hud;
pub mod perception;
pub mod projection;
pub mod render;
pub mod source;
pub mod voice;

use std::sync::Arc;

/// Create the GPU context that both rendering and inference share.
///
/// `xr` stays `None` for headless work (tests, previews, benchmarks); the
/// XR app fills it in and everything downstream is unchanged.
pub fn init_context(
    xr: Option<blade_graphics::xr::Desc>,
) -> Result<Arc<blade_graphics::Context>, blade_graphics::NotSupportedError> {
    let context = unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            presentation: false,
            xr,
            // Validation layers are absent on a retail Quest, and cost time
            // exactly where there is none to spare.
            validation: cfg!(debug_assertions) && !cfg!(target_os = "android"),
            ..Default::default()
        })
    }?;
    let info = context.device_information();
    log::info!(
        "GPU: {} ({}), driver {}",
        info.device_name,
        info.driver_name,
        info.driver_info
    );
    Ok(Arc::new(context))
}
