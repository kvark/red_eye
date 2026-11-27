//! RedEye: T-800 style augmented vision for Meta Quest 3.
//!
//! Passthrough is graded red, perception models run on the headset through
//! Meganeura, and a HUD built from the films' visual grammar is drawn over
//! the world with Blade. Everything is local; nothing leaves the device.
//!
//! # Layout
//!
//! * [`frame`] — the camera frame type (YUV420, as Camera2 delivers it).
//! * [`source`] — frame sources: test pattern, still image, and on Android
//!   the passthrough camera.
//! * [`grade`] — the red colour grade, shared by the shader and the
//!   passthrough colour map.
//! * [`render`] — Blade passes.
//! * [`perception`] — model outputs, tracking, and the observation the HUD
//!   reads.
//! * [`hud`] — the mechanics from `docs/concept.md`, as time-driven state.
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

use std::sync::Arc;

/// Create the GPU context that both rendering and inference share.
///
/// `xr` stays `None` for headless work (tests, previews, benchmarks); the
/// XR app fills it in and everything downstream is unchanged.
pub fn init_context(
    xr: Option<blade_graphics::XrDesc>,
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
