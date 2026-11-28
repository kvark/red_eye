//! The red colour grade: camera luminance in, T-800 red out.
//!
//! The grade is defined once, as a tone curve over luminance, and consumed
//! in two places that must agree:
//!
//! * [`crate::render::GradePass`] runs it per pixel on camera frames we
//!   draw ourselves, adding the spatial effects a lookup table cannot
//!   express (grain, scanlines, vignette).
//! * [`Grade::mono_to_rgba_lut`] bakes the same curve into the 256-entry
//!   table that `XR_FB_passthrough` accepts as a
//!   `XrPassthroughColorMapMonoToRgbaFB` style. With that, the compositor
//!   tints the system passthrough itself, at display rate and with correct
//!   stereo, and our own layer only has to carry the HUD.
//!
//! The model follows the films: after levels and gamma, red carries the
//! luminance and green and blue are weak powers of it, so midtones are pure
//! red; past a clip point everything blends to a hot pink-white. The presets
//! are fitted to web stills (see `docs/concept.md`, "Colour grade"), so they
//! are starting points to tune on the headset, not measurements.
//!
//! [`Grade::map`] is the reference. The WGSL in `shaders/grade.wgsl` mirrors
//! it, and `tests/gpu.rs` checks that they agree.
//!
//! Colours are display-referred (sRGB-encoded), because that is the space a
//! look is designed in. Whether the passthrough colour map is interpreted as
//! linear or encoded is not documented; check on device.

/// Where [`Grade::exposure_gain`] puts a scene's median, as a share of the
/// way from the black point to the white: midtones in the middle.
pub const EXPOSURE_TARGET: f32 = 0.5;

/// The most a dim scene is brightened. Past about four times, the
/// passthrough cameras' noise in a dark room becomes the picture.
pub const MAX_EXPOSURE: f32 = 4.0;

/// Parameters of the grade.
#[derive(Clone, Debug, PartialEq)]
pub struct Grade {
    /// Input luminance mapped to black. Crushing shadows is most of the
    /// T1 look, and hides the passthrough cameras' sensor noise.
    pub black: f32,
    /// Input luminance mapped to full red.
    pub white: f32,
    /// Exponent after levels; above 1 darkens the midtones.
    pub gamma: f32,
    /// Output floor: T2 lifts its blacks, T1 does not.
    pub lift: f32,
    /// Green as `scale * y^power` of the graded luminance `y`.
    pub green: [f32; 2],
    /// Blue as `scale * y^power`.
    pub blue: [f32; 2],
    /// Graded luminance where highlights start blowing out to [`Grade::hot`].
    pub clip: f32,
    pub hot: [f32; 3],
    /// Film grain amplitude, added to luminance before grading. Per-pixel
    /// only, so absent from the lookup table.
    pub grain: f32,
    /// Corner darkening, `0..=1`. Per-pixel only.
    pub vignette: f32,
    /// How much darker the dark scanline phase is, `0..=1`. The films only
    /// show scanlines when the T-800 is damaged. Per-pixel only.
    pub scanline_depth: f32,
    /// Scanline period in output pixels. Per-pixel only.
    pub scanline_period: f32,
}

impl Default for Grade {
    fn default() -> Self {
        Self::t1()
    }
}

impl Grade {
    /// The Terminator (1984): crushed blacks, pure red midtones, heavy
    /// grain, highlights clipping to pink-white.
    pub fn t1() -> Self {
        Self {
            black: 0.12,
            white: 0.75,
            gamma: 1.3,
            lift: 0.0,
            green: [0.14, 1.0],
            blue: [0.10, 1.0],
            clip: 0.92,
            hot: [1.0, 0.80, 0.75],
            grain: 0.08,
            vignette: 0.25,
            scanline_depth: 0.0,
            scanline_period: 3.0,
        }
    }

    /// Terminator 2 (1991): lifted blacks, lower contrast, magenta-pink
    /// highlights.
    pub fn t2() -> Self {
        Self {
            black: 0.02,
            white: 0.92,
            gamma: 0.9,
            lift: 0.08,
            green: [0.25, 2.0],
            blue: [0.30, 2.2],
            clip: 0.95,
            hot: [1.0, 0.80, 0.75],
            grain: 0.03,
            vignette: 0.12,
            scanline_depth: 0.0,
            scanline_period: 3.0,
        }
    }

    /// The T1 look for the compositor's passthrough: the same red and hot
    /// highlights, but blacks barely crushed, shadows lifted by a gamma
    /// under one, and a faint red floor. The passthrough cameras see a dim
    /// room dimly, so T1's black point and dark midtones, which hide sensor
    /// noise in a lit scene, swallow a dark one whole.
    pub fn passthrough() -> Self {
        Self {
            black: 0.03,
            gamma: 0.9,
            lift: 0.05,
            ..Self::t1()
        }
    }

    /// A failing unit: the T2 grade with strong scanlines and grain.
    /// Temporal artefacts (jitter, flicker, channel split) belong to the
    /// damage mechanic, not to the grade.
    pub fn damaged() -> Self {
        Self {
            grain: 0.12,
            vignette: 0.4,
            scanline_depth: 0.45,
            scanline_period: 3.0,
            ..Self::t2()
        }
    }

    /// The grade with every per-pixel effect turned off: exactly the part
    /// a lookup table can carry.
    pub fn tone_only(&self) -> Self {
        Self {
            grain: 0.0,
            vignette: 0.0,
            scanline_depth: 0.0,
            ..self.clone()
        }
    }

    /// The grade for input made `gain` times brighter: the levels come
    /// down by the same factor. A dim room then fills the grade's range
    /// instead of sinking under its black point, with no change to the
    /// curve, so [`Grade::map`] stays the one definition.
    pub fn exposed(&self, gain: f32) -> Self {
        let gain = gain.max(1e-3);
        Self {
            black: self.black / gain,
            white: self.white / gain,
            ..self.clone()
        }
    }

    /// The gain a camera image of luminance `luma` (8-bit, any layout)
    /// wants: enough to bring its median to [`EXPOSURE_TARGET`] of the way
    /// up the grade's levels, never darkening and at most [`MAX_EXPOSURE`].
    /// The median, not the bright end: a lamp or a window in a dark room
    /// would hold the bright end up and leave the rest black. Highlights
    /// clip hot, as the films' do. Every seventh sample is plenty.
    pub fn exposure_gain(&self, luma: &[u8]) -> f32 {
        let mut histogram = [0u32; 256];
        let mut count = 0;
        for &l in luma.iter().step_by(7) {
            histogram[l as usize] += 1;
            count += 1;
        }
        if count == 0 {
            return 1.0;
        }
        let mut seen = 0;
        let median = histogram
            .iter()
            .position(|&n| {
                seen += n;
                seen * 2 >= count
            })
            .unwrap_or(255);
        let median = (median as f32 + 0.5) / 255.0;
        let middle = self.black + EXPOSURE_TARGET * (self.white - self.black);
        (middle / median).clamp(1.0, MAX_EXPOSURE)
    }

    /// Graded luminance, `0..=1`: levels, gamma, then lift.
    pub fn curve(&self, luma: f32) -> f32 {
        let range = (self.white - self.black).max(1e-4);
        let y = ((luma - self.black) / range)
            .clamp(0.0, 1.0)
            .powf(self.gamma);
        self.lift + (1.0 - self.lift) * y
    }

    /// The reference tone map: luminance in `0..=1` to a display colour.
    pub fn map(&self, luma: f32) -> [f32; 3] {
        let y = self.curve(luma);
        let base = [
            y,
            self.green[0] * y.powf(self.green[1]),
            self.blue[0] * y.powf(self.blue[1]),
        ];
        let s = smoothstep(self.clip, 1.0, y);
        std::array::from_fn(|i| base[i] + (self.hot[i] - base[i]) * s)
    }

    /// The tone map baked for `XrPassthroughColorMapMonoToRgbaFB`, whose
    /// `textureColorMap` has 256 entries indexed by passthrough luminance.
    pub fn mono_to_rgba_lut(&self) -> [[f32; 4]; 256] {
        std::array::from_fn(|i| {
            let [r, g, b] = self.map(i as f32 / 255.0);
            [r, g, b, 1.0]
        })
    }

    /// Pack into the uniform layout `shaders/grade.wgsl` expects.
    pub fn uniforms(&self, time: f32, linear_target: bool) -> GradeUniforms {
        GradeUniforms {
            levels: [self.black, self.white, self.gamma, self.lift],
            channels: [self.green[0], self.green[1], self.blue[0], self.blue[1]],
            hot: [self.hot[0], self.hot[1], self.hot[2], self.clip],
            effects: [
                self.grain,
                self.vignette,
                self.scanline_depth,
                self.scanline_period.max(1.0),
            ],
            frame: [time, if linear_target { 1.0 } else { 0.0 }, 0.0, 0.0],
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `GradeParams` in `shaders/grade.wgsl`. Every member is a `vec4` so the
/// Rust and WGSL layouts cannot drift apart through padding rules.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GradeUniforms {
    /// black, white, gamma, lift
    pub levels: [f32; 4],
    /// green scale, green power, blue scale, blue power
    pub channels: [f32; 4],
    /// hot colour, clip
    pub hot: [f32; 4],
    /// grain, vignette, scanline depth, scanline period
    pub effects: [f32; 4],
    /// time in seconds, 1 if the target encodes sRGB itself (so the shader
    /// must emit linear values)
    pub frame: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_keeps_a_dim_room_visible() {
        // What T1 maps to black, the passthrough grade shows as dark red.
        let dim = 0.08;
        assert_eq!(Grade::t1().map(dim)[0], 0.0);
        let shown = Grade::passthrough().map(dim);
        assert!(shown[0] > 0.1 && shown[0] > 2.0 * shown[1], "{shown:?}");
        // Nothing is pure black: the floor.
        assert!(Grade::passthrough().map(0.0)[0] > 0.0);
    }

    #[test]
    fn exposure_is_a_gain_on_the_input() {
        let grade = Grade::t1();
        let exposed = grade.exposed(3.0);
        for luma in [0.02f32, 0.05, 0.1, 0.2, 0.3] {
            let want = grade.map((luma * 3.0).min(1.0));
            let got = exposed.map(luma);
            for c in 0..3 {
                assert!(
                    (want[c] - got[c]).abs() < 1e-5,
                    "{luma}: {got:?} vs {want:?}"
                );
            }
        }
        // A dim pixel that was black is now visible.
        assert_eq!(grade.map(0.08)[0], 0.0);
        assert!(exposed.map(0.08)[0] > 0.1);
    }

    #[test]
    fn dim_scenes_are_brightened_and_bright_ones_left_alone() {
        let grade = Grade::t1();
        // A lit room: values spread to near white.
        let bright: Vec<u8> = (0..10_000).map(|i| (i % 250) as u8).collect();
        assert_eq!(grade.exposure_gain(&bright), 1.0);
        // A dark room: everything under 40 of 255.
        let dim: Vec<u8> = (0..10_000).map(|i| (i % 40) as u8).collect();
        let gain = grade.exposure_gain(&dim);
        assert!(gain > 3.0 && gain <= MAX_EXPOSURE, "{gain}");
        // Black: as much as allowed, no more.
        assert_eq!(grade.exposure_gain(&[0u8; 1000]), MAX_EXPOSURE);
        assert_eq!(grade.exposure_gain(&[]), 1.0);
    }

    fn luminance(c: [f32; 4]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    fn presets() -> [(&'static str, Grade); 3] {
        [
            ("t1", Grade::t1()),
            ("t2", Grade::t2()),
            ("damaged", Grade::damaged()),
        ]
    }

    #[test]
    fn ramp_is_monotonic_in_brightness() {
        for (name, grade) in presets() {
            let lut = grade.mono_to_rgba_lut();
            for (i, pair) in lut.windows(2).enumerate() {
                assert!(
                    luminance(pair[1]) + 1e-6 >= luminance(pair[0]),
                    "{name}: entry {i} -> {} gets darker",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn red_dominates_everywhere() {
        for (name, grade) in presets() {
            for (i, [r, g, b, _]) in grade.mono_to_rgba_lut().into_iter().enumerate() {
                assert!(r >= g && r >= b, "{name}: entry {i} is not red-dominant");
            }
        }
    }

    #[test]
    fn t1_crushes_and_t2_lifts() {
        assert_eq!(Grade::t1().map(0.1), [0.0; 3]);
        assert!(Grade::t2().map(0.0)[0] > 0.05);
        assert_eq!(Grade::t1().map(1.0), Grade::t1().hot);
    }

    #[test]
    fn tone_only_keeps_the_curve() {
        let grade = Grade::damaged();
        let tone = grade.tone_only();
        assert_eq!(tone.grain, 0.0);
        assert_eq!(tone.mono_to_rgba_lut(), grade.mono_to_rgba_lut());
    }

    #[test]
    fn uniforms_are_vec4_aligned() {
        assert_eq!(std::mem::size_of::<GradeUniforms>(), 5 * 16);
    }
}
