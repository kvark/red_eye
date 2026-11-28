//! The HUD's look: Terminator 2's layout, drawn on the [`super::canvas`].
//!
//! - Labelled number blocks in the top corners: filler digits that churn,
//!   as in the films, interleaved with real telemetry.
//! - A status stack mid-right: scan mode, the named target, status, typed
//!   out line by line whenever it changes.
//! - One large state word bottom-centre with a block cursor: "MATCH" once
//!   the target is found, "SCANNING" while it is not.
//! - Corner brackets around the matched object, with its class name.
//! - With no target named, each confirmed track's class and confidence
//!   above its outline, which the renderer draws.
//!
//! The canvas covers exactly the camera image, so normalised track
//! rectangles map straight onto it. Positions and sizes are fractions of the
//! canvas, so the layout holds at any resolution.

use egui::{Color32, Pos2, Rect, Vec2, pos2, vec2};

use super::grid;
use super::targeting::{Phase, Targeting};
use super::text::Typewriter;
use crate::perception::track::{Track, TrackId};

/// Telemetry the number blocks report alongside their filler.
#[derive(Clone, Copy, Debug, Default)]
pub struct Telemetry {
    /// Display frames per second, as rendered.
    pub render_hz: f32,
    pub detect_hz: f32,
    pub latency_ms: f32,
}

const TEXT: Color32 = Color32::WHITE;
/// Filler and rules: present, but quieter than what matters.
const DIM: Color32 = Color32::from_rgba_premultiplied(190, 190, 190, 190);
/// A dark offset copy under everything: passthrough puts bright scenes
/// behind the HUD, where white text alone washes out.
const SHADOW: Color32 = Color32::from_rgba_premultiplied(0, 0, 0, 150);

/// The HUD's state across frames: targeting plus a glyph cache per text
/// size (a cache restarts whenever its font changes).
#[derive(Default)]
pub struct Hud {
    pub targeting: Targeting,
    /// Metres to each track that the headset's depth reaches, as the app
    /// measured them last; tracks without one show no range.
    pub ranges: std::collections::HashMap<TrackId, f32>,
    pub voice: Voice,
    small: grid::Glyphs,
    large: grid::Glyphs,
}

/// What the HUD shows of voice commands.
#[derive(Clone, Debug, Default)]
pub struct Voice {
    /// Someone is speaking now.
    pub speaking: bool,
    /// The last phrase heard, whether it was acted on, and when (HUD time).
    pub heard: Option<(String, bool, f32)>,
}

/// How long a heard phrase stays up.
const VOICE_SHOWN: f32 = 4.0;

/// Lays text into monospace cells through one grid batch.
struct Writer<'g> {
    builder: grid::Builder<'g>,
    cell: Vec2,
    canvas: Rect,
    shadow: Vec2,
}

impl<'g> Writer<'g> {
    fn new(ui: &egui::Ui, glyphs: &'g mut grid::Glyphs, size: f32) -> Self {
        let font = egui::FontId::monospace(size);
        let cell = ui.fonts_mut(|f| vec2(f.glyph_width(&font, 'M'), f.row_height(&font)));
        let canvas = ui.max_rect();
        let offset = (cell.y * 0.06).max(1.0);
        Self {
            builder: grid::Builder::new(ui, glyphs, font, canvas),
            cell,
            canvas,
            shadow: vec2(offset, offset),
        }
    }

    /// Write `text` with its first cell at `origin`; returns the end.
    fn write(&mut self, ui: &egui::Ui, origin: Pos2, text: &str, color: Color32) -> Pos2 {
        let mut buf = [0u8; 4];
        let mut x = origin.x;
        for c in text.chars() {
            if c != ' ' {
                let cell = Rect::from_min_size(pos2(x, origin.y), self.cell);
                let style = grid::Style::default();
                let glyph = c.encode_utf8(&mut buf);
                let shadow = cell.translate(self.shadow);
                self.builder
                    .text(ui, glyph, shadow, self.canvas, style, SHADOW);
                self.builder
                    .text(ui, glyph, cell, self.canvas, style, color);
            }
            x += self.cell.x;
        }
        pos2(x, origin.y)
    }

    /// The solid block cursor that ends the films' state lines.
    fn cursor(&mut self, at: Pos2, color: Color32) {
        let inset = vec2(self.cell.x * 0.1, self.cell.y * 0.2);
        let rect = Rect::from_min_size(at + inset, self.cell - 2.0 * inset);
        self.builder.fill(rect.translate(self.shadow), SHADOW);
        self.builder.fill(rect, color);
    }

    fn line(&mut self, from: Pos2, to: Pos2, width: f32, color: Color32) {
        let (sf, st) = (from + self.shadow, to + self.shadow);
        self.builder.line(sf, st, egui::Stroke::new(width, SHADOW));
        self.builder.line(from, to, egui::Stroke::new(width, color));
    }

    fn finish(self, ui: &egui::Ui) {
        if let Some(shape) = self.builder.finish() {
            ui.painter().add(shape);
        }
    }
}

/// Deterministic churn: a pseudo-random digit for a slot at a tick.
fn digit(row: u32, column: u32, tick: u32) -> char {
    let mut h = row.wrapping_mul(0x9E37_79B9)
        ^ column.wrapping_mul(0x85EB_CA6B)
        ^ tick.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    char::from(b'0' + (h % 10) as u8)
}

fn digits(row: u32, groups: u32, tick: u32) -> String {
    (0..groups)
        .map(|g| {
            (0..4)
                .map(|c| digit(row, g * 4 + c, tick))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

impl Hud {
    pub fn set_target(&mut self, target: Option<crate::perception::Label>, now: f32) {
        self.targeting.set_target(target, now);
    }

    /// Follow the latest tracks, then lay out the whole HUD into `ui`.
    pub fn draw(&mut self, ui: &mut egui::Ui, now: f32, tracks: &[Track], telemetry: &Telemetry) {
        let phase = self.targeting.update(tracks, now);
        let canvas = ui.max_rect();
        let (w, h) = (canvas.width(), canvas.height());
        let at = |x: f32, y: f32| canvas.min + vec2(x * w, y * h);
        let small = h * 0.024;
        let tick = (now * 8.0) as u32;

        let mut text = Writer::new(ui, &mut self.small, small);
        let row = text.cell.y;

        // Top left: analysis block, churning filler around real telemetry.
        let origin = at(0.04, 0.05);
        text.write(ui, origin, "ANALYSIS:", TEXT);
        let rule_y = origin.y + row * 1.15;
        text.line(
            pos2(origin.x, rule_y),
            pos2(origin.x + text.cell.x * 19.0, rule_y),
            1.0,
            DIM,
        );
        let lines = [
            format!(
                "FPS {:05.1} HZ  {}",
                telemetry.render_hz,
                digits(1, 1, tick)
            ),
            format!(
                "DET {:05.1} HZ  {}",
                telemetry.detect_hz,
                digits(2, 1, tick)
            ),
            format!(
                "LAT {:05.1} MS  {}",
                telemetry.latency_ms,
                digits(3, 1, tick)
            ),
            digits(4, 4, tick),
        ];
        for (i, l) in lines.iter().enumerate() {
            text.write(ui, origin + vec2(0.0, row * (1.4 + i as f32)), l, DIM);
        }

        // Top right: what is being tracked, by identity.
        let origin = at(0.70, 0.05);
        text.write(ui, origin, "TRACKING:", TEXT);
        let rule_y = origin.y + row * 1.15;
        text.line(
            pos2(origin.x, rule_y),
            pos2(origin.x + text.cell.x * 19.0, rule_y),
            1.0,
            DIM,
        );
        let mut listed: Vec<&Track> = tracks.iter().filter(|t| t.is_confirmed(2)).collect();
        listed.sort_by_key(|t| t.id);
        for i in 0..4 {
            let l = match listed.get(i) {
                Some(t) => format!("{:04} {}", t.id, t.label.name().to_uppercase()),
                None => digits(10 + i as u32, 2, tick / 2),
            };
            text.write(ui, origin + vec2(0.0, row * (1.4 + i as f32)), &l, DIM);
        }

        // Mid right: the status stack, typed whenever targeting changes.
        let since = match phase {
            Phase::Idle => 0.0,
            Phase::Searching { since } | Phase::Match { since, .. } => since,
        };
        let target = self.targeting.target();
        let matched = match phase {
            Phase::Match { track, .. } => tracks.iter().find(|t| t.id == track),
            _ => None,
        };
        let stack = [
            format!("SCAN MODE {}", digits(20, 1, 0)),
            format!(
                "TARGET: {}",
                target.map_or("NONE".to_owned(), |l| l.name().to_uppercase())
            ),
            match phase {
                Phase::Idle => "STATUS: STANDBY".to_owned(),
                Phase::Searching { .. } => "STATUS: SEARCHING".to_owned(),
                Phase::Match { .. } => "STATUS: LOCKED".to_owned(),
            },
            match matched {
                Some(t) => format!("PROBABILITY {:.4}", t.score),
                None => String::new(),
            },
        ];
        let origin = at(0.62, 0.40);
        let mut elapsed = now - since;
        for (i, l) in stack.iter().enumerate() {
            let typed = Typewriter::new(l.as_str());
            text.write(
                ui,
                origin + vec2(0.0, row * 1.2 * i as f32),
                typed.visible(elapsed),
                TEXT,
            );
            // Each line starts once the one above has finished.
            elapsed -= l.chars().count() as f32 / typed.chars_per_second;
        }

        // Under it, voice: a cue while someone speaks, then what was heard.
        let voice_at = origin + vec2(0.0, row * 1.2 * (stack.len() as f32 + 0.5));
        match self.voice.heard {
            _ if self.voice.speaking => {
                let dots = ".".repeat(1 + (now * 3.0) as usize % 3);
                text.write(ui, voice_at, &format!("VOICE INPUT{dots}"), DIM);
            }
            Some((ref phrase, accepted, at)) if now - at < VOICE_SHOWN => {
                let line = if accepted {
                    format!("VOICE INPUT: {}", phrase.to_uppercase())
                } else {
                    "VOICE INPUT: NO MATCH".to_owned()
                };
                let typed = Typewriter::new(&line);
                text.write(ui, voice_at, typed.visible(now - at), TEXT);
            }
            _ => {}
        }

        // Scanning everything: name each confirmed track over its outline.
        if target.is_none() {
            for t in &listed {
                let corner = at(t.rect.min[0], t.rect.min[1]);
                let mut label = format!("{} {:.2}", t.label.name().to_uppercase(), t.score);
                if let Some(range) = self.ranges.get(&t.id) {
                    label += &format!(" {range:.1}M");
                }
                text.write(ui, corner - vec2(0.0, row * 1.1), &label, TEXT);
            }
        }

        // Brackets and a name around the matched object.
        if let Some(t) = matched {
            let r = Rect::from_min_max(
                at(t.rect.min[0], t.rect.min[1]),
                at(t.rect.max[0], t.rect.max[1]),
            );
            let arm = (r.width().min(r.height()) * 0.25).max(6.0);
            let width = (h * 0.003).max(1.5);
            for (corner, dx, dy) in [
                (r.left_top(), 1.0, 1.0),
                (r.right_top(), -1.0, 1.0),
                (r.left_bottom(), 1.0, -1.0),
                (r.right_bottom(), -1.0, -1.0),
            ] {
                text.line(corner, corner + vec2(dx * arm, 0.0), width, TEXT);
                text.line(corner, corner + vec2(0.0, dy * arm), width, TEXT);
            }
            let label = t.label.name().to_uppercase();
            text.write(ui, r.left_top() - vec2(0.0, row * 1.2), &label, TEXT);
            if let Some(range) = self.ranges.get(&t.id) {
                let range = format!("RANGE {range:04.1} M");
                text.write(ui, r.left_bottom() + vec2(0.0, row * 0.3), &range, TEXT);
            }
        }
        text.finish(ui);

        // Bottom centre: the state word.
        let word = match phase {
            Phase::Idle => None,
            Phase::Searching { since } => Some(("SCANNING", since)),
            Phase::Match { since, .. } => Some(("MATCH", since)),
        };
        if let Some((word, since)) = word {
            let mut large = Writer::new(ui, &mut self.large, h * 0.075);
            let typed = Typewriter::new(word);
            let elapsed = now - since;
            // Centre the full word plus its cursor, so typing grows rightward.
            let width = large.cell.x * (word.chars().count() + 2) as f32;
            let origin = pos2(canvas.center().x - width * 0.5, canvas.min.y + h * 0.82);
            let end = large.write(ui, origin, typed.visible(elapsed), TEXT);
            if typed.cursor_on(elapsed) {
                large.cursor(end + vec2(large.cell.x * 0.5, 0.0), TEXT);
            }
            large.finish(ui);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn churn_is_deterministic_and_changes_with_time() {
        assert_eq!(digits(1, 4, 7), digits(1, 4, 7));
        assert_ne!(digits(1, 4, 7), digits(1, 4, 8));
        assert_eq!(digits(1, 2, 0).len(), 9);
        assert!(
            digits(3, 3, 5)
                .chars()
                .all(|c| c.is_ascii_digit() || c == ' ')
        );
    }
}
