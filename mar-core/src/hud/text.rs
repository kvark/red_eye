//! Time-based text effects: the HUD's text is never just *shown*, it is
//! typed out, scrolled, or blinked.
//!
//! Everything here is a pure function of elapsed time, so the renderer can
//! evaluate it at display rate and it can be tested without a GPU.

/// Text revealed one character at a time, with a block cursor that blinks
/// once typing is done.
#[derive(Clone, Debug)]
pub struct Typewriter {
    text: String,
    /// Reveal rate. The films type fast: a full analysis line lands in well
    /// under a second.
    pub chars_per_second: f32,
    /// Cursor blink period in seconds; zero hides the cursor entirely.
    pub blink_period: f32,
}

impl Typewriter {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            chars_per_second: 40.0,
            blink_period: 0.5,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Characters visible `elapsed` seconds after typing started.
    pub fn visible_chars(&self, elapsed: f32) -> usize {
        let total = self.text.chars().count();
        ((elapsed.max(0.0) * self.chars_per_second) as usize).min(total)
    }

    pub fn is_done(&self, elapsed: f32) -> bool {
        self.visible_chars(elapsed) == self.text.chars().count()
    }

    /// The visible prefix, cut on a character boundary.
    pub fn visible(&self, elapsed: f32) -> &str {
        let n = self.visible_chars(elapsed);
        match self.text.char_indices().nth(n) {
            Some((byte, _)) => &self.text[..byte],
            None => &self.text,
        }
    }

    /// Whether the cursor block is lit. Solid while typing, blinking after.
    pub fn cursor_on(&self, elapsed: f32) -> bool {
        if self.blink_period <= 0.0 {
            return false;
        }
        if !self.is_done(elapsed) {
            return true;
        }
        ((elapsed / (0.5 * self.blink_period)) as u64).is_multiple_of(2)
    }
}

/// Lines scrolling upwards through a fixed-height window, like the code
/// listings running down the side of the T-800's view.
#[derive(Clone, Debug)]
pub struct Scroll {
    lines: Vec<String>,
    pub lines_per_second: f32,
    /// Lines visible at once.
    pub window: usize,
}

impl Scroll {
    pub fn new(lines: Vec<String>, window: usize) -> Self {
        Self {
            lines,
            lines_per_second: 8.0,
            window,
        }
    }

    /// The visible window `elapsed` seconds in, oldest first, wrapping
    /// around the source so it never runs dry.
    pub fn visible(&self, elapsed: f32) -> impl Iterator<Item = &str> {
        let len = self.lines.len();
        let head = (elapsed.max(0.0) * self.lines_per_second) as usize;
        let count = if len == 0 { 0 } else { self.window };
        (0..count).map(move |i| self.lines[(head + i) % len].as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_reveals_and_finishes() {
        let tw = Typewriter::new("MATCH");
        assert_eq!(tw.visible(0.0), "");
        assert_eq!(tw.visible(2.0 / tw.chars_per_second + 1e-3), "MA");
        assert_eq!(tw.visible(10.0), "MATCH");
        assert!(tw.is_done(10.0));
    }

    #[test]
    fn typing_respects_char_boundaries() {
        let tw = Typewriter::new("ÄÖ");
        assert_eq!(tw.visible(1.0 / tw.chars_per_second + 1e-3), "Ä");
    }

    #[test]
    fn cursor_is_solid_while_typing_then_blinks() {
        let tw = Typewriter::new("SCAN MODE 43984");
        assert!(tw.cursor_on(0.05));
        let done = 1.0;
        let half = 0.5 * tw.blink_period;
        let a = tw.cursor_on(done);
        let b = tw.cursor_on(done + half);
        assert_ne!(a, b);
    }

    #[test]
    fn scroll_wraps() {
        let scroll = Scroll::new(vec!["A".into(), "B".into(), "C".into()], 2);
        let at = |t: f32| scroll.visible(t).collect::<Vec<_>>();
        assert_eq!(at(0.0), ["A", "B"]);
        assert_eq!(at(2.0 / scroll.lines_per_second + 1e-3), ["C", "A"]);
    }
}
