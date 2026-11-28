//! What the T-800 is looking for, and whether it has found it.
//!
//! The wearer names a class ("track the motorcycle"); every confirmed track
//! of that class is a candidate, and one becomes the match. The match
//! sticks to its track while the track lives, so the lock does not hop
//! between two motorcycles from frame to frame.

use mar_core::perception::Label;
use mar_core::perception::track::{Track, TrackId};

/// Where targeting is, with the time (seconds) it got there, which drives
/// the HUD's typing animations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// No target named.
    Idle,
    /// A target is named but nothing matching is in view.
    Searching { since: f32 },
    /// A confirmed track of the target class.
    Match { track: TrackId, since: f32 },
}

#[derive(Clone, Debug)]
pub struct Targeting {
    target: Option<Label>,
    /// Frames a track must be seen in before it can match.
    pub confirm_hits: u32,
    phase: Phase,
}

impl Default for Targeting {
    fn default() -> Self {
        Self {
            target: None,
            confirm_hits: 2,
            phase: Phase::Idle,
        }
    }
}

impl Targeting {
    pub fn target(&self) -> Option<Label> {
        self.target
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Name a new target, or clear it with `None`. Naming the current
    /// target again changes nothing.
    pub fn set_target(&mut self, target: Option<Label>, now: f32) {
        if target == self.target {
            return;
        }
        self.target = target;
        self.phase = match target {
            Some(_) => Phase::Searching { since: now },
            None => Phase::Idle,
        };
    }

    /// Follow the latest tracks.
    pub fn update(&mut self, tracks: &[Track], now: f32) -> Phase {
        let Some(target) = self.target else {
            self.phase = Phase::Idle;
            return self.phase;
        };
        let candidates = || {
            tracks
                .iter()
                .filter(|t| t.label == target && t.is_confirmed(self.confirm_hits))
        };
        self.phase = match self.phase {
            // Keep the matched track while it lives, even through a few
            // unconfirmed frames: the tracker already drops it once lost.
            Phase::Match { track, since } if tracks.iter().any(|t| t.id == track) => {
                Phase::Match { track, since }
            }
            _ => match candidates().max_by(|a, b| a.rect.area().total_cmp(&b.rect.area())) {
                // The largest candidate: the nearest, most likely.
                Some(best) => Phase::Match {
                    track: best.id,
                    since: now,
                },
                None => match self.phase {
                    Phase::Searching { since } => Phase::Searching { since },
                    _ => Phase::Searching { since: now },
                },
            },
        };
        self.phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mar_core::perception::Rect;

    fn track(id: TrackId, label: u16, hits: u32, size: f32) -> Track {
        Track {
            id,
            label: Label(label),
            rect: Rect::new(0.1, 0.1, 0.1 + size, 0.1 + size),
            score: 0.9,
            hits,
            misses: 0,
            age: hits,
        }
    }

    const MOTORCYCLE: Label = Label(3);

    #[test]
    fn idle_without_a_target() {
        let mut t = Targeting::default();
        assert_eq!(t.update(&[track(1, 3, 5, 0.2)], 0.0), Phase::Idle);
    }

    #[test]
    fn searches_then_matches_the_largest_confirmed_candidate() {
        let mut t = Targeting::default();
        t.set_target(Some(MOTORCYCLE), 1.0);
        assert_eq!(t.update(&[], 1.5), Phase::Searching { since: 1.0 });
        // Unconfirmed, and wrong class: still searching.
        let tracks = [track(1, 3, 1, 0.3), track(2, 0, 9, 0.5)];
        assert_eq!(t.update(&tracks, 2.0), Phase::Searching { since: 1.0 });
        let tracks = [track(1, 3, 3, 0.1), track(3, 3, 3, 0.3)];
        assert_eq!(
            t.update(&tracks, 2.5),
            Phase::Match {
                track: 3,
                since: 2.5
            }
        );
    }

    #[test]
    fn the_match_sticks_to_its_track() {
        let mut t = Targeting::default();
        t.set_target(Some(MOTORCYCLE), 0.0);
        t.update(&[track(1, 3, 3, 0.1)], 1.0);
        // A bigger motorcycle appears: the lock stays.
        let tracks = [track(1, 3, 4, 0.1), track(2, 3, 3, 0.4)];
        assert_eq!(
            t.update(&tracks, 2.0),
            Phase::Match {
                track: 1,
                since: 1.0
            }
        );
        // The matched track is gone: the next candidate takes over.
        assert_eq!(
            t.update(&[track(2, 3, 4, 0.4)], 3.0),
            Phase::Match {
                track: 2,
                since: 3.0
            }
        );
        assert_eq!(t.update(&[], 4.0), Phase::Searching { since: 4.0 });
    }

    #[test]
    fn retargeting_restarts_the_search_and_clearing_idles() {
        let mut t = Targeting::default();
        t.set_target(Some(MOTORCYCLE), 0.0);
        t.update(&[track(1, 3, 3, 0.1)], 1.0);
        t.set_target(Some(MOTORCYCLE), 2.0);
        assert!(
            matches!(t.phase(), Phase::Match { .. }),
            "same target is a no-op"
        );
        t.set_target(Some(Label(0)), 2.0);
        assert_eq!(t.phase(), Phase::Searching { since: 2.0 });
        t.set_target(None, 3.0);
        assert_eq!(t.phase(), Phase::Idle);
    }
}
