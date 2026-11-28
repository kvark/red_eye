//! Turning per-frame detections into persistent targets.
//!
//! A target lock is only convincing if the box keeps its identity while it
//! moves, so detections are associated across frames by box overlap, the
//! way ByteTrack and SORT do it, minus the motion model for now. Track IDs
//! are what the HUD keys its per-target state on (lock animation, analysis
//! panel, assigned designation).
//!
//! Detectors waver, and a HUD that follows every wobble flickers. Three
//! things hold still here: a track keeps its identity when the detector
//! changes its mind about the class (the label follows a decaying vote);
//! a confirmed track coasts through a few missed frames; and
//! [`FocusLock`] keeps attention on one object until another has held the
//! wearer's focus for a while.

use std::collections::HashMap;

use super::{Detection, Label, Rect};

pub type TrackId = u32;

#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub id: TrackId,
    pub label: Label,
    /// Smoothed box.
    pub rect: Rect,
    pub score: f32,
    /// Frames this track was matched in.
    pub hits: u32,
    /// Consecutive frames this track went unmatched.
    pub misses: u32,
    /// Total frames since the track was created.
    pub age: u32,
}

/// The confirmed track the wearer is looking at: the one whose box holds
/// `focus` (normalised image coordinates), the smallest if several do, or
/// else the one whose box comes nearest it.
pub fn focused(tracks: &[Track], focus: [f32; 2], min_hits: u32) -> Option<&Track> {
    let distance = |t: &Track| {
        let dx = (t.rect.min[0] - focus[0])
            .max(focus[0] - t.rect.max[0])
            .max(0.0);
        let dy = (t.rect.min[1] - focus[1])
            .max(focus[1] - t.rect.max[1])
            .max(0.0);
        (dx * dx + dy * dy).sqrt()
    };
    let area = |t: &Track| (t.rect.max[0] - t.rect.min[0]) * (t.rect.max[1] - t.rect.min[1]);
    tracks
        .iter()
        .filter(|t| t.is_confirmed(min_hits))
        .min_by(|a, b| {
            distance(a)
                .total_cmp(&distance(b))
                .then(area(a).total_cmp(&area(b)))
        })
}

/// Missed frames a confirmed track coasts through: about 0.3 s of
/// detection, longer than the detector's usual dropouts and short enough
/// that a box does not outstay its object by much.
pub const COAST_MISSES: u32 = 3;

impl Track {
    /// Seen often enough to stop being a flicker, and not lost for longer
    /// than a dropout. The HUD should only lock onto confirmed tracks.
    pub fn is_confirmed(&self, min_hits: u32) -> bool {
        self.hits >= min_hits && self.misses <= COAST_MISSES
    }
}

/// Which object the wearer attends to, held steady: the focused track
/// ([`focused`]) takes over only after it has been the focused one for
/// [`FocusLock::SWITCH_UPDATES`] tracker updates in a row, or at once when
/// the current one is lost or the focus has left its box.
#[derive(Clone, Debug, Default)]
pub struct FocusLock {
    current: Option<TrackId>,
    challenger: Option<(TrackId, u32)>,
}

impl FocusLock {
    /// About 0.3 s at the detector's rate.
    pub const SWITCH_UPDATES: u32 = 3;

    pub fn current(&self) -> Option<TrackId> {
        self.current
    }

    /// Follow the latest `tracks`, with the wearer's `focus` in normalised
    /// image coordinates.
    pub fn update(&mut self, tracks: &[Track], focus: [f32; 2], min_hits: u32) -> Option<TrackId> {
        let best = focused(tracks, focus, min_hits).map(|t| t.id);
        let current = self.current.and_then(|id| {
            tracks
                .iter()
                .find(|t| t.id == id && t.is_confirmed(min_hits))
        });
        let Some(current) = current else {
            self.current = best;
            self.challenger = None;
            return self.current;
        };
        match best {
            Some(best) if best != current.id => {
                let wins = match self.challenger {
                    Some((id, n)) if id == best => n + 1,
                    _ => 1,
                };
                // Focus leaving the current box altogether is decisive.
                if wins >= Self::SWITCH_UPDATES || !current.rect.contains(focus) {
                    self.current = Some(best);
                    self.challenger = None;
                } else {
                    self.challenger = Some((best, wins));
                }
            }
            _ => self.challenger = None,
        }
        self.current
    }
}

#[derive(Clone, Debug)]
pub struct TrackerConfig {
    /// Minimum overlap to associate a detection with a track.
    pub iou_threshold: f32,
    /// Unmatched frames before a track is dropped.
    pub max_misses: u32,
    /// Detections below this score never start a track.
    pub min_score: f32,
    /// How far each matched detection moves the smoothed box, `0..=1`.
    pub smoothing: f32,
    /// Minimum overlap to associate a detection of another class: the
    /// detector changing its mind about the same object, rather than a
    /// different one beside it.
    pub relabel_iou: f32,
    /// How much of a track's class vote each frame keeps, `0..1`.
    pub vote_decay: f32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            max_misses: 12,
            min_score: 0.4,
            smoothing: 0.6,
            relabel_iou: 0.6,
            vote_decay: 0.8,
        }
    }
}

pub struct Tracker {
    pub config: TrackerConfig,
    tracks: Vec<Track>,
    next_id: TrackId,
    /// Each track's decayed detection scores per class; its label is the
    /// leader.
    votes: HashMap<TrackId, Vec<(Label, f32)>>,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
            votes: HashMap::new(),
        }
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Advance by one detector frame.
    pub fn update(&mut self, detections: &[Detection]) -> &[Track] {
        // Greedy association by descending overlap. Another class needs a
        // closer match, and loses ties to the track's own class.
        let mut pairs = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in detections.iter().enumerate() {
                let iou = track.rect.iou(&det.rect);
                let same = det.label == track.label;
                let needed = if same {
                    self.config.iou_threshold
                } else {
                    self.config.relabel_iou
                };
                if iou >= needed {
                    pairs.push((iou + if same { 0.1 } else { 0.0 }, ti, di));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut track_used = vec![false; self.tracks.len()];
        let mut det_used = vec![false; detections.len()];
        for (_, ti, di) in pairs {
            if track_used[ti] || det_used[di] {
                continue;
            }
            track_used[ti] = true;
            det_used[di] = true;
            let det = &detections[di];
            let track = &mut self.tracks[ti];
            track.rect = track.rect.lerp(&det.rect, self.config.smoothing);
            track.score = det.score;
            track.hits += 1;
            track.misses = 0;
            let votes = self.votes.entry(track.id).or_default();
            for vote in votes.iter_mut() {
                vote.1 *= self.config.vote_decay;
            }
            match votes.iter_mut().find(|v| v.0 == det.label) {
                Some(vote) => vote.1 += det.score,
                None => votes.push((det.label, det.score)),
            }
            if let Some(&(leader, _)) = votes.iter().max_by(|a, b| a.1.total_cmp(&b.1)) {
                track.label = leader;
            }
        }

        for (track, used) in self.tracks.iter_mut().zip(&track_used) {
            track.age += 1;
            if !used {
                track.misses += 1;
            }
        }
        let max_misses = self.config.max_misses;
        let votes = &mut self.votes;
        self.tracks.retain(|t| {
            let keep = t.misses <= max_misses;
            if !keep {
                votes.remove(&t.id);
            }
            keep
        });

        for (det, used) in detections.iter().zip(det_used) {
            if used || det.score < self.config.min_score {
                continue;
            }
            self.votes
                .insert(self.next_id, vec![(det.label, det.score)]);
            self.tracks.push(Track {
                id: self.next_id,
                label: det.label,
                rect: det.rect,
                score: det.score,
                hits: 1,
                misses: 0,
                age: 1,
            });
            self.next_id += 1;
        }
        &self.tracks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_picks_the_box_it_falls_in_then_the_nearest() {
        let track = |id: TrackId, rect: Rect| Track {
            id,
            label: Label::PERSON,
            rect,
            score: 0.9,
            hits: 3,
            misses: 0,
            age: 3,
        };
        let tracks = [
            track(1, Rect::new(0.0, 0.0, 1.0, 1.0)),
            track(2, Rect::new(0.4, 0.4, 0.6, 0.6)),
            track(3, Rect::new(0.8, 0.0, 0.9, 0.1)),
        ];
        // Inside both 1 and 2: the smaller.
        assert_eq!(focused(&tracks, [0.5, 0.5], 2).unwrap().id, 2);
        // Inside only the big one.
        assert_eq!(focused(&tracks[..1], [0.1, 0.9], 2).unwrap().id, 1);
        // Outside all of 2 and 3: the nearer.
        assert_eq!(focused(&tracks[1..], [0.75, 0.05], 2).unwrap().id, 3);
        assert!(focused(&[], [0.5, 0.5], 2).is_none());
    }

    fn person(x: f32, score: f32) -> Detection {
        Detection {
            rect: Rect::new(x, 0.2, x + 0.2, 0.9),
            label: Label::PERSON,
            score,
        }
    }

    #[test]
    fn identity_survives_motion() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        let id = tracker.update(&[person(0.1, 0.9)])[0].id;
        for step in 1..20 {
            let tracks = tracker.update(&[person(0.1 + step as f32 * 0.01, 0.9)]);
            assert_eq!(tracks.len(), 1);
            assert_eq!(tracks[0].id, id);
        }
        assert!(tracker.tracks()[0].is_confirmed(10));
    }

    #[test]
    fn lost_tracks_expire_and_new_ones_get_fresh_ids() {
        let config = TrackerConfig {
            max_misses: 2,
            ..Default::default()
        };
        let mut tracker = Tracker::new(config);
        let first = tracker.update(&[person(0.1, 0.9)])[0].id;
        for _ in 0..3 {
            tracker.update(&[]);
        }
        assert!(tracker.tracks().is_empty());
        let second = tracker.update(&[person(0.1, 0.9)])[0].id;
        assert_ne!(first, second);
    }

    #[test]
    fn a_wavering_class_keeps_the_track_and_follows_the_vote() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        let id = tracker.update(&[person(0.1, 0.9)])[0].id;
        let dog = |score| Detection {
            label: Label(16),
            ..person(0.1, score)
        };
        // One frame of "dog": the same object, still a person by the vote.
        let tracks = tracker.update(&[dog(0.6)]);
        assert_eq!((tracks.len(), tracks[0].id), (1, id));
        assert_eq!(tracks[0].label, Label::PERSON);
        // Dog, again and again: the vote turns.
        for _ in 0..4 {
            tracker.update(&[dog(0.8)]);
        }
        let tracks = tracker.tracks();
        assert_eq!((tracks.len(), tracks[0].id), (1, id));
        assert_eq!(tracks[0].label, Label(16));
    }

    #[test]
    fn another_class_beside_a_track_is_a_new_target() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.update(&[person(0.1, 0.9)]);
        // Overlapping, but not the same box: a car next to the person.
        let car = Detection {
            label: Label(2),
            ..person(0.2, 0.9)
        };
        let tracks = tracker.update(&[car]);
        assert_eq!(tracks.len(), 2);
    }

    #[test]
    fn confirmed_tracks_coast_through_dropouts() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        for _ in 0..3 {
            tracker.update(&[person(0.1, 0.9)]);
        }
        for _ in 0..COAST_MISSES {
            tracker.update(&[]);
            assert!(
                tracker.tracks()[0].is_confirmed(2),
                "a dropout blinked it off"
            );
        }
        tracker.update(&[]);
        assert!(
            !tracker.tracks()[0].is_confirmed(2),
            "lost, but still shown"
        );
    }

    #[test]
    fn focus_lock_holds_until_a_challenger_persists() {
        let track = |id: TrackId, x: f32, width: f32| Track {
            id,
            label: Label::PERSON,
            rect: Rect::new(x, 0.2, x + width, 0.8),
            score: 0.9,
            hits: 3,
            misses: 0,
            age: 3,
        };
        // Two overlapping boxes; the focus is in both, and the smaller one,
        // 2, fits it better.
        let tracks = [track(1, 0.2, 0.3), track(2, 0.3, 0.2)];
        let mut lock = FocusLock::default();
        assert_eq!(lock.update(&tracks[..1], [0.45, 0.5], 2), Some(1));
        // Track 2 appears and is the better fit, but must persist.
        for _ in 1..FocusLock::SWITCH_UPDATES {
            assert_eq!(lock.update(&tracks, [0.45, 0.5], 2), Some(1));
        }
        assert_eq!(lock.update(&tracks, [0.45, 0.5], 2), Some(2));
        // The focus leaving the current box switches at once.
        let far = [track(2, 0.3, 0.2), track(3, 0.0, 0.1)];
        assert_eq!(lock.update(&far, [0.05, 0.5], 2), Some(3));
        // Losing the current one hands over at once too.
        assert_eq!(lock.update(&far[..1], [0.05, 0.5], 2), Some(2));
    }

    #[test]
    fn weak_detections_do_not_spawn_tracks() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        assert!(tracker.update(&[person(0.1, 0.1)]).is_empty());
    }
}
