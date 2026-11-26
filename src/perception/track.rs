//! Turning per-frame detections into persistent targets.
//!
//! A target lock is only convincing if the box keeps its identity while it
//! moves, so detections are associated across frames by box overlap, the
//! way ByteTrack and SORT do it, minus the motion model for now. Track IDs
//! are what the HUD keys its per-target state on (lock animation, analysis
//! panel, assigned designation).

use super::{Detection, Label, Rect};

pub type TrackId = u32;

#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub id: TrackId,
    pub label: Label,
    /// Smoothed box.
    pub rect: Rect,
    pub score: f32,
    /// Consecutive frames this track was matched.
    pub hits: u32,
    /// Consecutive frames this track went unmatched.
    pub misses: u32,
    /// Total frames since the track was created.
    pub age: u32,
}

impl Track {
    /// Seen often enough to stop being a flicker. The HUD should only lock
    /// onto confirmed tracks.
    pub fn is_confirmed(&self, min_hits: u32) -> bool {
        self.hits >= min_hits
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
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            max_misses: 12,
            min_score: 0.4,
            smoothing: 0.6,
        }
    }
}

pub struct Tracker {
    pub config: TrackerConfig,
    tracks: Vec<Track>,
    next_id: TrackId,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Advance by one detector frame.
    pub fn update(&mut self, detections: &[Detection]) -> &[Track] {
        // Greedy association by descending overlap, within a label.
        let mut pairs = Vec::new();
        for (ti, track) in self.tracks.iter().enumerate() {
            for (di, det) in detections.iter().enumerate() {
                if det.label != track.label {
                    continue;
                }
                let iou = track.rect.iou(&det.rect);
                if iou >= self.config.iou_threshold {
                    pairs.push((iou, ti, di));
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
        }

        for (track, used) in self.tracks.iter_mut().zip(&track_used) {
            track.age += 1;
            if !used {
                track.hits = 0;
                track.misses += 1;
            }
        }
        let max_misses = self.config.max_misses;
        self.tracks.retain(|t| t.misses <= max_misses);

        for (det, used) in detections.iter().zip(det_used) {
            if used || det.score < self.config.min_score {
                continue;
            }
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
    fn labels_do_not_swap_identity() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.update(&[person(0.1, 0.9)]);
        let car = Detection {
            label: Label(2),
            ..person(0.1, 0.9)
        };
        let tracks = tracker.update(&[car]);
        assert_eq!(tracks.len(), 2, "a car in the same place is a new target");
    }

    #[test]
    fn weak_detections_do_not_spawn_tracks() {
        let mut tracker = Tracker::new(TrackerConfig::default());
        assert!(tracker.update(&[person(0.1, 0.1)]).is_empty());
    }
}
