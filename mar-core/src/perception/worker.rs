//! The perception worker: a thread that owns the detector and the tracker
//! and publishes [`Observation`]s.
//!
//! The renderer runs at display rate and never waits for it: it offers
//! frames with [`Worker::submit`], which drops them while the worker is
//! busy (a queued frame would be stale by the time it ran), and reads
//! [`Worker::latest`] whenever it likes.
//!
//! The detector is built *on* the worker thread, by the closure given to
//! [`Worker::spawn`]: a Meganeura session owns command encoders and is not
//! `Sync`, while the `Arc<blade_graphics::Context>` it needs is freely
//! shareable.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use super::track::{Tracker, TrackerConfig};
use super::{Detector, Observation};
use crate::frame::Frame;

enum Command {
    Detect(Frame),
    Stop,
}

pub struct Worker {
    tx: mpsc::Sender<Command>,
    latest: Arc<Mutex<Option<Observation>>>,
    generation: Arc<AtomicU64>,
    ready: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Worker {
    /// Start the worker. `make` runs on the new thread; if it fails the
    /// worker logs why and never becomes ready, so frames are dropped and
    /// the app keeps running without perception.
    pub fn spawn<F>(make: F, tracker: TrackerConfig) -> Self
    where
        F: FnOnce() -> Result<Box<dyn Detector>, String> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let latest = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU64::new(0));
        let ready = Arc::new(AtomicBool::new(false));
        let handle = {
            let (latest, generation, ready) = (
                Arc::clone(&latest),
                Arc::clone(&generation),
                Arc::clone(&ready),
            );
            std::thread::Builder::new()
                .name("perception".into())
                .spawn(move || {
                    let mut detector = match make() {
                        Ok(detector) => detector,
                        Err(e) => {
                            log::error!("perception disabled: {e}");
                            return;
                        }
                    };
                    let mut tracker = Tracker::new(tracker);
                    ready.store(true, Ordering::Release);
                    while let Ok(Command::Detect(frame)) = rx.recv() {
                        let start = Instant::now();
                        let detections = detector.detect(&frame);
                        let tracks = tracker.update(&detections).to_vec();
                        let observation = Observation {
                            timestamp_ns: frame.timestamp_ns,
                            tracks,
                            latency_ms: start.elapsed().as_secs_f32() * 1e3,
                        };
                        if let Ok(mut slot) = latest.lock() {
                            *slot = Some(observation);
                        }
                        generation.fetch_add(1, Ordering::AcqRel);
                        ready.store(true, Ordering::Release);
                    }
                })
                .expect("spawn perception thread")
        };
        Self {
            tx,
            latest,
            generation,
            ready,
            handle: Some(handle),
        }
    }

    /// Whether the worker would accept a frame now.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Offer a frame. Returns `false`, without copying it, if the worker is
    /// busy or never started.
    pub fn submit(&self, frame: &Frame) -> bool {
        if self
            .ready
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.tx.send(Command::Detect(frame.clone())).is_ok()
    }

    /// The most recent observation, if any.
    pub fn latest(&self) -> Option<Observation> {
        self.latest.lock().ok().and_then(|o| o.clone())
    }

    /// Count of completed observations, to tell a fresh one from a repeat
    /// without comparing contents.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Stop);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Any other per-frame job on a thread of its own, with the same contract
/// as [`Worker`]: frames offered while it is busy are dropped, and the
/// renderer reads the latest result whenever it likes. Depth runs on one,
/// at its own pace, beside the detector.
pub struct Background<T> {
    tx: mpsc::Sender<Option<Frame>>,
    latest: Arc<Mutex<Option<T>>>,
    generation: Arc<AtomicU64>,
    ready: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl<T: Clone + Send + 'static> Background<T> {
    /// Start the job. `make` runs on the new thread and returns the function
    /// each frame is given to; if it fails, the job logs why and never
    /// becomes ready.
    pub fn spawn<F, J>(name: &str, make: F) -> Self
    where
        F: FnOnce() -> Result<J, String> + Send + 'static,
        J: FnMut(&Frame) -> T + 'static,
    {
        let (tx, rx) = mpsc::channel::<Option<Frame>>();
        let latest = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU64::new(0));
        let ready = Arc::new(AtomicBool::new(false));
        let handle = {
            let (latest, generation, ready) = (
                Arc::clone(&latest),
                Arc::clone(&generation),
                Arc::clone(&ready),
            );
            let label = name.to_owned();
            std::thread::Builder::new()
                .name(name.into())
                .spawn(move || {
                    let mut job = match make() {
                        Ok(job) => job,
                        Err(e) => {
                            log::error!("{label} disabled: {e}");
                            return;
                        }
                    };
                    ready.store(true, Ordering::Release);
                    while let Ok(Some(frame)) = rx.recv() {
                        let result = job(&frame);
                        if let Ok(mut slot) = latest.lock() {
                            *slot = Some(result);
                        }
                        generation.fetch_add(1, Ordering::AcqRel);
                        ready.store(true, Ordering::Release);
                    }
                })
                .expect("spawn background thread")
        };
        Self {
            tx,
            latest,
            generation,
            ready,
            handle: Some(handle),
        }
    }

    /// Offer a frame. Returns `false`, without copying it, if the job is
    /// busy or never started.
    pub fn submit(&self, frame: &Frame) -> bool {
        if self
            .ready
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        self.tx.send(Some(frame.clone())).is_ok()
    }

    /// The most recent result, if any.
    pub fn latest(&self) -> Option<T> {
        self.latest.lock().ok().and_then(|o| o.clone())
    }

    /// Count of completed results.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }
}

impl<T> Drop for Background<T> {
    fn drop(&mut self) {
        let _ = self.tx.send(None);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Eye;
    use crate::perception::{Detection, Label, Rect};
    use std::time::Duration;

    #[test]
    fn background_drops_frames_while_busy_and_keeps_the_latest() {
        let job = Background::spawn("test", || {
            Ok(|frame: &Frame| {
                std::thread::sleep(Duration::from_millis(30));
                frame.timestamp_ns
            })
        });
        let frame = |t: i64| {
            let mut f = Frame::from_rgb8(2, 2, &[0; 12], Eye::Left);
            f.timestamp_ns = t;
            f
        };
        while !job.submit(&frame(1)) {
            std::thread::sleep(Duration::from_millis(1));
        }
        // Busy: refused, not queued.
        assert!(!job.submit(&frame(2)));
        while job.generation() == 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(job.latest(), Some(1));
    }

    /// Reports one person whose box follows the frame timestamp, slowly.
    struct Fake {
        delay: Duration,
    }

    impl Detector for Fake {
        fn detect(&mut self, frame: &Frame) -> Vec<Detection> {
            std::thread::sleep(self.delay);
            let x = 0.1 + frame.timestamp_ns as f32 * 1e-3;
            vec![Detection {
                rect: Rect::new(x, 0.2, x + 0.2, 0.8),
                label: Label::PERSON,
                score: 0.9,
            }]
        }
    }

    fn wait_for_generation(worker: &Worker, generation: u64) {
        let start = Instant::now();
        while worker.generation() < generation {
            assert!(start.elapsed() < Duration::from_secs(5), "worker stalled");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn wait_ready(worker: &Worker) {
        let start = Instant::now();
        while !worker.is_ready() {
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "worker never ready"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn publishes_tracked_observations() {
        let worker = Worker::spawn(
            || {
                Ok(Box::new(Fake {
                    delay: Duration::ZERO,
                }))
            },
            TrackerConfig::default(),
        );
        let mut frame = Frame::new(4, 4, Eye::Left);
        for step in 1..=3u64 {
            wait_ready(&worker);
            frame.timestamp_ns = step as i64;
            assert!(worker.submit(&frame));
            wait_for_generation(&worker, step);
        }
        let observation = worker.latest().unwrap();
        assert_eq!(observation.timestamp_ns, 3);
        assert_eq!(observation.tracks.len(), 1);
        assert_eq!(observation.tracks[0].hits, 3, "one identity across frames");
    }

    #[test]
    fn drops_frames_while_busy() {
        let worker = Worker::spawn(
            || {
                Ok(Box::new(Fake {
                    delay: Duration::from_millis(50),
                }))
            },
            TrackerConfig::default(),
        );
        wait_ready(&worker);
        let frame = Frame::new(4, 4, Eye::Left);
        assert!(worker.submit(&frame));
        assert!(!worker.submit(&frame), "a busy worker must refuse frames");
        wait_for_generation(&worker, 1);
    }

    #[test]
    fn a_failed_detector_never_accepts_frames() {
        let worker = Worker::spawn(|| Err("no weights".into()), TrackerConfig::default());
        std::thread::sleep(Duration::from_millis(20));
        assert!(!worker.submit(&Frame::new(4, 4, Eye::Left)));
        assert!(worker.latest().is_none());
    }
}
