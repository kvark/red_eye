//! The voice thread: microphone, utterances, Moonshine, commands.
//!
//! Runs beside detection on the shared GPU context. Hearing a command takes
//! a burst of GPU work once per utterance, not per frame, so it needs no
//! frame dropping; utterances spoken while one is being heard wait their
//! turn in the microphone's buffer.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use blade_graphics as gpu;
use log::{info, warn};
use red_eye::voice::{Grammar, MIN_CONFIDENCE, Moonshine, Recognition, Vad, VadConfig};

use crate::microphone::Microphone;

/// What the wearer said, as the thread heard it.
#[derive(Clone, Debug)]
pub struct Heard {
    pub recognition: Recognition,
    /// Confident enough to act on.
    pub accepted: bool,
    /// Time from the end of speech to the command.
    pub latency_ms: f32,
}

struct Shared {
    /// Whether to listen: only while the app has focus.
    listening: AtomicBool,
    /// Whether someone is speaking now, for the HUD.
    speaking: AtomicBool,
    stop: AtomicBool,
    latest: Mutex<Option<Heard>>,
    generation: AtomicU64,
}

pub struct Listener {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

impl Listener {
    /// Start the voice thread on the model `tools/prepare_moonshine.py`
    /// wrote to `dir`, if it is there.
    pub fn spawn(dir: &Path, context: &Arc<gpu::Context>) -> Option<Self> {
        if !dir.join("commands.txt").exists() {
            warn!("no voice model at {}; no voice commands", dir.display());
            return None;
        }
        let shared = Arc::new(Shared {
            listening: AtomicBool::new(true),
            speaking: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            latest: Mutex::new(None),
            generation: AtomicU64::new(0),
        });
        let (dir, context, thread_shared) =
            (dir.to_path_buf(), Arc::clone(context), Arc::clone(&shared));
        let handle = std::thread::Builder::new()
            .name("voice".into())
            .spawn(move || {
                if let Err(e) = run(&dir, context, &thread_shared) {
                    warn!("voice commands stopped: {e}");
                }
            })
            .ok()?;
        Some(Self {
            shared,
            handle: Some(handle),
        })
    }

    /// Listen or not: the microphone is open only while listening.
    pub fn set_listening(&self, listening: bool) {
        self.shared.listening.store(listening, Ordering::Relaxed);
    }

    pub fn speaking(&self) -> bool {
        self.shared.speaking.load(Ordering::Relaxed)
    }

    /// Counts what has been heard, so the caller can tell a new result.
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(Ordering::Acquire)
    }

    pub fn latest(&self) -> Option<Heard> {
        self.shared.latest.lock().unwrap().clone()
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn run(dir: &Path, context: Arc<gpu::Context>, shared: &Shared) -> Result<(), String> {
    let grammar = Grammar::parse(
        &std::fs::read_to_string(dir.join("commands.txt")).map_err(|e| e.to_string())?,
    )?;
    let mut model = Moonshine::new(dir, Some(context))?;
    info!("voice ready: {} command phrases", grammar.phrases());
    let mut microphone: Option<Microphone> = None;
    let mut vad = Vad::new(VadConfig::default());
    let mut last_failure: Option<Instant> = None;
    while !shared.stop.load(Ordering::Relaxed) {
        if !shared.listening.load(Ordering::Relaxed) {
            if microphone.take().is_some() {
                // A fresh detector next time: the room may have changed.
                vad = Vad::new(VadConfig::default());
                shared.speaking.store(false, Ordering::Relaxed);
            }
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let Some(ref mut mic) = microphone else {
            // Retry a failed open now and then, not in a busy loop.
            if last_failure.is_some_and(|t| t.elapsed() < Duration::from_secs(5)) {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            match Microphone::open() {
                Ok(mic) => microphone = Some(mic),
                Err(e) => {
                    warn!("microphone: {e}");
                    last_failure = Some(Instant::now());
                }
            }
            continue;
        };
        let audio = match mic.read(Duration::from_millis(100)) {
            Ok(audio) => audio,
            Err(e) => {
                warn!("microphone read: {e}; reopening");
                microphone = None;
                continue;
            }
        };
        let utterances = vad.push(&audio);
        shared.speaking.store(vad.speaking(), Ordering::Relaxed);
        for utterance in utterances {
            let start = Instant::now();
            model.encode(&utterance);
            let Some(recognition) = model.recognise(&grammar) else {
                continue;
            };
            let heard = Heard {
                accepted: recognition.confidence > MIN_CONFIDENCE,
                latency_ms: start.elapsed().as_secs_f32() * 1000.0,
                recognition,
            };
            info!(
                "heard {:?} ({:.2}, {}) in {:.0} ms",
                heard.recognition.phrase,
                heard.recognition.confidence,
                if heard.accepted {
                    "accepted"
                } else {
                    "rejected"
                },
                heard.latency_ms
            );
            *shared.latest.lock().unwrap() = Some(heard);
            shared.generation.fetch_add(1, Ordering::Release);
        }
    }
    Ok(())
}
