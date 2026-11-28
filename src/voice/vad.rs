//! Voice activity: cutting utterances out of a microphone stream.
//!
//! An energy detector, which is enough for a command spoken to a headset a
//! few centimetres from the mouth. Each 20 ms frame's level is compared
//! with a noise floor that follows the quiet stretches. Speech opens an
//! utterance after a few loud frames, with some audio from before them so
//! the first consonant is kept, and a pause closes it. Utterances longer
//! than the model's window are cut there; very short ones are noise.

/// Samples per second the detector and the model take.
pub const SAMPLE_RATE: u32 = 16_000;
const FRAME: usize = 320; // 20 ms

/// Tuning, in frames of 20 ms and decibels.
#[derive(Clone, Copy, Debug)]
pub struct VadConfig {
    /// Level above the noise floor that counts as speech.
    pub threshold_db: f32,
    /// Level that never counts as speech, however quiet the room.
    pub min_level_db: f32,
    /// Loud frames in a row that open an utterance.
    pub open_frames: usize,
    /// Quiet frames in a row that close one.
    pub close_frames: usize,
    /// Audio kept from before the utterance opened.
    pub preroll_frames: usize,
    /// Fewest loud frames an utterance needs to be kept, and the most
    /// frames it may last before it is cut.
    pub min_voiced: usize,
    pub max_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold_db: 12.0,
            min_level_db: -50.0,
            open_frames: 3,
            close_frames: 20,
            preroll_frames: 10,
            min_voiced: 8,
            // The 2 s the model listens to.
            max_frames: 100,
        }
    }
}

/// Feeds on audio, yields utterances.
pub struct Vad {
    config: VadConfig,
    /// Samples not yet a whole frame.
    pending: Vec<f32>,
    /// Recent frames, for the pre-roll.
    history: std::collections::VecDeque<Vec<f32>>,
    /// The noise floor, in dB; `None` before the first frame.
    floor: Option<f32>,
    loud: usize,
    quiet: usize,
    /// Loud frames in the open utterance.
    voiced: usize,
    utterance: Option<Vec<f32>>,
}

impl Vad {
    pub fn new(config: VadConfig) -> Self {
        Self {
            config,
            pending: Vec::with_capacity(FRAME),
            history: Default::default(),
            floor: None,
            loud: 0,
            quiet: 0,
            voiced: 0,
            utterance: None,
        }
    }

    /// Whether an utterance is open: someone is speaking.
    pub fn speaking(&self) -> bool {
        self.utterance.is_some()
    }

    /// Take `samples` (16 kHz mono, -1..1) and return the utterances they
    /// complete.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut done = Vec::new();
        for &sample in samples {
            self.pending.push(sample);
            if self.pending.len() == FRAME {
                let frame = std::mem::replace(&mut self.pending, Vec::with_capacity(FRAME));
                if let Some(utterance) = self.frame(frame) {
                    done.push(utterance);
                }
            }
        }
        done
    }

    fn frame(&mut self, frame: Vec<f32>) -> Option<Vec<f32>> {
        let c = self.config;
        let power = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
        let level = 10.0 * power.max(1e-12).log10();
        // The first frame is the room: an absolute guess would take a noisy
        // one for speech. Starting mid-word is fine, as the floor drops at
        // once to the first quieter frame.
        let floor = *self.floor.get_or_insert(level);
        let speech = level > c.min_level_db && level > floor + c.threshold_db;
        // The floor rises slowly, so speech does not lift it but a noisier
        // room does, in seconds.
        let rate = if speech { 0.002 } else { 0.05 };
        self.floor = Some(if level < floor {
            level
        } else {
            floor + rate * (level - floor)
        });

        if let Some(ref mut utterance) = self.utterance {
            utterance.extend_from_slice(&frame);
            self.quiet = if speech { 0 } else { self.quiet + 1 };
            self.voiced += speech as usize;
            let frames = utterance.len() / FRAME;
            if self.quiet >= c.close_frames || frames >= c.max_frames {
                let mut utterance = self.utterance.take().expect("open");
                // Trailing silence carries nothing.
                let spoken = frames - self.quiet.min(frames);
                utterance.truncate((spoken + 2).min(frames) * FRAME);
                let voiced = std::mem::take(&mut self.voiced);
                self.quiet = 0;
                self.loud = 0;
                // A click opens an utterance too; only enough voice keeps it.
                return (voiced >= c.min_voiced).then_some(utterance);
            }
            return None;
        }

        self.loud = if speech { self.loud + 1 } else { 0 };
        self.history.push_back(frame);
        if self.history.len() > c.preroll_frames + c.open_frames {
            self.history.pop_front();
        }
        if self.loud >= c.open_frames {
            self.utterance = Some(self.history.drain(..).flatten().collect());
            self.quiet = 0;
            self.voiced = self.loud;
        }
        None
    }
}

/// Resample `input` from `from` Hz to `to` Hz: averaging over each output
/// sample's span (a box filter against aliasing), then linear. Good enough
/// for speech from a microphone that will not open at 16 kHz.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let len = (input.len() as f64 / ratio).floor() as usize;
    (0..len)
        .map(|i| {
            let start = i as f64 * ratio;
            let end = ((i + 1) as f64 * ratio).min(input.len() as f64);
            if ratio <= 1.0 {
                let j = start.floor() as usize;
                let t = (start - j as f64) as f32;
                let next = input.get(j + 1).copied().unwrap_or(input[j]);
                return input[j] * (1.0 - t) + next * t;
            }
            let (a, b) = (
                start.floor() as usize,
                (end.ceil() as usize).min(input.len()),
            );
            let span = &input[a..b.max(a + 1)];
            span.iter().sum::<f32>() / span.len() as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32, amplitude: f32) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE as f32) as usize;
        (0..n)
            .map(|i| amplitude * (i as f32 * 2.0 * std::f32::consts::PI * 300.0 / 16000.0).sin())
            .collect()
    }

    fn noise(seconds: f32, amplitude: f32, seed: &mut u32) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE as f32) as usize;
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                amplitude * ((*seed >> 8) as f32 / (1 << 24) as f32 - 0.5)
            })
            .collect()
    }

    #[test]
    fn a_word_between_silences_is_one_utterance() {
        let mut seed = 7;
        let mut vad = Vad::new(VadConfig::default());
        let mut done = vad.push(&noise(1.0, 0.002, &mut seed));
        assert!(done.is_empty() && !vad.speaking());
        done.extend(vad.push(&tone(0.8, 0.3)));
        assert!(vad.speaking());
        done.extend(vad.push(&noise(1.0, 0.002, &mut seed)));
        assert_eq!(done.len(), 1);
        // The word, a little before it and a little after, not the silence.
        let seconds = done[0].len() as f32 / SAMPLE_RATE as f32;
        assert!((0.8..1.2).contains(&seconds), "{seconds} s");
    }

    #[test]
    fn clicks_and_steady_noise_are_not_speech() {
        let mut seed = 11;
        let mut vad = Vad::new(VadConfig::default());
        // A loud room: steady noise well above the absolute minimum.
        let mut done = vad.push(&noise(3.0, 0.05, &mut seed));
        // A 60 ms click: opens, but is too short to keep.
        done.extend(vad.push(&tone(0.06, 0.5)));
        done.extend(vad.push(&noise(1.0, 0.05, &mut seed)));
        assert!(done.is_empty(), "{} utterances", done.len());
    }

    #[test]
    fn long_speech_is_cut_at_the_window() {
        let mut seed = 3;
        let mut vad = Vad::new(VadConfig::default());
        let mut done = vad.push(&noise(0.5, 0.002, &mut seed));
        done.extend(vad.push(&tone(3.0, 0.3)));
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].len(), 100 * FRAME);
    }

    #[test]
    fn resampling_keeps_duration_and_level() {
        let from = tone(1.0, 0.5);
        let up = resample(&from, 16_000, 48_000);
        assert_eq!(up.len(), 48_000);
        let down = resample(&up, 48_000, 16_000);
        assert_eq!(down.len(), 16_000);
        let rms = |x: &[f32]| (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt();
        assert!((rms(&down) - rms(&from)).abs() < 0.01);
    }
}
