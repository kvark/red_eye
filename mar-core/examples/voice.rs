//! Voice commands from WAV files: what each says, how sure the model is,
//! and how long hearing it takes. For calibrating `voice::MIN_CONFIDENCE`
//! against commands and against other speech.
//!
//! ```text
//! cargo run --release --example voice -- ref/moonshine_tiny_2s a.wav b.wav ...
//! ```
//!
//! WAVs are 16-bit mono at any rate; each goes through the voice activity
//! detector first, as the microphone's audio does.

use std::path::Path;
use std::time::Instant;

use mar_core::voice::vad::{self, SAMPLE_RATE};
use mar_core::voice::{Command, Grammar, MIN_CONFIDENCE, Moonshine, Vad, VadConfig};

fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let field = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{}: not a WAV", path.display()));
    }
    // Walk the chunks for the format and the data.
    let (mut at, mut rate, mut channels, mut samples) = (12, 0, 0, Vec::new());
    while at + 8 <= bytes.len() {
        let (id, len) = (&bytes[at..at + 4], field(at + 4) as usize);
        let body = &bytes[at + 8..(at + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => {
                channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes([body[14], body[15]]);
                if bits != 16 {
                    return Err(format!("{}: {bits}-bit, want 16", path.display()));
                }
            }
            b"data" => {
                samples = body
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .step_by(channels.max(1))
                    .map(|&b| i16::from_le_bytes(b) as f32 / 32768.0)
                    .collect();
            }
            _ => {}
        }
        at += 8 + len + (len & 1);
    }
    Ok(vad::resample(&samples, rate, SAMPLE_RATE))
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (dir, wavs) = args.split_first().ok_or("usage: voice MODEL_DIR WAV...")?;
    let dir = Path::new(dir);
    let grammar = Grammar::parse(
        &std::fs::read_to_string(dir.join("commands.txt")).map_err(|e| e.to_string())?,
    )?;
    let mut model = Moonshine::new(dir, None)?;
    println!("{} phrases", grammar.phrases());
    for wav in wavs {
        let audio = read_wav(Path::new(wav))?;
        // Room tone before, as a microphone would give, then the clip.
        let mut vad = Vad::new(VadConfig::default());
        let mut utterances = vad.push(&vec![0.0; SAMPLE_RATE as usize / 2]);
        utterances.extend(vad.push(&audio));
        utterances.extend(vad.push(&vec![0.0; SAMPLE_RATE as usize]));
        let name = Path::new(wav).file_name().unwrap().to_string_lossy();
        if utterances.is_empty() {
            println!("{name:40} no speech");
            continue;
        }
        for utterance in utterances {
            let start = Instant::now();
            model.encode(&utterance);
            let heard = model.recognise(&grammar);
            let ms = start.elapsed().as_secs_f32() * 1000.0;
            match heard {
                Some(heard) => {
                    let what = match heard.command {
                        Command::Track(label) => format!("track {}", label.name()),
                        Command::Cancel => "cancel".into(),
                    };
                    let verdict = if heard.confidence > MIN_CONFIDENCE {
                        "accept"
                    } else {
                        "reject"
                    };
                    println!(
                        "{name:40} {:.2} s  {what:24} confidence {:6.2} {verdict}  {ms:5.0} ms",
                        utterance.len() as f32 / SAMPLE_RATE as f32,
                        heard.confidence
                    );
                }
                None => println!("{name:40} nothing  {ms:5.0} ms"),
            }
        }
    }
    Ok(())
}
