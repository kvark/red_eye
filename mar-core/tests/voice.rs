//! Moonshine-tiny through Meganeura's ONNX importer against onnxruntime,
//! and a spoken command heard through the grammar.
//!
//! Needs a GPU (lavapipe is enough) and the prepared exports, with a
//! recording of "track the bicycle" (16-bit mono at any rate; any synthesiser):
//!
//! ```text
//! tools/fetch_moonshine.sh track_the_bicycle.wav
//! cargo test --release --test voice -- --include-ignored
//! ```

mod common;

use std::path::Path;

use mar_core::perception::Label;
use mar_core::voice::{Command, Grammar, MIN_CONFIDENCE, Moonshine};

const DIR: &str = "ref/moonshine_tiny_2s";

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytemuck::pod_collect_to_vec(&bytes)
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytemuck::pod_collect_to_vec(&bytes)
}

fn worst_relative_error(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len());
    got.iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs() / (1.0 + e.abs()))
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "requires a working GPU context and ref/moonshine_tiny_2s"]
fn moonshine_matches_onnxruntime() {
    common::at_references();
    let dir = Path::new(DIR);
    let mut model = Moonshine::new(dir, None).expect("model");
    let hidden = model
        .encode(&read_f32(&dir.join("encoder/input.bin")))
        .to_vec();
    let error = worst_relative_error(&hidden, &read_f32(&dir.join("encoder/output.bin")));
    println!("encoder: worst relative error {error:.2e}");
    assert!(error < 1e-3);
    // The decoder's reference ran on these same hidden states.
    let ids = read_u32(&dir.join("decoder/ids.bin"));
    let logits = model.logits(&ids).to_vec();
    let error = worst_relative_error(&logits, &read_f32(&dir.join("decoder/logits.bin")));
    println!("decoder: worst relative error {error:.2e}");
    assert!(error < 1e-3);
}

#[test]
#[ignore = "requires a working GPU context and ref/moonshine_tiny_2s with speech"]
fn a_spoken_command_is_recognised() {
    common::at_references();
    let dir = Path::new(DIR);
    let grammar =
        Grammar::parse(&std::fs::read_to_string(dir.join("commands.txt")).expect("commands.txt"))
            .expect("grammar");
    let mut model = Moonshine::new(dir, None).expect("model");
    model.encode(&read_f32(&dir.join("speech.bin")));
    let free = model.transcribe();
    println!("free decoding: {free:?}");
    let heard = model.recognise(&grammar).expect("a command");
    println!("constrained: {heard:?}");
    let bicycle = Command::Track(Label::from_name("bicycle").unwrap());
    let accepted = heard.confidence > MIN_CONFIDENCE;
    // Never act on the wrong command: whatever the recording's voice, a
    // command confident enough to accept must be the one spoken.
    if accepted {
        assert_eq!(heard.command, bicycle);
    }
    // And where the model hears the phrase even unconstrained, the grammar
    // must find and accept it. A robotic synthesiser can defeat the model
    // (espeak-ng's default voice comes out as "Track the mysql"); then only
    // the rule above applies.
    let said = std::fs::read_to_string(dir.join("speech.txt")).unwrap_or_default();
    if said.to_lowercase().contains("bicycle") {
        assert!(
            accepted && heard.command == bicycle,
            "free decoding heard {said:?}, the grammar {heard:?}"
        );
    }
}
