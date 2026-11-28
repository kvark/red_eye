//! Moonshine-tiny: speech to text, for voice commands.
//!
//! Moonshine (Useful Sensors, MIT, <https://arxiv.org/abs/2410.15608>)
//! reads the raw 16 kHz waveform, with no spectrogram, and is small: 27M
//! parameters, an encoder of 0.76 GMAC for 2 s and 0.38 GMAC per decoder
//! step. Meganeura imports the Transformers export
//! (`onnx-community/moonshine-tiny-ONNX`) once `tools/prepare_moonshine.py`
//! fixes its shapes.
//!
//! Static shapes mean the encoder always hears exactly its window (shorter
//! audio is padded with silence) and the decoder always takes its full
//! token count: the transcript so far, padded with end tokens. Causal
//! attention keeps the padding from reaching earlier positions, so the
//! logits at the transcript's last position are the next token's.

use std::path::Path;
use std::sync::Arc;

use super::grammar::{self, END, Grammar, Recognition, START};

/// One imported network, its session and weights set.
struct Network {
    session: meganeura::Session,
    /// Each input's name and element count, from the graph: the plan's
    /// buffers may be padded.
    inputs: Vec<(String, usize)>,
    /// The first output's element count.
    output: usize,
}

impl Network {
    fn load(model: &Path, gpu: Option<Arc<blade_graphics::Context>>) -> Result<Self, String> {
        let onnx = meganeura::load::onnx::load_onnx(model)
            .map_err(|e| format!("{}: {e}", model.display()))?;
        let graph = &onnx.graph;
        let inputs = graph
            .nodes()
            .iter()
            .filter_map(|node| match node.op {
                meganeura::graph::Op::Input { ref name } => {
                    Some((name.clone(), node.ty.shape.iter().product()))
                }
                _ => None,
            })
            .collect();
        let output = graph
            .outputs()
            .first()
            .map(|&id| graph.node(id).ty.shape.iter().product())
            .ok_or_else(|| format!("{}: no output", model.display()))?;
        let (mut session, _) = meganeura::build(
            graph,
            meganeura::SessionConfig {
                mode: meganeura::Mode::Inference,
                gpu,
                ..Default::default()
            },
        );
        for (name, _) in session.plan().param_buffers.clone() {
            let data = onnx
                .weights
                .get(&name)
                .ok_or_else(|| format!("{}: no weight for {name}", model.display()))?;
            session.set_parameter(&name, data);
        }
        Ok(Self {
            session,
            inputs,
            output,
        })
    }

    /// How many values the input called `name` takes.
    fn input(&self, name: &str) -> Result<usize, String> {
        self.inputs
            .iter()
            .find(|(n, _)| n == name)
            .map(|&(_, len)| len)
            .ok_or_else(|| format!("no input {name}"))
    }
}

/// Moonshine's encoder and decoder, compiled.
pub struct Moonshine {
    encoder: Network,
    decoder: Network,
    /// Audio samples the encoder takes.
    samples: usize,
    /// Encoder output, frames × width, which the decoder attends to.
    hidden: Vec<f32>,
    /// Token positions the decoder takes.
    tokens: usize,
    /// All positions' logits, positions × vocabulary.
    logits: Vec<f32>,
    vocab: usize,
}

impl Moonshine {
    /// Load the encoder and decoder that `tools/prepare_moonshine.py` wrote
    /// to `dir`, and compile them on `gpu` (a fresh context if `None`).
    pub fn new(dir: &Path, gpu: Option<Arc<blade_graphics::Context>>) -> Result<Self, String> {
        let encoder = Network::load(&dir.join("encoder").join("model.onnx"), gpu.clone())?;
        let decoder = Network::load(&dir.join("decoder").join("model.onnx"), gpu)?;
        let samples = encoder.input("input_values")?;
        let hidden_len = decoder.input("encoder_hidden_states")?;
        let tokens = decoder.input("input_ids")?;
        let logits_len = decoder.output;
        if logits_len % tokens != 0 {
            return Err(format!(
                "{logits_len} logits do not split over {tokens} positions"
            ));
        }
        log::info!(
            "Moonshine: {:.1} s window, {} decoder positions, {} + {} dispatches",
            samples as f32 / super::vad::SAMPLE_RATE as f32,
            tokens,
            encoder.session.plan().dispatches.len(),
            decoder.session.plan().dispatches.len(),
        );
        Ok(Self {
            encoder,
            decoder,
            samples,
            hidden: vec![0.0; hidden_len],
            tokens,
            logits: vec![0.0; logits_len],
            vocab: logits_len / tokens,
        })
    }

    /// Audio samples one utterance may hold.
    pub fn window(&self) -> usize {
        self.samples
    }

    /// Token positions the decoder takes, start and end included.
    pub fn positions(&self) -> usize {
        self.tokens
    }

    /// Encode `audio` (16 kHz mono, -1..1; padded or cut to the window).
    /// Returns the hidden states, which the decoding calls then use.
    pub fn encode(&mut self, audio: &[f32]) -> &[f32] {
        let mut input = vec![0.0; self.samples];
        let n = audio.len().min(self.samples);
        input[..n].copy_from_slice(&audio[..n]);
        let session = &mut self.encoder.session;
        session.set_input("input_values", &input);
        session.step();
        session.wait_read_output(0, &mut self.hidden);
        self.decoder
            .session
            .set_input("encoder_hidden_states", &self.hidden);
        &self.hidden
    }

    /// The decoder's logits for every position, given `transcript` (start
    /// token first, at most `positions()` long) over the last encoded
    /// audio. Row `transcript.len() - 1` predicts the next token.
    pub fn logits(&mut self, transcript: &[u32]) -> &[f32] {
        let mut ids = vec![END; self.tokens];
        let n = transcript.len().min(self.tokens);
        ids[..n].copy_from_slice(&transcript[..n]);
        let session = &mut self.decoder.session;
        session.set_input_u32("input_ids", &ids);
        session.step();
        session.wait_read_output(0, &mut self.logits);
        &self.logits
    }

    fn next_logits(&mut self, transcript: &[u32]) -> Vec<f32> {
        let row = transcript.len() - 1;
        let vocab = self.vocab;
        self.logits(transcript)[row * vocab..(row + 1) * vocab].to_vec()
    }

    /// What the model hears in the last encoded audio, freely: token ids,
    /// start and end included. For tests and the log.
    pub fn transcribe(&mut self) -> Vec<u32> {
        let mut transcript = vec![START];
        while transcript.len() < self.tokens && transcript.last() != Some(&END) {
            let next = self.next_logits(&transcript);
            let token = (0..next.len())
                .max_by(|&a, &b| next[a].total_cmp(&next[b]))
                .unwrap_or(END as usize);
            transcript.push(token as u32);
        }
        transcript
    }

    /// The command `grammar` allows that the last encoded audio is likeliest
    /// to be, and how sure the model is.
    pub fn recognise(&mut self, grammar: &Grammar) -> Option<Recognition> {
        if grammar.longest() > self.tokens {
            log::warn!(
                "the grammar's phrases take {} positions; the decoder has {}",
                grammar.longest(),
                self.tokens
            );
            return None;
        }
        grammar::decode(grammar, |transcript| self.next_logits(transcript))
    }
}
