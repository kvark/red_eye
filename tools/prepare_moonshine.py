#!/usr/bin/env python3
"""Prepare Moonshine-tiny for Meganeura and the command grammar for the app.

    python tools/prepare_moonshine.py ref/moonshine_tiny ref/moonshine_tiny_2s [--seconds 2] [--tokens 12] [--speech WAV]

Meganeura runs static shapes, so the encoder takes exactly `seconds` of
16 kHz audio and the decoder exactly `tokens` token ids (the app pads and
reads the logits at the current position). Both are simplified at those
shapes, which folds the shape arithmetic away. Writes:

- `encoder/model.onnx`, with `input.bin` (a deterministic test signal) and
  `output.bin` (its hidden states) from onnxruntime;
- `decoder/model.onnx` (logits only), with `ids.bin` (u32), `hidden.bin`
  and `logits.bin` from onnxruntime;
- `commands.txt`: every phrase of the command grammar with its token ids,
  so the app needs no tokenizer. One line per phrase:
  `ACTION<TAB>LABEL<TAB>PHRASE<TAB>IDS`, where LABEL is a COCO name or `-`;
- with `--speech WAV` (16 kHz mono), `speech.bin` holding it padded or cut
  to length, and `speech.txt` with what free decoding hears in it, for the
  end-to-end test.

Needs `pip install onnx onnxruntime onnxsim tokenizers`.
"""

import argparse
import os
import re
import wave

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from tokenizers import Tokenizer

SAMPLE_RATE = 16000
START, END = 1, 2

# The command grammar. The verbs name a target; "cancel" drops it.
VERBS = ["Track", "Target", "Find", "Lock on", "Lock onto", "Follow"]
ARTICLES = ["the", "a", "that"]
CANCELS = ["Cancel", "Stop", "Stand down", "Abort"]
# What people say for a COCO class that is not its name.
SYNONYMS = {
    "person": ["man", "woman", "guy", "human", "kid", "child"],
    "bicycle": ["bike"],
    "motorcycle": ["motorbike"],
    "airplane": ["plane"],
    "car": ["vehicle"],
    "cell phone": ["phone"],
    "couch": ["sofa"],
    "dining table": ["table"],
    "potted plant": ["plant"],
    "sports ball": ["ball"],
    "tv": ["television", "monitor", "screen"],
    "laptop": ["computer"],
    "cup": ["mug"],
}


def coco_names(repo):
    """The app's label names, from `src/perception/labels.rs`."""
    source = open(os.path.join(repo, "src", "perception", "labels.rs"), encoding="utf-8").read()
    block = source[source.index("COCO_NAMES") :]
    block = block[block.index("[", block.index("=")) : block.index("];")]
    return re.findall(r'"([^"]+)"', block)


def phrases(names):
    for name in names:
        for spoken in [name] + SYNONYMS.get(name, []):
            for verb in VERBS:
                for article in ARTICLES:
                    yield "track", name, f"{verb} {article} {spoken}"
    for cancel in CANCELS:
        yield "cancel", "-", cancel


def simplify(model, shapes):
    model, ok = onnxsim.simplify(
        model, overwrite_input_shapes=shapes, skipped_optimizers=["fuse_pad_into_conv"]
    )
    assert ok, "onnxsim could not validate the simplified model"
    return model


def session(path):
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def read_wav(path, samples):
    with wave.open(path) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, f"{path}: want mono 16-bit"
        rate = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if rate != SAMPLE_RATE:
        # Linear is enough for a test recording; espeak-ng writes 22050 Hz.
        t = np.arange(int(len(x) * SAMPLE_RATE / rate)) * rate / SAMPLE_RATE
        x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
    # Some silence first, as a microphone gives before speech.
    x = np.concatenate([np.zeros(SAMPLE_RATE // 10, np.float32), x])
    return np.pad(x, (0, max(0, samples - len(x))))[:samples]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("out_dir")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--tokens", type=int, default=12)
    p.add_argument("--speech")
    a = p.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    samples = int(a.seconds * SAMPLE_RATE)

    # Encoder.
    out = os.path.join(a.out_dir, "encoder")
    os.makedirs(out, exist_ok=True)
    source = os.path.join(a.model_dir, "onnx", "encoder_model.onnx")
    encoder = simplify(onnx.load(source), {"input_values": [1, samples]})
    encoder_path = os.path.join(out, "model.onnx")
    onnx.save(encoder, encoder_path)
    t = np.arange(samples) / SAMPLE_RATE
    rng = np.random.default_rng(1984)
    signal = (0.3 * np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 3 * t) + 0.05 * rng.standard_normal(samples))
    signal = signal.astype(np.float32)[None]
    hidden = session(encoder_path).run(None, {"input_values": signal})[0]
    signal.tofile(os.path.join(out, "input.bin"))
    hidden.astype(np.float32).tofile(os.path.join(out, "output.bin"))
    frames, width = hidden.shape[1], hidden.shape[2]
    print(f"encoder: {samples} samples -> {frames}x{width}, {len(encoder.graph.node)} nodes")

    # Decoder: logits only, at a fixed token count.
    out = os.path.join(a.out_dir, "decoder")
    os.makedirs(out, exist_ok=True)
    decoder = onnx.load(os.path.join(a.model_dir, "onnx", "decoder_model.onnx"))
    keep = [o for o in decoder.graph.output if o.name == "logits"]
    del decoder.graph.output[:]
    decoder.graph.output.extend(keep)
    decoder = simplify(
        decoder, {"input_ids": [1, a.tokens], "encoder_hidden_states": [1, frames, width]}
    )
    decoder_path = os.path.join(out, "model.onnx")
    onnx.save(decoder, decoder_path)
    tokenizer = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    prompt = [START] + tokenizer.encode("Track the bicycle", add_special_tokens=False).ids
    ids = np.array([prompt + [END] * (a.tokens - len(prompt))], np.int64)
    logits = session(decoder_path).run(None, {"input_ids": ids, "encoder_hidden_states": hidden})[0]
    ids.astype(np.uint32).tofile(os.path.join(out, "ids.bin"))
    hidden.astype(np.float32).tofile(os.path.join(out, "hidden.bin"))
    logits.astype(np.float32).tofile(os.path.join(out, "logits.bin"))
    print(f"decoder: {a.tokens} tokens -> {logits.shape}, {len(decoder.graph.node)} nodes")

    # The grammar as token ids.
    lines, longest = [], 0
    for action, label, phrase in phrases(coco_names(repo)):
        tokens = tokenizer.encode(phrase, add_special_tokens=False).ids
        longest = max(longest, len(tokens))
        lines.append(f"{action}\t{label}\t{phrase}\t{' '.join(map(str, tokens))}")
    # The start token, the phrase and the end token must fit.
    assert longest + 2 <= a.tokens, f"a phrase takes {longest} tokens; raise --tokens"
    with open(os.path.join(a.out_dir, "commands.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"commands: {len(lines)} phrases, the longest {longest} tokens")

    if a.speech:
        audio = read_wav(a.speech, samples)[None]
        audio.tofile(os.path.join(a.out_dir, "speech.bin"))
        enc, dec = session(encoder_path), session(decoder_path)
        states = enc.run(None, {"input_values": audio})[0]
        heard = [START]
        while len(heard) < a.tokens and heard[-1] != END or len(heard) == 1:
            padded = np.array([heard + [END] * (a.tokens - len(heard))], np.int64)
            step = dec.run(None, {"input_ids": padded, "encoder_hidden_states": states})[0]
            heard.append(int(step[0, len(heard) - 1].argmax()))
        text = tokenizer.decode(heard, skip_special_tokens=True)
        open(os.path.join(a.out_dir, "speech.txt"), "w", encoding="utf-8").write(text + "\n")
        print(f"speech: free decoding hears {text!r}")


if __name__ == "__main__":
    main()
