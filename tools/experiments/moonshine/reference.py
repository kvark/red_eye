"""Moonshine-tiny reference: encoder + greedy no-cache decoder in onnxruntime."""
import sys, os, json, glob, wave, time
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

d = sys.argv[1]; audio_dir = sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "pad"
cfg = json.load(open(os.path.join(d, "generation_config.json")))
START, EOS = cfg["decoder_start_token_id"], cfg["eos_token_id"]
tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
so = ort.SessionOptions(); so.log_severity_level = 3
enc = ort.InferenceSession(os.path.join(d, "onnx/encoder_model.onnx"), so, providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(os.path.join(d, "onnx/decoder_model.onnx"), so, providers=["CPUExecutionProvider"])
MAX = 16

def load(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

def trim(x, thr=0.01, margin=1600):
    idx = np.nonzero(np.abs(x) > thr)[0]
    return x[max(0, idx[0] - margin): idx[-1] + margin]

for p in sorted(glob.glob(os.path.join(audio_dir, "*.wav"))):
    x = load(p)
    if mode == "trim": x = trim(x)
    t0 = time.time()
    h = enc.run(None, {"input_values": x[None]})[0]
    t1 = time.time()
    ids = [START]
    while len(ids) < MAX:
        logits = dec.run(["logits"], {"input_ids": np.array([ids], np.int64), "encoder_hidden_states": h})[0]
        nxt = int(logits[0, -1].argmax()); ids.append(nxt)
        if nxt == EOS: break
    t2 = time.time()
    text = tok.decode(ids, skip_special_tokens=True)
    want = os.path.basename(p)[:-4].replace("_", " ")
    obj = want.split()[-1]
    print(f"{os.path.basename(p):26s} samples={x.size:6d} h={list(h.shape)} ids={ids} -> {text!r} object_ok={obj in text.lower()} enc={1e3*(t1-t0):.0f}ms dec={1e3*(t2-t1):.0f}ms")
