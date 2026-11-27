"""Greedy decoding constrained to a command grammar: verb + article + COCO
label (or "cancel"), as a token trie over the tokenizer's own encodings."""
import sys, os, io, wave, subprocess
import numpy as np, onnxruntime as ort
from scipy.signal import resample_poly
from tokenizers import Tokenizer
d, labels_path = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
labels = [l.strip() for l in open(labels_path) if l.strip()]
phrases = ["Cancel"] + [f"{v} {a} {l}" for v in ["Track", "Target", "Find", "Lock on"] for a in ["the", "a"] for l in labels]
trie = {}
for p in phrases:
    node = trie
    for t in tok.encode(p, add_special_tokens=False).ids + [2]:
        node = node.setdefault(t, {})
so = ort.SessionOptions(); so.log_severity_level = 3
enc = ort.InferenceSession(os.path.join(d, "onnx/encoder_model.onnx"), so, providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(os.path.join(d, "onnx/decoder_model.onnx"), so, providers=["CPUExecutionProvider"])
cmds = ["track the motorcycle", "target the person", "find a car", "lock on the dog", "track the bicycle", "cancel",
        "track the truck", "find the cup", "lock on the cat", "target the bus"]
def synth(c, voice, rate):
    raw = subprocess.run(["espeak-ng", "-v", voice, "-s", str(rate), "--stdout", c], check=True, capture_output=True).stdout
    with wave.open(io.BytesIO(raw)) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return resample_poly(x, 16000, sr).astype(np.float32)
def decode(h, constrained):
    ids, node = [1], trie
    while len(ids) < 16:
        logits = dec.run(["logits"], {"input_ids": np.array([ids], np.int64), "encoder_hidden_states": h})[0][0, -1]
        if constrained:
            allowed = np.array(list(node)); nxt = int(allowed[logits[allowed].argmax()]); node = node[nxt]
        else:
            nxt = int(logits.argmax())
        ids.append(nxt)
        if nxt == 2: break
    return tok.decode(ids, skip_special_tokens=True), len(ids) - 1
N = 32000
for voice, rate in [("en-us", 130), ("en-us", 150), ("en-us", 160), ("en-us+m3", 130), ("en-us+m3", 160), ("en-gb", 130), ("en-gb", 160), ("en-us+f2", 130), ("en-us+f2", 160)]:
    res = {False: [0, []], True: [0, []]}; maxsteps = 0
    for c in cmds:
        s = synth(c, voice, rate); x = np.concatenate([np.zeros(2400, np.float32), s])[:N]; x = np.pad(x, (0, N - len(x)))
        h = enc.run(None, {"input_values": x[None]})[0]
        for k in (False, True):
            t, n = decode(h, k); maxsteps = max(maxsteps, n)
            res[k][0] += c.split()[-1] in t.lower().split() or c == "cancel" and "cancel" in t.lower(); res[k][1].append(t)
    print(f"{voice:9s} {rate}: free {res[False][0]}/10, constrained {res[True][0]}/10, max steps {maxsteps}; constrained: {res[True][1]}")
