"""Object-word accuracy over espeak voices, rates and fixed window lengths."""
import sys, os, io, wave, subprocess, json
import numpy as np, onnxruntime as ort
from scipy.signal import resample_poly
from tokenizers import Tokenizer
d = sys.argv[1]
tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
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
def transcribe(x):
    h = enc.run(None, {"input_values": x[None]})[0]; ids = [1]
    while len(ids) < 16:
        nxt = int(dec.run(["logits"], {"input_ids": np.array([ids], np.int64), "encoder_hidden_states": h})[0][0, -1].argmax()); ids.append(nxt)
        if nxt == 2: break
    return tok.decode(ids, skip_special_tokens=True), len(ids)
for voice in ["en-us", "en-us+m3", "en-gb", "en-us+f2"]:
    for rate in [130, 160]:
        speech = [synth(c, voice, rate) for c in cmds]
        for win in [2.0, 3.0, 4.0]:
            N = int(win * 16000); ok = 0; texts = []; steps = []
            for c, s in zip(cmds, speech):
                x = np.concatenate([np.zeros(2400, np.float32), s])[:N]; x = np.pad(x, (0, N - len(x)))
                t, n = transcribe(x); texts.append(t); steps.append(n - 1)
                ok += c.split()[-1] in t.lower()
            print(f"{voice:9s} rate {rate} window {win:.0f}s: {ok}/10 objects, steps max {max(steps)}  {texts}")
