"""Synthesise fixed-length 16 kHz mono command WAVs with espeak-ng (test audio only)."""
import subprocess, sys, os, io, wave
import numpy as np
from scipy.signal import resample_poly

out = sys.argv[1]
N = 64000
cmds = ["track the motorcycle", "target the person", "find a car", "lock on the dog",
        "track the bicycle", "cancel", "track the truck", "find the cup", "lock on the cat", "target the bus"]
os.makedirs(out, exist_ok=True)
for c in cmds:
    raw = subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "--stdout", c], check=True, capture_output=True).stdout
    with wave.open(io.BytesIO(raw)) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    x = resample_poly(x, 16000, sr).astype(np.float32)
    n = len(x)
    # Put 0.25 s of lead-in silence, then pad/trim to 4 s.
    lead = np.zeros(4000, np.float32)
    x = np.concatenate([lead, x])[:N]
    x = np.pad(x, (0, N - len(x)))
    name = c.replace(" ", "_")
    with wave.open(os.path.join(out, name + ".wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"{name}: espeak {sr} Hz, speech {n/16000:.2f} s -> {len(x)} samples")
