"""Turn the cut decoder's encoder_hidden_states input into an initializer
holding a real encoder output, so onnx_check (which sets only the first input)
compares the decoder numerically. Cross-attention K/V then fold at import."""
import sys, os, wave
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
cut_dir, out_dir, enc_path, wav = sys.argv[1:5]
with wave.open(wav) as w:
    x = (np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768)[None]
so = ort.SessionOptions(); so.log_severity_level = 3
h = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"]).run(None, {"input_values": x})[0]
m = onnx.load(os.path.join(cut_dir, "model.onnx"))
keep = [i for i in m.graph.input if i.name != "encoder_hidden_states"]
del m.graph.input[:]; m.graph.input.extend(keep)
m.graph.initializer.append(numpy_helper.from_array(h.astype(np.float32), "encoder_hidden_states"))
os.makedirs(out_dir, exist_ok=True); path = os.path.join(out_dir, "model.onnx"); onnx.save(m, path)
e = np.fromfile(os.path.join(cut_dir, "input.bin"), np.float32).reshape(1, 16, 288)
logits = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).run(None, {keep[0].name: e})[0]
e.tofile(os.path.join(out_dir, "input.bin")); logits.astype(np.float32).tofile(os.path.join(out_dir, "output_0.bin"))
print("argmax per position", logits[0].argmax(-1)[:6])
