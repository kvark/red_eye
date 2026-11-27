"""Replace each InstanceNormalization with ReduceMean/Sub/Mul/Sqrt/Div (an
equivalent the importer takes), to find what fails after it. Writes
out_dir/model.onnx plus an onnxruntime reference on a real WAV, or on the
original input.bin of a reference dir."""
import sys, os, wave
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, numpy_helper

src, out_dir, wav = sys.argv[1], sys.argv[2], sys.argv[3]
m = onnx.load(src)
g = m.graph
consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
for n in g.node:
    if n.op_type == "Constant":
        consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
new = []
k = 0
for n in g.node:
    if n.op_type != "InstanceNormalization":
        new.append(n); continue
    eps = [a.f for a in n.attribute if a.name == "epsilon"][0]
    x, s, b = n.input; y = n.output[0]; p = f"_in{k}_"; k += 1
    # scale/bias are per channel [C]; here C == 1 so they broadcast as scalars.
    assert consts[s].size == 1 and consts[b].size == 1, "only C==1 handled"
    g.initializer.extend([numpy_helper.from_array(np.array(eps, np.float32), p + "eps"),
                          numpy_helper.from_array(consts[s].reshape(()).astype(np.float32), p + "s"),
                          numpy_helper.from_array(consts[b].reshape(()).astype(np.float32), p + "b")])
    new += [helper.make_node("ReduceMean", [x], [p + "mean"], axes=[-1], keepdims=1),
            helper.make_node("Sub", [x, p + "mean"], [p + "d"]),
            helper.make_node("Mul", [p + "d", p + "d"], [p + "d2"]),
            helper.make_node("ReduceMean", [p + "d2"], [p + "var"], axes=[-1], keepdims=1),
            helper.make_node("Add", [p + "var", p + "eps"], [p + "ve"]),
            helper.make_node("Sqrt", [p + "ve"], [p + "std"]),
            helper.make_node("Div", [p + "d", p + "std"], [p + "n"]),
            helper.make_node("Mul", [p + "n", p + "s"], [p + "ns"]),
            helper.make_node("Add", [p + "ns", p + "b"], [y])]
del g.node[:]; g.node.extend(new)
os.makedirs(out_dir, exist_ok=True)
onnx.checker.check_model(m)
path = os.path.join(out_dir, "model.onnx"); onnx.save(m, path)
if wav.endswith(".wav"):
    with wave.open(wav) as w:
        x = (np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768)[None]
else:
    x = np.fromfile(wav, np.float32).reshape(1, -1)
so = ort.SessionOptions(); so.log_severity_level = 3
ref = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"]).run(None, {"input_values": x})[0]
got = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).run(None, {"input_values": x})[0]
print("patched vs original max abs diff", float(np.abs(ref - got).max()), "shape", got.shape)
x.tofile(os.path.join(out_dir, "input.bin")); ref.astype(np.float32).tofile(os.path.join(out_dir, "output_0.bin"))
