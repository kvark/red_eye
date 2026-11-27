"""Fix the no-cache decoder's input shapes (input_ids [1, L], encoder_hidden_states
[1, T, D]), optionally keep only `logits`, optionally constant-fold with onnxsim,
and write out_dir/model.onnx with an onnxruntime reference.

input.bin holds input_ids as f32 (onnx_check sets only the first input), so the
reference is for orientation only: encoder_hidden_states stays unset there."""
import argparse, os, json
import numpy as np, onnx, onnxruntime as ort, onnxsim

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("out_dir")
p.add_argument("--len", type=int, default=16); p.add_argument("--t", type=int, default=165); p.add_argument("--d", type=int, default=288)
p.add_argument("--logits-only", action="store_true"); p.add_argument("--simplify", action="store_true")
p.add_argument("--hidden", help="encoder hidden states .npy for the reference")
a = p.parse_args()
m = onnx.load(a.model)
shapes = {"input_ids": [1, a.len], "encoder_hidden_states": [1, a.t, a.d]}
if a.logits_only:
    keep = [o for o in m.graph.output if o.name == "logits"]
    del m.graph.output[:]; m.graph.output.extend(keep)
if a.simplify:
    m, ok = onnxsim.simplify(m, overwrite_input_shapes=shapes, skipped_optimizers=["fuse_pad_into_conv"])
    assert ok
else:
    for i in m.graph.input:
        if i.name in shapes:
            d = i.type.tensor_type.shape; d.ClearField("dim")
            for v in shapes[i.name]: d.dim.add().dim_value = v
os.makedirs(a.out_dir, exist_ok=True)
path = os.path.join(a.out_dir, "model.onnx"); onnx.save(m, path)
ids = np.array([[1, 17026, 278, 10992, 23090] + [2] * (a.len - 5)], np.int64)
h = np.load(a.hidden) if a.hidden else np.random.default_rng(1984).standard_normal((1, a.t, a.d)).astype(np.float32)
so = ort.SessionOptions(); so.log_severity_level = 3
s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
outs = s.run(None, {"input_ids": ids, "encoder_hidden_states": h})
ids.astype(np.float32).tofile(os.path.join(a.out_dir, "input.bin"))
np.save(os.path.join(a.out_dir, "hidden.npy"), h)
for k, o in enumerate(outs): o.astype(np.float32).tofile(os.path.join(a.out_dir, f"output_{k}.bin"))
meta = {"inputs": {k: v for k, v in shapes.items()}, "outputs": [{"name": o.name, "shape": list(r.shape)} for o, r in zip(s.get_outputs(), outs)],
        "nodes": len(m.graph.node), "note": "onnx_check sets only input_ids; encoder_hidden_states is unset there"}
json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"), indent=1)
print(json.dumps({k: meta[k] for k in ("inputs", "nodes")}), [o["name"] for o in meta["outputs"]][:3], len(meta["outputs"]))
