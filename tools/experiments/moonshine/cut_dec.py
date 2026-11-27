"""Cut the simplified decoder after its token-embedding Gather, so the
embedding lookup (16 rows of the table) happens outside and the rest can be
checked by the importer. Inputs become [inputs_embeds, encoder_hidden_states]."""
import sys, os
import numpy as np, onnx, onnxruntime as ort
src_dir, out_dir = sys.argv[1], sys.argv[2]
m = onnx.load(os.path.join(src_dir, "model.onnx"))
g = next(n for n in m.graph.node if n.op_type == "Gather" and n.input[1] == "input_ids")
emb = g.output[0]
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "model.onnx")
onnx.utils.extract_model(os.path.join(src_dir, "model.onnx"), path, [emb, "encoder_hidden_states"], [o.name for o in m.graph.output])
cut = onnx.load(path)
for i in cut.graph.input:
    if i.name == emb:
        d = i.type.tensor_type.shape; d.ClearField("dim")
        for v in (1, 16, 288): d.dim.add().dim_value = v
onnx.save(cut, path)
table = next(t for t in m.graph.initializer if t.name == g.input[0])
table = onnx.numpy_helper.to_array(table)
ids = np.fromfile(os.path.join(src_dir, "input.bin"), np.float32).astype(np.int64)
x = table[ids][None].astype(np.float32)
h = np.load(os.path.join(src_dir, "hidden.npy"))
so = ort.SessionOptions(); so.log_severity_level = 3
outs = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).run(None, {emb: x, "encoder_hidden_states": h})
x.tofile(os.path.join(out_dir, "input.bin")); np.save(os.path.join(out_dir, "hidden.npy"), h)
for k, o in enumerate(outs): o.astype(np.float32).tofile(os.path.join(out_dir, f"output_{k}.bin"))
print("cut at", emb, "table", table.shape, "inputs", [(i.name) for i in cut.graph.input], "outputs", len(outs))
