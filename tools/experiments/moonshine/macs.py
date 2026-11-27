"""Multiply-accumulates of MatMul/Gemm/Conv in a static-shape ONNX graph."""
import sys, collections
import numpy as np, onnx
from onnx import shape_inference
m = shape_inference.infer_shapes(onnx.load(sys.argv[1]))
g = m.graph
sh = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(g.value_info) + list(g.input) + list(g.output)}
for t in g.initializer: sh[t.name] = list(t.dims)
tot = collections.Counter()
for n in g.node:
    if n.op_type in ("MatMul", "Gemm"):
        a, b = sh[n.input[0]], sh[n.input[1]]; o = sh[n.output[0]]
        k = a[-2] if (n.op_type == "Gemm" and any(x.name == "transA" and x.i for x in n.attribute)) else a[-1]
        macs = int(np.prod(o)) * k
        key = "vocab head" if o[-1] == 32768 else ("attention (QK, AV)" if n.input[1] not in {t.name for t in g.initializer} else "projections/MLP")
        tot[key] += macs
    elif n.op_type == "Conv":
        w = sh[n.input[1]]; o = sh[n.output[0]]
        tot["conv stem"] += int(np.prod(o)) * int(np.prod(w[1:]))
print(sys.argv[1], {k: f"{v/1e6:.1f} M" for k, v in tot.items()}, f"total {sum(tot.values())/1e6:.1f} MMAC")
