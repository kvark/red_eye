"""Static pass mirroring Meganeura b25f763's ONNX importer (src/load/onnx.rs,
onnx/fold.rs): propagate import-time constness the way `fold` and the
shape-only arms do, then list every node that stays on the GPU path and has no
match arm, or whose attributes/shapes an arm rejects."""
import sys, collections
import numpy as np, onnx
from onnx import numpy_helper, shape_inference

TRANSLATE = set("""Relu Sigmoid Neg Abs Log Reciprocal Gelu Silu SiLU Sqrt Exp Tanh Pow ReduceMean ReduceSum ReduceMax Erf
Clip Cast Shape Add Sub Mul Div MatMul Gemm Softmax LogSoftmax LayerNormalization LayerNorm RMSNorm SimplifiedLayerNormalization
Gather Transpose Reshape Flatten Squeeze Unsqueeze Expand Identity Dropout Constant Conv ConvTranspose Resize GridSample TopK
Pad Tile GatherElements Concat Slice Split GroupNormalization GroupNorm BatchNormalization MaxPool GlobalAveragePool""".split())
FOLD = set("Add Sub Mul Div Equal Greater Less And Or Not Sin Cos Log Neg Cast Where Range ConstantOfShape Expand Concat Gather Slice ScatterND ReduceSum MatMul".split())
ALIAS = set("Reshape Squeeze Unsqueeze Identity Dropout Cast Flatten".split())

m = shape_inference.infer_shapes(onnx.load(sys.argv[1]))
g = m.graph
shape = {}
for v in list(g.value_info) + list(g.input) + list(g.output):
    shape[v.name] = [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
const = {t.name: numpy_helper.to_array(t) for t in g.initializer}
for t in g.initializer: shape[t.name] = list(t.dims)
role_hint = lambda n: n.name.rsplit("/", 1)[0] if "/" in n.name else n.name
gaps = collections.defaultdict(list)
def gap(n, why): gaps[(n.op_type, why)].append(role_hint(n))
for n in g.node:
    ins = [i for i in n.input if i]
    allc = all(i in const for i in ins)
    op = n.op_type
    if op == "Constant":
        const[n.output[0]] = numpy_helper.to_array(n.attribute[0].t); continue
    if op == "Shape":
        const[n.output[0]] = np.array(shape.get(n.input[0], []), np.float32); continue
    if allc and (op in FOLD or op in ALIAS):
        try:
            import onnx.reference
            outs = onnx.reference.ReferenceEvaluator(onnx.helper.make_graph([n], "x", [], [onnx.helper.make_tensor_value_info(o, 0, None) for o in n.output],
                    initializer=[numpy_helper.from_array(np.asarray(const[i]), i) for i in ins])).run(None, {})
            if op in ALIAS or (0 < outs[0].size <= 1 << 22):
                for o, v in zip(n.output, outs): const[o] = v
                continue
        except Exception as e:
            pass
    if op not in TRANSLATE:
        gap(n, "no match arm" + (" (not folded: an input is computed on the GPU)" if not allc else " (constant, but fold does not evaluate it)")); continue
    a = {x.name: onnx.helper.get_attribute_value(x) for x in n.attribute}
    xs = shape.get(n.input[0], [])
    if op in ("ReduceMean", "ReduceSum", "ReduceMax"):
        axes = list(a.get("axes", [])) or (list(const[n.input[1]].astype(int)) if len(n.input) > 1 and n.input[1] in const else [])
        if not (len(axes) == 1 and (axes[0] == -1 or axes[0] == len(xs) - 1)): gap(n, f"axes {axes} of rank {len(xs)} (last axis only)")
    elif op in ("Softmax", "LogSoftmax"):
        ax = a.get("axis", -1)
        if ax not in (-1, len(xs) - 1): gap(n, f"axis {ax}")
    elif op == "Pow":
        e = const.get(n.input[1])
        if e is None or e.size != 1 or float(e) not in (1, 2, 3, 0.5, -0.5, -1): gap(n, f"exponent {e}")
    elif op == "Gather" and n.input[1] not in const:
        gap(n, f"indices computed at runtime ({n.input[1]}); needs axis 0 and U32 indices, but the importer makes every graph input F32")
    elif op == "Gemm":
        if a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0: gap(n, "scaled Gemm")
    elif op in ("Expand",) and n.input[1] not in const: gap(n, "runtime shape")
    elif op == "Slice" and not (n.input[1] in const and n.input[2] in const): gap(n, "runtime starts/ends")
    elif op == "Conv":
        p = list(a.get("pads", [])); h = len(p) // 2
        if p and p[:h] != p[h:]: gap(n, f"asymmetric pads {p}")
    elif op == "InstanceNormalization": pass
total = collections.Counter(n.op_type for n in g.node)
print(sys.argv[1], "nodes", len(g.node))
if not gaps: print("  no gaps found")
for (op, why), where in sorted(gaps.items()):
    print(f"  {op} x{len(where)}: {why}; at {sorted(set(where))[:4]}{' ...' if len(set(where)) > 4 else ''}")
