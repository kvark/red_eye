"""Static report on an ONNX model for Meganeura: inputs/outputs, op histogram,
runtime ops the importer lacks (with counts and an example node), attribute
forms it rejects, and Conv/MatMul/Gemm MACs from shape inference.

    python -I inspect_onnx.py model.onnx [--shape name=1,3,H,W ...] [--json out.json]
"""

import argparse
import collections
import json

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference

RUNTIME = set(
    """Relu Sigmoid Neg Abs Log Reciprocal Gelu Silu SiLU Sqrt Exp Tanh Pow ReduceMean ReduceSum
    ReduceMax Erf Clip Cast Shape Add Sub Mul Div MatMul Gemm Softmax LogSoftmax LayerNormalization
    LayerNorm RMSNorm SimplifiedLayerNormalization Gather Transpose Reshape Flatten Squeeze Unsqueeze
    Expand Identity Dropout Constant Conv ConvTranspose Resize GridSample TopK Pad Tile GatherElements
    Concat Slice Split GroupNormalization GroupNorm BatchNormalization MaxPool GlobalAveragePool""".split()
)
FOLD_ONLY = set("Equal Greater Less And Or Not Sin Cos Where Range ConstantOfShape ScatterND".split())
# Ops whose output Meganeura's folder evaluates when every input is constant.
FOLDS = FOLD_ONLY | set("Add Sub Mul Div Log Neg Cast Expand Concat Gather Slice ReduceSum MatMul".split())
VIEWS = set("Reshape Unsqueeze Squeeze Flatten Identity Dropout Shape Constant".split())


def attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    return default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--shape", action="append", default=[])
    p.add_argument("--json")
    p.add_argument("--no-macs", action="store_true")
    args = p.parse_args()
    model = onnx.load(args.model)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    fixed = {}
    for s in args.shape:
        name, dims = s.split("=")
        fixed[name] = [int(d) for d in dims.split(",")]
    for i in g.input:
        if i.name in fixed:
            d = i.type.tensor_type.shape
            d.ClearField("dim")
            for v in fixed[i.name]:
                d.dim.add().dim_value = v

    def dims(vi):
        t = vi.type.tensor_type
        return [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in t.shape.dim], onnx.TensorProto.DataType.Name(t.elem_type)

    report = {"inputs": [], "outputs": []}
    for i in g.input:
        if i.name not in inits:
            report["inputs"].append((i.name, *dims(i)))
    for o in g.output:
        report["outputs"].append((o.name, *dims(o)))
    hist = collections.Counter(n.op_type for n in g.node)
    report["nodes"] = len(g.node)
    report["histogram"] = hist.most_common()

    # Constness: initializers, Constant outputs, and outputs of ops whose
    # inputs are all constant and which the importer folds or views.
    const = set(inits)
    for n in g.node:
        if n.op_type == "Constant":
            const.update(n.output)
        elif n.op_type == "Shape":
            const.update(n.output)  # static shapes: the importer knows them
        elif all((x == "" or x in const) for x in n.input) and (n.op_type in FOLDS or n.op_type in VIEWS):
            const.update(n.output)
    runtime_nodes = [n for n in g.node if not (all((x == "" or x in const) for x in n.input) and (n.op_type in FOLDS or n.op_type in VIEWS)) and n.op_type not in ("Constant", "Shape")]

    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as e:  # noqa: BLE001
        print("shape inference failed:", e)
        inferred = model
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        shapes[vi.name] = [d.dim_value if d.HasField("dim_value") else None for d in vi.type.tensor_type.shape.dim]
    for name, t in inits.items():
        shapes[name] = list(t.dims)

    def cval(name):
        if name in inits:
            return numpy_helper.to_array(inits[name])
        for n in g.node:
            if n.op_type == "Constant" and n.output[0] == name:
                return numpy_helper.to_array(attr(n, "value"))
        return None

    gaps = collections.defaultdict(list)
    for n in runtime_nodes:
        op = n.op_type
        why = None
        if op not in RUNTIME:
            why = "op not implemented" + (" at runtime (folds only)" if op in FOLD_ONLY else "")
        elif op in ("Conv", "MaxPool", "AveragePool"):
            strides = attr(n, "strides", [])
            dil = attr(n, "dilations", [])
            pads = attr(n, "pads", [])
            if strides and len(set(strides)) > 1:
                why = f"strides {strides}"
            elif dil and any(d != 1 for d in dil):
                why = f"dilations {dil}"
            elif pads and pads[: len(pads) // 2] != pads[len(pads) // 2:]:
                why = f"asymmetric pads {pads}"
            elif op == "Conv" and len(shapes.get(n.input[1], [])) not in (0, 3, 4):
                why = f"kernel rank {len(shapes.get(n.input[1], []))}"
        elif op == "Pow":
            v = cval(n.input[1])
            if v is None or float(v.reshape(-1)[0]) not in (1, 2, 3, 0.5, -0.5, -1):
                why = f"exponent {None if v is None else v.reshape(-1)[:1]}"
        elif op in ("Softmax", "LogSoftmax"):
            ax = attr(n, "axis", -1)
            r = len(shapes.get(n.input[0]) or [])
            if ax not in (-1, r - 1):
                why = f"axis {ax} of rank {r}"
        elif op in ("ReduceMean", "ReduceSum", "ReduceMax"):
            axes = attr(n, "axes")
            if axes is None and len(n.input) > 1:
                v = cval(n.input[1])
                axes = None if v is None else v.reshape(-1).tolist()
            r = len(shapes.get(n.input[0]) or [])
            if axes is None or not (len(axes) == 1 and axes[0] in (-1, r - 1)):
                why = f"axes {axes} of rank {r}"
        elif op == "Gather":
            if n.input[1] not in const:
                why = f"runtime indices, axis {attr(n, 'axis', 0)}"
        elif op == "Transpose":
            s = shapes.get(n.input[0]) or []
            perm = attr(n, "perm")
            # merge unit axes and axes that move together
            keep = [i for i in range(len(s)) if s[i] != 1]
            pm = [a for a in (perm or []) if a in keep]
            groups = 0
            prev = None
            for a in pm:
                if prev is None or a != prev + 1:
                    groups += 1
                prev = a
            if groups > 4:
                why = f"{groups} moving axes of {s} perm {perm}"
        elif op == "Gemm":
            if attr(n, "alpha", 1.0) != 1.0 or attr(n, "beta", 1.0) != 1.0:
                why = "scaled Gemm"
        elif op in ("Clip",):
            if any(x and x not in const for x in n.input[1:]):
                why = "runtime bounds"
        elif op == "Slice":
            if any(x and x not in const for x in n.input[1:3]):
                why = "runtime starts/ends"
        elif op == "Resize":
            mode = attr(n, "mode", b"nearest")
            if len(n.input) < 3:
                why = "opset-10 scales"
            elif any(x and x not in const for x in n.input[2:]):
                why = "runtime scales/sizes"
        elif op == "TopK":
            v = cval(n.input[1]) if n.input[1] in const else None
            if v is None:
                why = "runtime k"
            elif int(v.reshape(-1)[0]) > 512:
                why = f"k={int(v.reshape(-1)[0])}"
        elif op == "Gelu":
            a = attr(n, "approximate", b"none")
            if a not in (b"none", b"tanh"):
                why = f"approximate {a}"
        elif op in ("Reshape", "Expand", "Tile", "Pad", "Split"):
            if any(x and x not in const for x in n.input[1:]):
                why = "runtime shape/repeats/pads"
        if why:
            gaps[(op, why)].append(n.name)
    report["gaps"] = [(op, why, len(names), names[:3]) for (op, why), names in sorted(gaps.items(), key=lambda kv: -len(kv[1]))]

    macs = collections.Counter()
    unknown = collections.Counter()
    for n in inferred.graph.node:
        op = n.op_type
        def sh(x):
            return shapes.get(x)
        try:
            if op == "Conv":
                out = sh(n.output[0]); w = sh(n.input[1])
                macs[op] += int(np.prod(out)) * int(np.prod(w[1:]))
            elif op == "ConvTranspose":
                x = sh(n.input[0]); w = sh(n.input[1]); gr = attr(n, "group", 1)
                macs[op] += int(np.prod(x)) * (w[1]) * int(np.prod(w[2:]))
            elif op == "MatMul":
                a = sh(n.input[0]); b = sh(n.input[1]); out = sh(n.output[0])
                k = a[-1]
                macs[op] += int(np.prod(out)) * k
            elif op == "Gemm":
                a = sh(n.input[0]); out = sh(n.output[0])
                k = a[0] if attr(n, "transA", 0) else a[1]
                macs[op] += int(np.prod(out)) * k
        except Exception:  # noqa: BLE001
            unknown[op] += 1
    report["gmacs"] = {k: round(v / 1e9, 3) for k, v in macs.items()}
    report["gmacs_total"] = round(sum(macs.values()) / 1e9, 3)
    report["mac_unknown"] = dict(unknown)
    print(json.dumps(report, indent=1, default=str))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=str)


if __name__ == "__main__":
    main()
