"""Rewrite the ops Meganeura lacks into ones it has, to find the gaps behind
the first one and to check the rest of the graph numerically. Exact
rewrites where they exist; Sin/Cos by a Taylor polynomial (valid for
inputs within about [-pi, 3pi]).

    python -I probe.py in.onnx out.onnx [--ops Einsum,Where,Sin,Cos,...]
"""

import argparse
import math

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--ops", default="Einsum,Where,Sin,Cos,Equal,Not,GreaterOrEqual,Greater,Less")
    args = p.parse_args()
    ops = set(args.ops.split(","))
    m = onnx.load(args.src)
    inf = shape_inference.infer_shapes(m, data_prop=True)
    sh = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
    for t in m.graph.initializer:
        sh[t.name] = list(t.dims)
    g = m.graph
    new_nodes = []
    k = [0]

    def const(arr, dtype=np.float32):
        k[0] += 1
        name = f"_probe_c{k[0]}"
        g.initializer.append(numpy_helper.from_array(np.asarray(arr, dtype), name))
        return name

    def tmp():
        k[0] += 1
        return f"_probe_t{k[0]}"

    changed = {}
    for n in g.node:
        op = n.op_type
        if op not in ops:
            new_nodes.append(n)
            continue
        if op == "ReduceMean":
            axes = [a.ints for a in n.attribute if a.name == "axes"]
            axes = list(axes[0]) if axes else (numpy_helper.to_array(next(t for t in g.initializer if t.name == n.input[1])).tolist() if len(n.input) > 1 else None)
            keep = next((a.i for a in n.attribute if a.name == "keepdims"), 1)
            if axes in ([2, 3], [-2, -1]) and len(sh[n.input[0]]) == 4 and keep:
                changed["ReduceMean->GlobalAveragePool"] = changed.get("ReduceMean->GlobalAveragePool", 0) + 1
                new_nodes.append(helper.make_node("GlobalAveragePool", [n.input[0]], [n.output[0]]))
            else:
                new_nodes.append(n)
            continue
        changed[op] = changed.get(op, 0) + 1
        if op == "ScatterND":
            # Constant row written into a runtime tensor: Slice + Concat.
            data, idx, upd = n.input
            iv = next(numpy_helper.to_array(t) for t in g.initializer if t.name == idx) if any(t.name == idx for t in g.initializer) else None
            if iv is None:
                for c in m.graph.node:
                    if c.op_type == "Constant" and c.output[0] == idx:
                        iv = numpy_helper.to_array(c.attribute[0].t)
            iv = iv.reshape(-1, iv.shape[-1])
            assert iv.shape[0] == 1 and list(iv[0][:-1]) == [0] * (iv.shape[1] - 1), iv
            axis, row = iv.shape[1] - 1, int(iv[0][-1])
            assert row == 0
            rest = tmp()
            new_nodes.append(helper.make_node("Slice", [data, const([1], np.int64), const([2**31], np.int64), const([axis], np.int64)], [rest]))
            new_nodes.append(helper.make_node("Concat", [upd, rest], [n.output[0]], axis=axis))
            continue
        if op == "Einsum":
            eq = helper.get_attribute_value(n.attribute[0]).decode()
            a, b = n.input
            if eq == "bchw,bnc->bnhw":
                bs, c, h, w = sh[a]
                a2 = tmp()
                new_nodes.append(helper.make_node("Reshape", [a, const([bs, c, h * w], np.int64)], [a2]))
                mm = tmp()
                new_nodes.append(helper.make_node("MatMul", [b, a2], [mm]))
                new_nodes.append(helper.make_node("Reshape", [mm, const([bs, sh[b][1], h, w], np.int64)], [n.output[0]]))
            else:
                raise SystemExit(f"einsum {eq} not handled")
        elif op == "Where":
            c, a, b = n.input
            cf, d, md = tmp(), tmp(), tmp()
            new_nodes.append(helper.make_node("Cast", [c], [cf], to=onnx.TensorProto.FLOAT))
            new_nodes.append(helper.make_node("Sub", [a, b], [d]))
            new_nodes.append(helper.make_node("Mul", [cf, d], [md]))
            new_nodes.append(helper.make_node("Add", [md, b], [n.output[0]]))
        elif op in ("Sin", "Cos"):
            # sin(x) = -sin(x - pi); cos(x) = sin(x + pi/2)
            x = n.input[0]
            y = tmp()
            shift = math.pi if op == "Sin" else math.pi - math.pi / 2
            new_nodes.append(helper.make_node("Sub", [x, const(shift)], [y]))
            y2 = tmp()
            new_nodes.append(helper.make_node("Mul", [y, y], [y2]))
            coeffs = [(-1) ** i / math.factorial(2 * i + 1) for i in range(10)]
            acc = const(coeffs[-1])
            for cf_ in reversed(coeffs[:-1]):
                t1, t2 = tmp(), tmp()
                new_nodes.append(helper.make_node("Mul", [acc, y2], [t1]))
                new_nodes.append(helper.make_node("Add", [t1, const(cf_)], [t2]))
                acc = t2
            s = tmp()
            new_nodes.append(helper.make_node("Mul", [acc, y], [s]))
            new_nodes.append(helper.make_node("Neg", [s], [n.output[0]]))
        else:
            # Comparisons and logic as float arithmetic stand-ins (shape only).
            out = n.output[0]
            ins = [i for i in n.input if i]
            casts = []
            for i in ins:
                t = tmp()
                new_nodes.append(helper.make_node("Cast", [i], [t], to=onnx.TensorProto.FLOAT))
                casts.append(t)
            if len(casts) == 1:
                t = tmp()
                new_nodes.append(helper.make_node("Neg", [casts[0]], [t]))
            else:
                t = tmp()
                new_nodes.append(helper.make_node("Sub", casts[:2], [t]))
            new_nodes.append(helper.make_node("Cast", [t], [out], to=onnx.TensorProto.BOOL))
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, args.dst)
    print("rewrote", changed)


if __name__ == "__main__":
    main()
