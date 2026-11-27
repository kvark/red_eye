import sys, onnx
from onnx import shape_inference
m = onnx.load(sys.argv[1]); op = sys.argv[2]; depth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
m = shape_inference.infer_shapes(m)
vi = {v.name: [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim] for v in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output)}
for t in m.graph.initializer: vi[t.name] = list(t.dims)
nodes = list(m.graph.node)
for i, n in enumerate(nodes):
    if n.op_type == op:
        for j in range(max(0, i - depth), min(len(nodes), i + depth + 1)):
            k = nodes[j]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in k.attribute if a.name != "value"}
            print((">> " if j == i else "   ") + f"{k.op_type} {attrs} in={[(x, vi.get(x)) for x in k.input]} out={[(x, vi.get(x)) for x in k.output]}")
        print("----")
        if len(sys.argv) > 4: break
