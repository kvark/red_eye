import sys, collections, onnx
for p in sys.argv[1:]:
    m = onnx.load(p, load_external_data=False)
    g = m.graph
    inits = {t.name for t in g.initializer}
    def dims(v): return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]
    print(p, "opset", [(o.domain, o.version) for o in m.opset_import])
    for i in g.input:
        if i.name not in inits: print("  in ", i.name, i.type.tensor_type.elem_type, dims(i))
    for o in g.output: print("  out", o.name, o.type.tensor_type.elem_type, dims(o))
    c = collections.Counter(n.op_type for n in g.node)
    print("  nodes", len(g.node), "inits", len(g.initializer), dict(c.most_common()))
