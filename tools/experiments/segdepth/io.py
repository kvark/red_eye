import sys, onnx, collections
for p in sys.argv[1:]:
    m = onnx.load(p, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    f = lambda vi: (vi.name, [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim], onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type))
    print(p.split('/dl/')[-1], 'opset', [(o.domain, o.version) for o in m.opset_import], 'nodes', len(m.graph.node))
    print('  in ', [f(i) for i in m.graph.input if i.name not in inits])
    print('  out', [f(o) for o in m.graph.output])
    print('  ops', collections.Counter(n.op_type for n in m.graph.node).most_common(40))
