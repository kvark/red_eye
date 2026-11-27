import sys, onnx
from onnx import shape_inference
m = onnx.load(sys.argv[1])
try: m = shape_inference.infer_shapes(m, data_prop=True)
except Exception as e: print('infer failed', e)
sh = {vi.name: ([d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim], vi.type.tensor_type.elem_type) for vi in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
inits = {t.name: list(t.dims) for t in m.graph.initializer}
prod = {o: n for n in m.graph.node for o in n.output}
targets = sys.argv[2].split(',')
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
def show(name, d, ind=''):
    n = prod.get(name)
    if n is None: print(ind, name, 'INIT' if name in inits else 'INPUT', inits.get(name, sh.get(name))); return
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute if a.name != 'value'}
    print(ind, n.op_type, n.name, '->', [(o, sh.get(o)) for o in n.output], attrs if len(str(attrs)) < 200 else '')
    if d > 0:
        for i in n.input:
            if i: show(i, d - 1, ind + '   ')
for t in targets:
    n = next(n for n in m.graph.node if n.name == t)
    show(n.output[0], depth)
    print('  consumers:', [(c.op_type, c.name) for c in m.graph.node if n.output[0] in c.input])
