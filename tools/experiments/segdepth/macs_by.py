import sys, collections, numpy as np, onnx
from onnx import shape_inference
m = onnx.load(sys.argv[1]); depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
m = shape_inference.infer_shapes(m, data_prop=True)
sh = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim] for vi in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
for t in m.graph.initializer: sh[t.name] = list(t.dims)
acc = collections.Counter(); big = []
for n in m.graph.node:
    mac = 0
    if n.op_type == 'Conv': mac = np.prod(sh[n.output[0]]) * np.prod(sh[n.input[1]][1:])
    elif n.op_type == 'ConvTranspose': mac = np.prod(sh[n.input[0]]) * sh[n.input[1]][1] * np.prod(sh[n.input[1]][2:])
    elif n.op_type in ('MatMul', 'Gemm'):
        a = sh[n.input[0]]; tA = any(x.name=='transA' and x.i for x in n.attribute)
        mac = np.prod(sh[n.output[0]]) * (a[-2] if tA else a[-1])
    elif n.op_type == 'Einsum':
        print('einsum', [x.s for x in n.attribute], [sh.get(i) for i in n.input], sh.get(n.output[0]))
    if mac:
        key = '/'.join(n.name.strip('/').split('/')[:depth]); acc[key] += mac; big.append((mac, n.name, n.op_type, [sh.get(i) for i in n.input]))
for k, v in acc.most_common(15): print(f'{v/1e9:7.3f} G  {k}')
print('total', sum(acc.values())/1e9)
for b in sorted(big, reverse=True)[:8]: print(f'{b[0]/1e9:.3f}', b[1:])
