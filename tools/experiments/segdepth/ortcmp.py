import sys, json, numpy as np, onnxruntime as ort, os
d = sys.argv[1]; meta = json.load(open(f'{d}/meta.json'))
s = ort.InferenceSession(f'{d}/model.onnx', providers=['CPUExecutionProvider'])
x = np.fromfile(f'{d}/input.bin', np.float32).reshape(meta['input']['shape'])
for i, o in enumerate(s.run(None, {s.get_inputs()[0].name: x})):
    e = np.fromfile(f'{d}/output_{i}.bin', np.float32)
    print(i, o.shape, 'max rel err vs reference', float(np.max(np.abs(o.ravel()-e)/(1+np.abs(e)))))
