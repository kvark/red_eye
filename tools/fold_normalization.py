"""Fold per-channel input normalisation into a model's first convolution.

    python tools/fold_normalization.py in.onnx out.onnx MEAN,MEAN,MEAN STD,STD,STD

Models trained on ImageNet statistics take `(x - mean) / std`, which
RedEye's preprocessing does not compute: it produces `x` in 0..1, as
D-FINE takes. When the input feeds one convolution without padding, the
normalisation is exactly a change of that convolution's weights and bias:

    W' = W / std[c]
    b' = b - sum over c, kh, kw of W[o, c, kh, kw] * mean[c] / std[c]

so the folded model takes `x` directly. (With padding, the zeros would turn
into `-mean / std` at the border, and the fold would not be exact.)
"""

import sys

import numpy as np
import onnx
from onnx import numpy_helper


def main() -> None:
    src, dst, mean, std = sys.argv[1:5]
    mean = np.array([float(v) for v in mean.split(",")], dtype=np.float64)
    std = np.array([float(v) for v in std.split(",")], dtype=np.float64)
    model = onnx.load(src)
    graph = model.graph
    name = graph.input[0].name
    convs = [n for n in graph.node if n.op_type == "Conv" and n.input[0] == name]
    others = [n for n in graph.node if name in n.input and n.op_type not in ("Conv", "Shape")]
    if len(convs) != 1 or others:
        sys.exit(f"{src}: the input must feed one Conv (and Shapes), not {convs + others}")
    conv = convs[0]
    pads = next((a.ints for a in conv.attribute if a.name == "pads"), [])
    if any(pads):
        sys.exit(f"{src}: the first Conv pads its input, so the fold would not be exact")
    inits = {i.name: i for i in graph.initializer}
    w = numpy_helper.to_array(inits[conv.input[1]]).astype(np.float64)
    if w.shape[1] != len(mean):
        sys.exit(f"{src}: {w.shape[1]} input channels, {len(mean)} means")
    b = (
        numpy_helper.to_array(inits[conv.input[2]]).astype(np.float64)
        if len(conv.input) > 2
        else np.zeros(w.shape[0])
    )
    scaled = w / std[None, :, None, None]
    b = b - (scaled * mean[None, :, None, None]).sum(axis=(1, 2, 3))
    inits[conv.input[1]].CopyFrom(numpy_helper.from_array(scaled.astype(np.float32), conv.input[1]))
    if len(conv.input) > 2:
        inits[conv.input[2]].CopyFrom(numpy_helper.from_array(b.astype(np.float32), conv.input[2]))
    else:
        bias = conv.output[0] + "_folded_bias"
        graph.initializer.append(numpy_helper.from_array(b.astype(np.float32), bias))
        conv.input.append(bias)
    onnx.save(model, dst)
    print(f"folded mean {mean.tolist()} std {std.tolist()} into {conv.name}: {dst}")


if __name__ == "__main__":
    main()
