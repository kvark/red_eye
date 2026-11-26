"""Convert an official YOLOX ONNX export to the safetensors RedEye loads.

    python tools/yolox_to_safetensors.py ref/yolox_nano.onnx ref/yolox_nano.safetensors

The official exports (github.com/Megvii-BaseDetection/YOLOX releases,
Apache-2.0) have BatchNorm folded into every convolution and anonymous
initializer names. The only stable identity a convolution has is its
position in execution order, which `torch.onnx` takes from the module's
forward pass, so each one is stored as `conv.NNN.weight` / `conv.NNN.bias`
by that position. `src/perception/yolox.rs` creates its convolutions in
the same forward order and checks every shape on load.
"""

import hashlib
import json
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    tensors = {}
    layout = []
    convs = [n for n in model.graph.node if n.op_type == "Conv"]
    for index, node in enumerate(convs):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        weight = inits[node.input[1]].astype(np.float32)
        bias = inits[node.input[2]].astype(np.float32)
        tensors[f"conv.{index:03}.weight"] = np.ascontiguousarray(weight)
        tensors[f"conv.{index:03}.bias"] = np.ascontiguousarray(bias)
        layout.append(
            {
                "shape": list(weight.shape),
                "group": attrs.get("group", 1),
                "stride": attrs["strides"][0],
                "pad": attrs["pads"][0],
            }
        )
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    metadata = {
        "source": src.rsplit("/", 1)[-1],
        "source_sha256": digest,
        "convs": str(len(convs)),
        "layout": json.dumps(layout),
        "license": "Apache-2.0 (Megvii YOLOX)",
    }
    save_file(tensors, dst, metadata=metadata)
    print(f"wrote {len(convs)} convolutions to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
