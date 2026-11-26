"""Prepare an ONNX export for Meganeura's importer and dump an onnxruntime
reference to check it against.

    python tools/onnx_reference.py model.onnx out_dir 1,3,640,640 [image] [--scale 255]

Meganeura's importer needs static shapes. Exports from Transformers and
similar keep the batch and image sizes symbolic and compute positional
tables, masks and reshapes at run time, so the model is first given a fixed
input shape and constant-folded with onnx-simplifier. The folded model is
written to `out_dir/model.onnx`, alongside:

- `input.bin`: the input, f32, in the model's own layout. A deterministic
  pseudo-random tensor by default; with an image, the image resized to the
  input's height and width, RGB, divided by `--scale`.
- `output_N.bin`: each output from onnxruntime, f32, in graph order.
- `meta.json`: shapes and names.

`--start-at TENSOR` cuts the graph so that an intermediate tensor becomes the
input, for layers RedEye's own preprocessing already performs (YOLOX's
Focus slicing, for one). `first:OP` names the output of the first node of
type OP, since exported tensor names are often anonymous numbers. The reference input is then that tensor's value
when the full model runs on the input above.

`cargo run --release --example onnx_check -- out_dir` then compares.
"""

import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out_dir")
    parser.add_argument("shape", help="static input shape, e.g. 1,3,640,640")
    parser.add_argument("image", nargs="?")
    parser.add_argument("--scale", type=float, default=255.0)
    parser.add_argument("--start-at", help="intermediate tensor to make the input")
    args = parser.parse_args()
    shape = [int(x) for x in args.shape.split(",")]

    model = onnx.load(args.model)
    (graph_input,) = [i for i in model.graph.input if i.name not in {t.name for t in model.graph.initializer}]
    folded, ok = onnxsim.simplify(model, overwrite_input_shapes={graph_input.name: shape})
    if not ok:
        raise SystemExit("onnx-simplifier could not validate the folded model")
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.onnx")
    onnx.save(folded, model_path)

    if args.image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB").resize((shape[3], shape[2]), Image.BILINEAR)
        x = (np.asarray(image, dtype=np.float32) / args.scale).transpose(2, 0, 1)[None]
    else:
        x = np.random.default_rng(1984).uniform(0.0, 1.0, size=shape).astype(np.float32)

    if args.start_at and args.start_at.startswith("first:"):
        op = args.start_at.split(":", 1)[1]
        args.start_at = next(n.output[0] for n in folded.graph.node if n.op_type == op)
    if args.start_at:
        # Run the full model once to capture the cut tensor's value.
        probe = onnx.load(model_path)
        probe.graph.output.append(onnx.helper.make_empty_tensor_value_info(args.start_at))
        probe_session = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
        x = probe_session.run([args.start_at], {graph_input.name: x})[0]
        outputs = [o.name for o in folded.graph.output]
        onnx.utils.extract_model(model_path, model_path, [args.start_at], outputs)
        cut = onnx.load(model_path)
        cut.graph.input[0].type.tensor_type.shape.Clear()
        for dim in x.shape:
            cut.graph.input[0].type.tensor_type.shape.dim.add().dim_value = dim
        onnx.save(cut, model_path)
        graph_input = cut.graph.input[0]
        shape = list(x.shape)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {graph_input.name: x})
    x.astype(np.float32).tofile(os.path.join(args.out_dir, "input.bin"))
    for index, output in enumerate(outputs):
        output.astype(np.float32).tofile(os.path.join(args.out_dir, f"output_{index}.bin"))
    meta = {
        "source": os.path.basename(args.model),
        "input": {"name": graph_input.name, "shape": shape},
        "outputs": [
            {"name": o.name, "shape": list(r.shape)} for o, r in zip(session.get_outputs(), outputs)
        ],
        "image": os.path.basename(args.image) if args.image else None,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
