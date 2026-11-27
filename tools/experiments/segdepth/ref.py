"""onnx_reference.py with ImageNet normalisation, extra inputs baked in as
constants, and the first input taken from an image or a .npy file.

    python -I ref.py model.onnx out_dir --shape 1,3,H,W [--image img] [--norm imagenet]
        [--input-npy x.npy] [--bake name=file.npy ...] [--simplify] [--keep-outputs a,b]

Writes out_dir/model.onnx (single input), input.bin, output_N.bin, meta.json
and outputs.npz (the reference outputs, for inspection).
"""

import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
from onnx import numpy_helper

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("out_dir")
    p.add_argument("--shape", required=True)
    p.add_argument("--image")
    p.add_argument("--norm", choices=["imagenet", "unit", "none"], default="imagenet")
    p.add_argument("--input-npy")
    p.add_argument("--bake", action="append", default=[])
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--keep-outputs")
    p.add_argument("--resample", default="bilinear")
    args = p.parse_args()
    shape = [int(x) for x in args.shape.split(",")]

    model = onnx.load(args.model)
    g = model.graph
    inits = {i.name for i in g.initializer}
    real_inputs = [i for i in g.input if i.name not in inits]
    baked = {}
    for b in args.bake:
        name, path = b.split("=")
        baked[name] = np.load(path)
    # Replace baked inputs with initializers.
    for i in list(g.input):
        if i.name in baked:
            arr = baked[i.name]
            g.input.remove(i)
            g.initializer.append(numpy_helper.from_array(arr, i.name))
    (first,) = [i for i in g.input if i.name not in {t.name for t in g.initializer}]
    if args.keep_outputs:
        keep = args.keep_outputs.split(",")
        outs = [o for o in g.output if o.name in keep]
        del g.output[:]
        g.output.extend(outs)

    if args.simplify:
        model, ok = onnxsim.simplify(
            model,
            overwrite_input_shapes={first.name: shape},
            skipped_optimizers=["fuse_pad_into_conv"],
        )
        if not ok:
            raise SystemExit("onnxsim could not validate")
    else:
        d = first.type.tensor_type.shape
        d.ClearField("dim")
        for v in shape:
            d.dim.add().dim_value = v
        # Clear stale value_info / output dims that came from symbolic shapes.
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "model.onnx")
    size = sum(t.ByteSize() for t in model.graph.initializer)
    if size > 1.8e9:
        onnx.save(model, path, save_as_external_data=True, location="model.onnx_data")
    else:
        onnx.save(model, path)

    if args.input_npy:
        x = np.load(args.input_npy).astype(np.float32).reshape(shape)
    elif args.image:
        from PIL import Image

        rs = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[args.resample]
        img = Image.open(args.image).convert("RGB").resize((shape[-1], shape[-2]), rs)
        x = np.asarray(img, np.float32) / 255.0
        if args.norm == "imagenet":
            x = (x - MEAN) / STD
        x = x.transpose(2, 0, 1).reshape(shape).astype(np.float32)
    else:
        x = np.random.default_rng(1984).uniform(0, 1, size=shape).astype(np.float32)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    outs = sess.run(None, {in_name: x})
    x.tofile(os.path.join(args.out_dir, "input.bin"))
    for k, o in enumerate(outs):
        o.astype(np.float32).tofile(os.path.join(args.out_dir, f"output_{k}.bin"))
    np.savez(os.path.join(args.out_dir, "outputs.npz"), input=x, **{o.name: v for o, v in zip(sess.get_outputs(), outs)})
    meta = {
        "source": args.model,
        "input": {"name": in_name, "shape": shape},
        "baked": {k: list(v.shape) for k, v in baked.items()},
        "outputs": [{"name": o.name, "shape": list(v.shape)} for o, v in zip(sess.get_outputs(), outs)],
        "image": args.image,
        "norm": args.norm,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
