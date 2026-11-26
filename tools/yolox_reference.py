"""Run an official YOLOX ONNX model on a fixed input and dump both, so the
Meganeura graph can be checked against onnxruntime.

    python tools/yolox_reference.py ref/yolox_nano.onnx ref/yolox_nano_reference [image]

Writes `input.bin` (the Focus-sliced input RedEye's graph takes,
`[12, H/2, W/2]` f32), `output.bin` (`[anchors, 85]` f32, exactly as the
ONNX model emits it), and `meta.json`. Without an image the input is a
deterministic pseudo-random picture in the model's 0..255 BGR range.
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort


def letterbox(path: str, height: int, width: int) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    r = min(height / image.height, width / image.width)
    size = (int(image.width * r), int(image.height * r))
    resized = np.asarray(image.resize(size, Image.BILINEAR), dtype=np.float32)
    canvas = np.full((height, width, 3), 114.0, dtype=np.float32)
    canvas[: size[1], : size[0]] = resized[..., ::-1]  # RGB -> BGR
    return canvas.transpose(2, 0, 1)


def focus(x: np.ndarray) -> np.ndarray:
    # Same order as YOLOX's Focus: top-left, bottom-left, top-right, bottom-right.
    return np.concatenate(
        [x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]], axis=0
    )


def main(model_path: str, out_dir: str, image: str | None) -> None:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _, _, height, width = session.get_inputs()[0].shape
    if image:
        x = letterbox(image, height, width)
    else:
        rng = np.random.default_rng(1984)
        x = rng.uniform(0.0, 255.0, size=(3, height, width)).astype(np.float32)
    (output,) = session.run(None, {session.get_inputs()[0].name: x[None]})
    os.makedirs(out_dir, exist_ok=True)
    focus(x).astype(np.float32).tofile(os.path.join(out_dir, "input.bin"))
    output[0].astype(np.float32).tofile(os.path.join(out_dir, "output.bin"))
    meta = {
        "model": os.path.basename(model_path),
        "input_hw": [height, width],
        "output_shape": list(output[0].shape),
        "image": os.path.basename(image) if image else None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote reference to {out_dir}: {meta}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
