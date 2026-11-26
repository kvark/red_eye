#!/bin/sh
# Fetch D-FINE-N (Apache-2.0; the Transformers export on the Hugging Face
# Hub) and prepare it for Meganeura's importer at the input size RedEye
# runs, with an onnxruntime reference on the YOLOX demo photo.
#
#   pip install -r tools/requirements.txt
#   tools/fetch_dfine.sh            # 640x480, the cameras' 4:3
#   tools/fetch_dfine.sh 640 640
#
# Writes ref/dfine_nano_WxH/{model.onnx, input.bin, output_*.bin, meta.json}.
# model.onnx is what the app loads. PYTHON selects the interpreter.
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
WIDTH="${1:-640}"
HEIGHT="${2:-480}"
REPO=https://huggingface.co/onnx-community/dfine-nano-coco-ONNX/resolve/main
SHA=f7febe9695cd6f71120abbeb0642d3d6b6c6bcc36f0ba1d79b66d302b5860814
ASSETS=https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets
mkdir -p ref

# The Hub rate-limits anonymous downloads; back off instead of failing.
fetch() {
    curl -fsSL --retry 6 --retry-delay 15 --retry-all-errors -o "$1" "$2"
}
[ -f ref/dog.jpg ] || fetch ref/dog.jpg "$ASSETS/dog.jpg"
[ -f ref/dfine_nano.onnx ] || fetch ref/dfine_nano.onnx "$REPO/onnx/model.onnx"
echo "$SHA  ref/dfine_nano.onnx" | sha256sum -c -

# The export computes positional tables and its box-weighting table from
# scalar parameters at run time; folding them here leaves the importer
# only real work.
"$PYTHON" -I tools/onnx_reference.py ref/dfine_nano.onnx \
    "ref/dfine_nano_${WIDTH}x${HEIGHT}" "1,3,$HEIGHT,$WIDTH" ref/dog.jpg --simplify
