#!/bin/sh
# Fetch Depth Anything V2 Small (Apache-2.0; the Transformers export on the
# Hugging Face Hub) and prepare it for Meganeura's importer at the input size
# RedEye runs, with an onnxruntime reference on the YOLOX demo photo.
#
#   pip install -r tools/requirements.txt
#   tools/fetch_depth.sh            # 336x252, the cameras' 4:3 in 14-pixel patches
#   tools/fetch_depth.sh 266 266
#
# Writes ref/depth_small_WxH/{model.onnx, input.bin, output_0.bin, meta.json}.
# model.onnx is what the app loads: it takes RGB in 0..1, like D-FINE, with
# the ImageNet normalisation folded into its patch embedding. Its output is
# relative inverse depth at the input's resolution. PYTHON selects the
# interpreter.
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
WIDTH="${1:-336}"
HEIGHT="${2:-252}"
REPO=https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main
SHA=afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c
ASSETS=https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets
mkdir -p ref/depth

fetch() {
    curl -fsSL --retry 6 --retry-delay 15 --retry-all-errors -o "$1" "$2"
}
[ -f ref/dog.jpg ] || fetch ref/dog.jpg "$ASSETS/dog.jpg"
[ -f ref/depth/da2_small.onnx ] || fetch ref/depth/da2_small.onnx "$REPO/onnx/model.onnx"
echo "$SHA  ref/depth/da2_small.onnx" | sha256sum -c -

"$PYTHON" -I tools/fold_normalization.py ref/depth/da2_small.onnx \
    ref/depth/da2_small_folded.onnx 0.485,0.456,0.406 0.229,0.224,0.225
"$PYTHON" -I tools/onnx_reference.py ref/depth/da2_small_folded.onnx \
    "ref/depth_small_${WIDTH}x${HEIGHT}" "1,3,$HEIGHT,$WIDTH" ref/dog.jpg --simplify
