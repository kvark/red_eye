#!/bin/sh
# Fetch the official YOLOX exports (Megvii, Apache-2.0) and prepare what the
# app and tests use: converted safetensors, onnxruntime references, and a
# cut export for the ONNX importer check (examples/onnx_check.rs).
#
#   pip install -r tools/requirements.txt
#   tools/fetch_yolox.sh            # nano and tiny
#   tools/fetch_yolox.sh nano       # just nano
#
# PYTHON selects the interpreter (default python3). Everything lands in
# ref/, which is not committed.
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
RELEASE=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0
ASSETS=https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/main/assets
mkdir -p ref

if [ ! -f ref/dog.jpg ]; then
    curl -fsSL -o ref/dog.jpg "$ASSETS/dog.jpg"
fi

for model in ${*:-nano tiny}; do
    case "$model" in
        nano) sha=c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d ;;
        tiny) sha=427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7 ;;
        *) echo "unknown model $model (nano, tiny)" >&2; exit 2 ;;
    esac
    onnx="ref/yolox_$model.onnx"
    if [ ! -f "$onnx" ]; then
        curl -fsSL -o "$onnx" "$RELEASE/yolox_$model.onnx"
    fi
    echo "$sha  $onnx" | sha256sum -c -
    "$PYTHON" -I tools/yolox_to_safetensors.py "$onnx" "ref/yolox_$model.safetensors"
    "$PYTHON" -I tools/yolox_reference.py "$onnx" "ref/yolox_${model}_reference"
    "$PYTHON" -I tools/yolox_reference.py "$onnx" "ref/yolox_${model}_dog" ref/dog.jpg
    # The official export, whole, for Meganeura's ONNX importer.
    "$PYTHON" -I tools/onnx_reference.py "$onnx" "ref/yolox_${model}_onnx" \
        1,3,416,416 ref/dog.jpg --scale 1
done
