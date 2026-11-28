#!/bin/sh
# Fetch Moonshine-tiny (MIT; the Transformers export on the Hugging Face Hub)
# and prepare it for Meganeura's importer: a 2 s encoder, a 12-position
# decoder, onnxruntime references, and the command grammar as token ids.
#
#   pip install -r tools/requirements.txt
#   tools/fetch_moonshine.sh [SPEECH.wav]
#
# Writes ref/moonshine_tiny_2s/ (see tools/prepare_moonshine.py). With a
# recording of "track the bicycle" (any rate, 16-bit mono; espeak-ng makes
# one), it also writes the end-to-end test's input. Push the directory's
# commands.txt, encoder/model.onnx and decoder/model.onnx to
# /data/local/tmp/red_eye/moonshine/ for the app. PYTHON selects the
# interpreter.
set -eu
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
REPO=https://huggingface.co/onnx-community/moonshine-tiny-ONNX/resolve/main
DIR=ref/moonshine_tiny
mkdir -p "$DIR/onnx"

fetch() {
    [ -f "$DIR/$1" ] || curl -fsSL --retry 6 --retry-delay 15 --retry-all-errors -o "$DIR/$1" "$REPO/$1"
}
fetch onnx/encoder_model.onnx
fetch onnx/decoder_model.onnx
fetch tokenizer.json
(cd "$DIR" && sha256sum -c -) <<SUMS
cbbf580f703b2af2137e0f6d14cd87f31cc67bd858bfd8715403a9489982d1a5  onnx/encoder_model.onnx
19dfea5cba225eb38c0dc7b6e6b9464da6cb46266eab7dbcc8f3d5f9a5f5d79e  onnx/decoder_model.onnx
7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f  tokenizer.json
SUMS

if [ $# -ge 1 ]; then
    "$PYTHON" -I tools/prepare_moonshine.py "$DIR" ref/moonshine_tiny_2s --speech "$1"
else
    "$PYTHON" -I tools/prepare_moonshine.py "$DIR" ref/moonshine_tiny_2s
fi
