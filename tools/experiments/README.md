# Model evaluation scripts

Throwaway tooling from evaluating candidate models through Meganeura's ONNX
importer. These scripts are not part of the app or of CI. They are kept so
the evaluations can be rerun on another machine and extended. Results and
conclusions live in `docs/technology.md` and `TASKS.md`. Run everything from
the repository root, with the venv from `tools/requirements.txt` plus the
extras each part lists. Downloads land in `ref/` (gitignored). Hugging Face
rate-limits anonymous downloads, so the fetchers retry with backoff.

The scripts encode Meganeura's op coverage at `b25f763` by hand
(`gaps.py`, `inspect_onnx.py`), so update those lists as the importer grows.

## `moonshine/`: speech for voice commands (TASKS.md item 3)

Extras: `pip install tokenizers scipy`, and `apt install espeak-ng` (test
audio only; nothing ships). Model: `onnx-community/moonshine-tiny-ONNX` (MIT),
downloaded to `ref/moonshine_tiny/` (`onnx/encoder_model.onnx`,
`onnx/decoder_model.onnx`, `tokenizer.json`, `generation_config.json`).

| Script | Does |
|---|---|
| `make_audio.py OUT` | espeak-ng command WAVs, 16 kHz mono, padded to 4 s |
| `reference.py MODEL_DIR AUDIO_DIR [pad\|trim]` | encoder plus free greedy decoding in onnxruntime; prints transcripts |
| `constrained.py MODEL_DIR coco.txt` | free and grammar-constrained decoding (token trie of "verb article label") over 9 espeak voices |
| `sweep.py MODEL_DIR` | accuracy across voices, rates and window lengths |
| `patch_enc.py SRC OUT WAV` | rewrites `InstanceNormalization` into ops the importer takes, with an onnxruntime reference |
| `prep_dec.py MODEL OUT [--simplify] [--logits-only]` | fixes the decoder's input shapes |
| `cut_dec.py SRC OUT` | cuts the decoder after its token-embedding `Gather` (the importer has no integer inputs) |
| `bake_dec.py CUT OUT ENCODER WAV` | bakes real encoder states in, so `onnx_check` can compare the decoder |
| `gaps.py MODEL` | static pass mirroring the importer: every unsupported op or attribute, with counts |
| `macs.py MODEL` | MACs by role (projections, attention, vocab head) |
| `inspect.py`, `context.py` | inputs and outputs, op histogram; neighbourhood of an op |

## `segdepth/`: segmentation and depth for the wireframe (TASKS.md items 4–6)

Models are fetched with `fetch.sh REPO FILE...` into `ref/segdepth/dl/`.
The evaluated ones are:

- `onnx-community/depth-anything-v2-small` `onnx/model.onnx` (Apache-2.0)
- `onnx-community/depth-anything-v3-small` (Apache-2.0)
- `VirnectUnityTeam/RF-DETR-Seg-Nano` `rf-detr-seg-nano.onnx`: a
  third-party export of Roboflow's Apache-2.0 model, with its output names
  swapped. Re-export from `Roboflow/rf-detr-seg-nano` before relying on it.
- `onnx-community/EdgeTAM-ONNX` `onnx/vision_encoder.onnx` and
  `onnx/prompt_encoder_mask_decoder.onnx` (Apache-2.0)

| Script | Does |
|---|---|
| `ref.py MODEL OUT --shape ...` | like `tools/onnx_reference.py`, adding ImageNet normalisation, inputs baked in as constants (`--bake name=x.npy`), output selection, and `outputs.npz` |
| `probe.py SRC DST --ops ...` | rewrites unsupported ops into supported ones, to find later gaps and check numerics. A diagnostic, not a fix |
| `inspect_onnx.py MODEL` | inputs and outputs, every importer gap with counts, and GMACs |
| `macs_by.py MODEL [DEPTH]` | MACs per module |
| `rfdetr_decode.py DIR` | decodes RF-DETR-Seg outputs on `ref/dog.jpg` into `masks.png`. Needs `ref/segdepth/dl/rfdetr_config.json`, the model's `id2label` |
| `depthviz.py NPZ KEY OUT` | a depth map beside the photo, with mean depth per region |
| `ortcmp.py DIR` | re-checks a reference directory against onnxruntime |
| `io.py`, `ctx.py` | inputs and outputs, op histogram; producers of a named node |

Then check through Meganeura with
`cargo run --release --example onnx_check -- <dir>`.
