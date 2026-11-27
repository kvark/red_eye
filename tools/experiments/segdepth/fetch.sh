#!/bin/bash
# fetch REPO FILE...
set -e
base=ref/segdepth/dl
repo=$1; shift
for f in "$@"; do
  out="$base/$repo/$f"
  mkdir -p "$(dirname "$out")"
  if [ ! -s "$out" ]; then
    curl -fsSL --retry 6 --retry-delay 15 --retry-all-errors -o "$out.part" "https://huggingface.co/$repo/resolve/main/$f" && mv "$out.part" "$out"
  fi
  echo "$(sha256sum "$out")"
done
