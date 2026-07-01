#!/usr/bin/env bash
# Download the Sapiens pose TorchScript weights into ./weights/ so they can be
# baked into the image (COPY weights/ in the Dockerfile).
#
# Sapiens is a GATED model on Hugging Face — you must accept the license once at
# https://huggingface.co/facebook/sapiens-pose-1b-torchscript and provide a token.
#
#   export HF_TOKEN=hf_xxx
#   ./download_weights.sh                 # 1b (default)
#   MODEL_SIZE=1b ./download_weights.sh   # smaller, faster for local dev
set -euo pipefail

# NOTE: there is NO 2b pose checkpoint. Original Sapiens pose tops out at 1b
# (torchscript, ungated). 0.6b/0.3b are smaller. (Sapiens2 has a 5b pose model
# but only as safetensors — needs custom model code, not torch.jit.load.)
MODEL_SIZE="${MODEL_SIZE:-1b}"
REPO="facebook/sapiens-pose-${MODEL_SIZE}-torchscript"
DEST="$(dirname "$0")/weights"

mkdir -p "$DEST"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: set HF_TOKEN (accept the license first at https://huggingface.co/${REPO})" >&2
  exit 1
fi

# hf download pulls just the *.pt2 (skip readme/images).
huggingface-cli download "$REPO" \
  --include "*.pt2" \
  --local-dir "$DEST" \
  --local-dir-use-symlinks False \
  --token "$HF_TOKEN"

echo "Downloaded into $DEST:"
ls -lh "$DEST"/*.pt2
