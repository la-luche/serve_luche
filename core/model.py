"""Load the Sapiens pose TorchScript model ONCE, at import time.

Loading at import (not lazily inside the request handler) is what lets RunPod
FlashBoot snapshot a *warm* worker — otherwise every cold start re-pays the
~4 GB weight load. See README "Cold start".

The weights (`*.pt2`) are baked into the image under ./weights/ (or point
WEIGHTS_PATH at a specific file). We support both the fp32 `-torchscript`
checkpoint (run under bf16 autocast) and the `-bfloat16` checkpoint (params
already bf16). Which one is loaded is auto-detected from the params' dtype.
"""
from __future__ import annotations

import glob
import os

import torch

# Sapiens pose input is H=1024, W=768 (see the goliath 1024x768 config).
MODEL_INPUT_H = 1024
MODEL_INPUT_W = 768

# ImageNet-style normalization on the 0-255 pixel scale (Sapiens pose default).
# NOTE: order here is R,G,B. If INPUT_BGR is flipped in postproc, the mean/std
# are reordered there to match.
NORM_MEAN = (123.675, 116.28, 103.53)
NORM_STD = (58.395, 57.12, 57.375)

_WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")


def _find_weights() -> str:
    explicit = os.environ.get("WEIGHTS_PATH")
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"WEIGHTS_PATH={explicit} does not exist")
        return explicit
    hits = sorted(glob.glob(os.path.join(_WEIGHTS_DIR, "*.pt2")))
    if not hits:
        raise FileNotFoundError(
            f"No *.pt2 weights found in {_WEIGHTS_DIR}. Run ./download_weights.sh "
            "or set WEIGHTS_PATH."
        )
    # Prefer a 2b checkpoint if several are present.
    for h in hits:
        if "2b" in os.path.basename(h).lower():
            return h
    return hits[0]


def _load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = _find_weights()
    model = torch.jit.load(path, map_location=device).eval().to(device)
    params = list(model.parameters())
    dtype = params[0].dtype if params else torch.float32

    # Warm up so cuDNN autotune + FlashBoot snapshot capture the real kernels.
    if device == "cuda":
        run_dtype = dtype if dtype != torch.float32 else torch.float32
        dummy = torch.zeros(
            1, 3, MODEL_INPUT_H, MODEL_INPUT_W, device=device, dtype=run_dtype
        )
        autocast = dtype == torch.float32
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=autocast
        ):
            model(dummy)
        torch.cuda.synchronize()

    return model, device, dtype, path


MODEL, DEVICE, MODEL_DTYPE, WEIGHTS_FILE = _load()
# True when we should wrap the forward pass in bf16 autocast (fp32 model on GPU).
USE_AUTOCAST = DEVICE == "cuda" and MODEL_DTYPE == torch.float32
