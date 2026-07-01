"""Load Sapiens2 pose model ONCE at import, bf16 + torch.compile (warm worker).

Portable across RunPod / Vast: the 20 GB checkpoint is NOT baked into the image.
On first init we ensure it exists under $WEIGHTS_DIR (download from HF — ungated),
which each platform points at its own persistent storage:
  RunPod -> WEIGHTS_DIR=/runpod-volume   (network volume, persists across cold starts)
  Vast   -> WEIGHTS_DIR=/workspace       (instance disk)
The Inductor compile-cache also lives under $WEIGHTS_DIR so the ~2 min compile is
paid once, not on every cold start.
"""
from __future__ import annotations

import glob
import os

import torch

MODEL_SIZE = os.environ.get("MODEL_SIZE", "5b")           # 0.4b/0.8b/1b/5b
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
HF_REPO = f"facebook/sapiens2-pose-{MODEL_SIZE}"
CKPT_NAME = f"sapiens2_{MODEL_SIZE}_pose.safetensors"

# Persist Inductor's compiled kernels on the shared volume across cold starts.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(WEIGHTS_DIR, "inductor_cache"))

_COMPILE = os.environ.get("COMPILE", "1") == "1"


def _find_config() -> str:
    """The keypoints308 config .py ships inside the installed sapiens2 package."""
    import sapiens  # noqa: F401  (installed via `pip install -e` of the sapiens2 repo)

    root = os.path.dirname(os.path.dirname(os.path.abspath(sapiens.__file__)))
    pattern = os.path.join(
        root, "sapiens", "pose", "configs", "keypoints308", "*",
        f"sapiens2_{MODEL_SIZE}_keypoints308_*-1024x768.py",
    )
    hits = glob.glob(pattern)
    if not hits:
        raise FileNotFoundError(f"no sapiens2 config matching {pattern}")
    return hits[0]


def _ensure_checkpoint() -> str:
    path = os.path.join(WEIGHTS_DIR, CKPT_NAME)
    if os.path.isfile(path):
        return path
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download

    # ungated download; token optional (HF_TOKEN honored if set)
    src = hf_hub_download(
        repo_id=HF_REPO, filename=CKPT_NAME, local_dir=WEIGHTS_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    return src


def _load():
    from sapiens.pose.models import init_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _find_config()
    ckpt = _ensure_checkpoint()
    model = init_model(cfg, ckpt, device=device)
    model.eval()
    model = model.to(torch.bfloat16)

    fwd = model
    if _COMPILE and device == "cuda":
        fwd = torch.compile(model)  # default mode (max-autotune gave no gain on 5B)
        # warm up + populate the on-disk inductor cache
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            dummy = torch.randn(1, 3, 1024, 768, device=device, dtype=torch.bfloat16)
            fwd(dummy)
        torch.cuda.synchronize()

    return model, fwd, device


# MODEL keeps .pipeline/.data_preprocessor/.codec/.cfg/.pose_metainfo (eager wrapper);
# FORWARD is the compiled callable used for the heavy forward pass.
MODEL, FORWARD, DEVICE = _load()
