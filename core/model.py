"""Load Sapiens2 pose model ONCE at import, bf16 + torch.compile (warm worker).

Portable across RunPod / Vast: the 20 GB checkpoint is NOT baked into the image.
On RunPod, prefer its cached-model mount so the checkpoint is present before the
worker starts and model download time is not billed. Otherwise ensure it exists
under $WEIGHTS_DIR (download from HF — ungated):
  RunPod cached model -> /runpod-volume/huggingface-cache/hub/...
  RunPod/Vast fallback -> WEIGHTS_DIR (persistent storage when configured)
The Inductor compile-cache also lives under $WEIGHTS_DIR so the ~2 min compile is
paid once, not on every cold start.
"""
from __future__ import annotations

import glob
import os
import time

import torch

from .log import log

MODEL_SIZE = os.environ.get("MODEL_SIZE", "5b")           # 0.4b/0.8b/1b/5b
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
HF_REPO = f"facebook/sapiens2-pose-{MODEL_SIZE}"
CKPT_NAME = f"sapiens2_{MODEL_SIZE}_pose.safetensors"
RUNPOD_HF_CACHE = os.environ.get(
    "RUNPOD_HF_CACHE", "/runpod-volume/huggingface-cache/hub"
)

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


def _runpod_cached_checkpoint() -> str | None:
    """Resolve a RunPod cached-model snapshot without copying the checkpoint.

    RunPod follows Hugging Face's cache layout. Prefer the revision referenced by
    ``refs/main`` and fall back to any complete snapshot, which also handles a
    cache prepared without that ref file.
    """
    repo_dir = os.path.join(
        RUNPOD_HF_CACHE, f"models--{HF_REPO.replace('/', '--')}"
    )
    ref_path = os.path.join(repo_dir, "refs", "main")
    try:
        with open(ref_path, encoding="utf-8") as f:
            revision = f.read().strip()
        candidate = os.path.join(repo_dir, "snapshots", revision, CKPT_NAME)
        if revision and os.path.isfile(candidate):
            return candidate
    except OSError:
        pass

    snapshots = glob.glob(os.path.join(repo_dir, "snapshots", "*", CKPT_NAME))
    complete = [path for path in snapshots if os.path.isfile(path)]
    return max(complete, key=os.path.getmtime) if complete else None


def _ensure_checkpoint() -> str:
    cached = _runpod_cached_checkpoint()
    if cached:
        gb = os.path.getsize(cached) / 1e9
        log(f"RunPod cached checkpoint present ({gb:.1f} GB) at {cached}")
        return cached

    path = os.path.join(WEIGHTS_DIR, CKPT_NAME)
    if os.path.isfile(path):
        gb = os.path.getsize(path) / 1e9
        log(f"checkpoint present ({gb:.1f} GB) at {path} — no download")
        return path
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download

    log(f"downloading {CKPT_NAME} from {HF_REPO} -> {WEIGHTS_DIR} (20 GB for 5b)...")
    t = time.time()
    src = hf_hub_download(
        repo_id=HF_REPO, filename=CKPT_NAME, local_dir=WEIGHTS_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    log(f"checkpoint downloaded in {time.time() - t:.1f}s")
    return src


def _load():
    from sapiens.pose.models import init_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading model: size={MODEL_SIZE} device={device} compile={_COMPILE} "
        f"weights_dir={WEIGHTS_DIR}")
    cfg = _find_config()
    ckpt = _ensure_checkpoint()

    t = time.time()
    model = init_model(cfg, ckpt, device=device)
    model.eval()
    log(f"init_model done in {time.time() - t:.1f}s")

    # init_model gives .pipeline/.data_preprocessor but NOT .codec — build the
    # UDPHeatmap decoder from the config (as sapiens2's vis_pose demo does).
    from sapiens.pose.datasets import UDPHeatmap

    codec_cfg = dict(model.cfg.codec)
    codec_cfg.pop("type", None)
    model.codec = UDPHeatmap(**codec_cfg)
    model = model.to(torch.bfloat16)
    log("model -> bf16, codec attached")

    fwd = model
    if _COMPILE and device == "cuda":
        log("torch.compile enabled — compiling on first forward (can take minutes)")
        fwd = torch.compile(model)
    else:
        log("torch.compile DISABLED — running eager")

    return model, fwd, device


# MODEL keeps .pipeline/.data_preprocessor/.codec/.cfg/.pose_metainfo (eager wrapper);
# FORWARD is the compiled callable used for the heavy forward pass.
MODEL, FORWARD, DEVICE = _load()
