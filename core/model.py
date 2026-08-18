"""Load Sapiens2 pose model once and optionally warm its first forward.

Production images bake a bf16 copy of the checkpoint (~10 GB for 5b) at
``WEIGHTS_DIR`` — see build-weights.yml — so nothing downloads at runtime.
Fallbacks, in order: the legacy RunPod cached-model mount, then a direct
Hugging Face download (dedicated workers should persist ``WEIGHTS_DIR``).
RunPod Serverless defaults to eager inference: compiling improves throughput but
adds a large, host-dependent first-forward delay. Dedicated workers can opt back
in with ``COMPILE=1`` and keep the Inductor cache on persistent storage.
"""
from __future__ import annotations

import glob
import gc
import os
import time

import torch

from .log import log

MODEL_SIZE = os.environ.get("MODEL_SIZE", "5b")           # 0.4b/0.8b/1b/5b
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
HF_REPO = f"facebook/sapiens2-pose-{MODEL_SIZE}"
CKPT_NAME = f"sapiens2_{MODEL_SIZE}_pose.safetensors"
SAPIENS_MODEL_REVISION = os.environ.get(
    "SAPIENS_MODEL_REVISION", "ada1f29aa1fd454ca28665c700923a0101b6b24f"
)
RUNPOD_HF_CACHE = os.environ.get(
    "RUNPOD_HF_CACHE", "/runpod-volume/huggingface-cache/hub"
)

# Persist Inductor's compiled kernels on the shared volume across cold starts.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(WEIGHTS_DIR, "inductor_cache"))

_COMPILE = os.environ.get("COMPILE", "0") == "1"
_MODEL_LOAD_DEVICE = os.environ.get("MODEL_LOAD_DEVICE")


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
    """Resolve only the pinned RunPod cached-model snapshot."""
    repo_dir = os.path.join(
        RUNPOD_HF_CACHE, f"models--{HF_REPO.replace('/', '--')}"
    )
    candidate = os.path.join(
        repo_dir, "snapshots", SAPIENS_MODEL_REVISION, CKPT_NAME
    )
    return candidate if os.path.isfile(candidate) else None


def _ensure_checkpoint() -> str:
    # Baked-into-image checkpoint (bf16, written by build-weights.yml) — the
    # production path. No network, no HF, cold start = local NVMe read only.
    baked = os.path.join(WEIGHTS_DIR, CKPT_NAME)
    if os.path.isfile(baked):
        gb = os.path.getsize(baked) / 1e9
        log(f"baked checkpoint present ({gb:.1f} GB) at {baked}")
        return baked

    cached = _runpod_cached_checkpoint()
    if cached:
        gb = os.path.getsize(cached) / 1e9
        log(f"RunPod cached checkpoint present ({gb:.1f} GB) at {cached}")
        return cached

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download

    log(
        f"resolving {CKPT_NAME} from {HF_REPO}@{SAPIENS_MODEL_REVISION} "
        f"-> {WEIGHTS_DIR} (20 GB for 5b)..."
    )
    t = time.time()
    src = hf_hub_download(
        repo_id=HF_REPO,
        filename=CKPT_NAME,
        revision=SAPIENS_MODEL_REVISION,
        local_dir=WEIGHTS_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    log(f"checkpoint downloaded in {time.time() - t:.1f}s")
    return src


def _load():
    from .model_loader import load_pose_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The 5B checkpoint is ~20 GB in fp32. The fast path builds on ``meta`` and
    # streams each tensor to CUDA as bf16. MODEL_LOAD_DEVICE=cpu remains the
    # safe upstream fallback, because restoring the fp32 model directly on a
    # 16–24 GB GPU would OOM before conversion.
    load_device = _MODEL_LOAD_DEVICE or device
    if load_device == "cuda" and device != "cuda":
        raise RuntimeError("MODEL_LOAD_DEVICE=cuda requested but CUDA is unavailable")
    log(f"loading model: size={MODEL_SIZE} device={device} compile={_COMPILE} "
        f"load_device={load_device} weights_dir={WEIGHTS_DIR}")
    cfg = _find_config()
    ckpt = _ensure_checkpoint()

    t = time.time()
    model = load_pose_model(
        cfg,
        ckpt,
        device=load_device,
        target_device=device if device == "cuda" else None,
        target_dtype=torch.bfloat16 if device == "cuda" else None,
    )
    model.eval()
    log(f"init_model done in {time.time() - t:.1f}s")

    # init_model gives .pipeline/.data_preprocessor but NOT .codec — build the
    # UDPHeatmap decoder from the config (as sapiens2's vis_pose demo does).
    from sapiens.pose.datasets import UDPHeatmap

    codec_cfg = dict(model.cfg.codec)
    codec_cfg.pop("type", None)
    model.codec = UDPHeatmap(**codec_cfg)
    t = time.time()
    model = model.to(device=device, dtype=torch.bfloat16)
    log(f"final model device/dtype normalization done in {time.time() - t:.1f}s")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        log(
            f"model -> bf16 on cuda, codec attached | "
            f"VRAM allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB"
        )
    else:
        log("model -> bf16 on cpu, codec attached")

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


def warmup(batch_size: int | None = None) -> float:
    """Pay CUDA initialization (and optional compilation) before worker readiness."""
    if DEVICE != "cuda":
        log("model warmup skipped: CUDA unavailable")
        return 0.0

    batch = batch_size or int(os.environ.get("BATCH_SIZE", "1"))
    if batch < 1:
        raise ValueError("warmup batch size must be >= 1")

    log(f"model warmup starting: batch={batch} compile={_COMPILE}")
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    inputs = torch.zeros(
        (batch, 3, 1024, 768),
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        output = FORWARD(inputs)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    del output, inputs
    torch.cuda.empty_cache()
    log(f"model warmup done in {elapsed:.1f}s; peak VRAM={peak_gib:.2f} GiB")
    return elapsed
