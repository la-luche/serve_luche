"""Memory-efficient Sapiens checkpoint restore with an upstream fallback."""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path

from .log import log


def _meta_tensors(model) -> list[str]:
    names = [name for name, value in model.named_parameters() if value.is_meta]
    names.extend(name for name, value in model.named_buffers() if value.is_meta)
    return names


def _load_safetensors_meta(config_path: str, checkpoint: str, device: str):
    """Build on ``meta`` and assign checkpoint tensors without an fp32 copy."""
    import torch
    from safetensors.torch import load_file
    from sapiens.engine.config import Config
    from sapiens.engine.datasets import Compose
    from sapiens.registry import MODELS

    started = time.time()
    config = Config.fromfile(config_path)
    if "init_cfg" in config.model["backbone"]:
        config.model["backbone"].pop("init_cfg")

    stage = time.time()
    with torch.device("meta"):
        model = MODELS.build(config.model)
    log(f"meta model graph built in {time.time() - stage:.1f}s")

    # These are small runtime helpers and should have real CPU storage.
    data_preprocessor = MODELS.build(config.data_preprocessor)

    stage = time.time()
    state_dict = load_file(checkpoint, device="cpu")
    log(f"safetensors mapped in {time.time() - stage:.1f}s")

    stage = time.time()
    incompat = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    gc.collect()
    log(f"state tensors assigned in {time.time() - stage:.1f}s")

    remaining_meta = _meta_tensors(model)
    if remaining_meta:
        preview = ", ".join(remaining_meta[:8])
        raise RuntimeError(
            f"meta restore left {len(remaining_meta)} tensors unmaterialized: {preview}"
        )
    if incompat.missing_keys:
        log(f"checkpoint missing keys: {incompat.missing_keys}")
    if incompat.unexpected_keys:
        log(f"checkpoint unexpected keys: {incompat.unexpected_keys}")

    model.cfg = config
    model.data_preprocessor = data_preprocessor
    model.pipeline = Compose(config.test_pipeline)
    model.to(device)
    model.eval()
    log(f"meta checkpoint restore done in {time.time() - started:.1f}s")
    return model


def load_pose_model(
    config_path: str,
    checkpoint: str,
    device: str,
    *,
    use_meta: bool | None = None,
):
    """Restore Sapiens, falling back to upstream if meta loading is unsupported."""
    if use_meta is None:
        use_meta = os.environ.get("FAST_META_LOAD", "1") == "1"

    if use_meta and Path(checkpoint).suffix == ".safetensors":
        try:
            return _load_safetensors_meta(config_path, checkpoint, device)
        except Exception as exc:
            # Startup latency is secondary to availability. A future upstream
            # architecture may create a tensor that cannot live on ``meta``;
            # preserve the proven loader rather than boot-looping the endpoint.
            log(f"meta checkpoint restore failed; using upstream loader: {exc!r}")
            gc.collect()

    from sapiens.pose.models import init_model

    return init_model(config_path, checkpoint, device=device)
