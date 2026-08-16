"""RunPod Serverless entrypoint.

- `{"input": {"ping": true}}`  -> unauthenticated build fingerprint (git_sha +
  api_key_sha256 + model_size). Use it to verify which build a container runs.
- Everything else requires `api_key` (checked by SHA-256 against the baked hash).

Production preloads and warms Sapiens2 before registering the worker with
RunPod. That makes worker readiness honest and keeps model initialization out of
the first real job. Set ``PRELOAD_MODEL=0`` only for local/control-plane smoke
tests that intentionally need a lightweight process.
"""
import os
import time
from collections.abc import Callable

import runpod

from core import auth
from core.log import log
from core.request import parse_infer_input

_RUN_VIDEO: Callable | None = None


def _enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _load_runtime(*, warm: bool) -> Callable:
    global _RUN_VIDEO
    if _RUN_VIDEO is None:
        from core.infer import run_video

        _RUN_VIDEO = run_video
    if warm:
        from core.model import warmup

        warmup()
    return _RUN_VIDEO


def prepare_worker() -> None:
    if not _enabled("PRELOAD_MODEL", True):
        log("model preload disabled; first inference will initialize the runtime")
        return

    started = time.time()
    warm = _enabled("WARMUP_ON_START", True)
    log(f"preparing worker before RunPod readiness: warmup={warm}")
    _load_runtime(warm=warm)
    log(f"worker runtime ready in {time.time() - started:.1f}s")


def handler(event):
    inp = event.get("input") if isinstance(event, dict) else None
    if not isinstance(inp, dict):
        return {
            "error": "invalid input: input must be a JSON object",
            "git_sha": os.environ.get("GIT_SHA"),
        }

    if inp.get("ping"):
        return auth.version()

    if not auth.check(inp.get("api_key")):
        log("job rejected: unauthorized")
        return {"error": "unauthorized"}
    try:
        video_url, kwargs = parse_infer_input(inp)
        log(
            f"job accepted: stride={kwargs['frame_stride']} "
            f"batch={kwargs.get('batch_size')} "
            f"person_detection={kwargs['person_detection']} "
            f"detector_stride={kwargs['person_detection_stride']} "
            f"overflow={kwargs['person_box_overflow']}"
        )
    except (TypeError, ValueError) as exc:
        return {
            "error": f"invalid input: {exc}",
            "git_sha": os.environ.get("GIT_SHA"),
        }

    # Let operational model/download/GPU failures escape so RunPod marks the
    # job FAILED and applies its normal retry/metrics semantics. Optional DETR
    # failures are handled explicitly inside run_video as full-frame fallback.
    run_video = _load_runtime(warm=False)
    result = run_video(video_url, **kwargs)
    result["git_sha"] = os.environ.get("GIT_SHA", "unknown")
    return result


def main() -> None:
    prepare_worker()
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
