"""RunPod serverless entrypoint.

- `{"input": {"ping": true}}`  -> unauthenticated build fingerprint (git_sha +
  api_key_sha256 + model_size). Use it to verify which build a container runs.
- Everything else requires `api_key` (checked by SHA-256 against the baked hash).

The heavy Sapiens2 model is lazy-imported inside the video path so `ping` (and
auth failures) return instantly without paying the ~5 min cold-load. The model
loads on the first authenticated video job.
"""
import os

import runpod

from core import auth
from core.log import log
from core.request import parse_infer_input


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
    from core.infer import run_video  # lazy: triggers model load on first job

    result = run_video(video_url, **kwargs)
    result["git_sha"] = os.environ.get("GIT_SHA", "unknown")
    return result


runpod.serverless.start({"handler": handler})
