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


def handler(event):
    inp = event.get("input") or {}

    if inp.get("ping"):
        return auth.version()

    if not auth.check(inp.get("api_key")):
        log("job rejected: unauthorized")
        return {"error": "unauthorized"}
    log(f"job accepted: stride={inp.get('frame_stride', 1)} batch={inp.get('batch_size')}")

    video_url = inp.get("video_url")
    if not video_url:
        return {"error": "missing 'video_url' in input"}

    kwargs = {
        "result_put_url": inp.get("result_put_url"),
        "frame_stride": int(inp.get("frame_stride", 1)),  # default: every frame
    }
    if inp.get("batch_size"):
        kwargs["batch_size"] = int(inp["batch_size"])

    try:
        from core.infer import run_video  # lazy: triggers model load on first job

        result = run_video(video_url, **kwargs)
        result["git_sha"] = os.environ.get("GIT_SHA", "unknown")
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "git_sha": os.environ.get("GIT_SHA")}


runpod.serverless.start({"handler": handler})
