"""RunPod serverless entrypoint.

Importing core.infer triggers the model load at module level (before
serverless.start), which is what FlashBoot needs to snapshot a warm worker.
The handler itself is a thin shim over run_video().
"""
import runpod

from core.infer import run_video


def handler(event):
    inp = event.get("input") or {}
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
        return run_video(video_url, **kwargs)
    except Exception as e:  # surface the error to the job result
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})
