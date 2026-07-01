"""Vast.ai serverless entrypoint — FastAPI HTTP server (same core as RunPod).

  GET  /healthz   unauthenticated build fingerprint (git_sha + api_key_sha256)
  POST /infer     requires api_key; runs the pose pipeline

Model is lazy-loaded on the first /infer so /healthz stays instant.
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from core import auth

app = FastAPI(title="serve_luche")


class InferRequest(BaseModel):
    api_key: str
    video_url: str
    result_put_url: str | None = None
    frame_stride: int = 1  # default: every frame
    batch_size: int | None = None


@app.get("/healthz")
def healthz():
    return auth.version()


@app.post("/infer")
def infer(req: InferRequest):
    if not auth.check(req.api_key):
        return {"error": "unauthorized"}
    try:
        from core.infer import run_video, DEFAULT_BATCH_SIZE

        result = run_video(
            req.video_url,
            result_put_url=req.result_put_url,
            frame_stride=req.frame_stride,
            batch_size=req.batch_size or DEFAULT_BATCH_SIZE,
        )
        result["git_sha"] = os.environ.get("GIT_SHA", "unknown")
        return result
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
