"""Vast.ai serverless entrypoint — FastAPI HTTP server (same core as RunPod).

  GET  /healthz   unauthenticated build fingerprint (git_sha + api_key_sha256)
  POST /infer     requires api_key; runs the pose pipeline

Model is lazy-loaded on the first /infer so /healthz stays instant.
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from core import auth

app = FastAPI(title="serve_luche")


class InferRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    api_key: str = Field(min_length=1)
    video_url: str = Field(min_length=1)
    result_put_url: str | None = None
    frame_stride: int = Field(default=1, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    person_detection: StrictBool = False
    person_detection_stride: int = Field(default=5, ge=1)
    person_box_overflow: float = Field(default=0.25, ge=0.0, le=2.0)
    person_detection_threshold: float = Field(default=0.3, gt=0.0, le=1.0)


@app.get("/healthz")
def healthz():
    return auth.version()


@app.post("/infer")
def infer(req: InferRequest):
    if not auth.check(req.api_key):
        return {"error": "unauthorized"}
    from core.infer import run_video, DEFAULT_BATCH_SIZE

    result = run_video(
        req.video_url,
        result_put_url=req.result_put_url,
        frame_stride=req.frame_stride,
        batch_size=req.batch_size or DEFAULT_BATCH_SIZE,
        person_detection=req.person_detection,
        person_detection_stride=req.person_detection_stride,
        person_box_overflow=req.person_box_overflow,
        person_detection_threshold=req.person_detection_threshold,
    )
    result["git_sha"] = os.environ.get("GIT_SHA", "unknown")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
