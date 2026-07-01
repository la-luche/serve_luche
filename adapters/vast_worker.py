"""Vast.ai serverless entrypoint — a FastAPI HTTP server.

Vast's serverless model is instance-centric: a PyWorker HTTP proxy on the GPU
instance fronts a local model server and reports metrics to the autoscaler. This
file is that local server. It exposes:

  GET  /healthz        readiness (model is loaded at import → ready when up)
  POST /infer          {"video_url", "result_put_url", "frame_stride", ...}

Importing core.infer at module load means the model is resident before the
server accepts traffic (same warm-start property we want on RunPod).

To wire into a Vast *workergroup* with autoscaling, front this with their
PyWorker template pointing at http://localhost:$PORT/infer and set a
BenchmarkConfig hitting /infer. See README "Deploy → Vast".
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from core.infer import DEFAULT_BATCH_SIZE, run_video

app = FastAPI(title="sapiens-serve")


class InferRequest(BaseModel):
    video_url: str
    result_put_url: str | None = None
    frame_stride: int = 1  # default: every frame
    batch_size: int = DEFAULT_BATCH_SIZE


@app.get("/healthz")
def healthz():
    return {"status": "ready"}


@app.post("/infer")
def infer(req: InferRequest):
    try:
        return run_video(
            req.video_url,
            result_put_url=req.result_put_url,
            frame_stride=req.frame_stride,
            batch_size=req.batch_size,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
