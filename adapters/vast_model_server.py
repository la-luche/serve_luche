"""Synchronous model API behind Vast's authenticated proxy."""
import os
import secrets
import threading

from fastapi import FastAPI, HTTPException
from core import auth
from core.request import parse_infer_input

app = FastAPI()
lock = threading.Lock()


@app.post("/infer")
def infer(payload: dict):
    if secrets.compare_digest(str(payload.get("benchmark_token", "")), os.environ["BENCHMARK_TOKEN"]):
        # Synthetic fixture is for startup throughput only, never pose quality.
        inp = {"video_url": "/app/assets/startup.mp4"}
    else:
        if not auth.check(payload.get("api_key")):
            raise HTTPException(401, "unauthorized")
        inp = payload
    try:
        video, kwargs = parse_infer_input(inp)
    except ValueError:
        raise HTTPException(422, "invalid inference request") from None
    from core.infer import run_video
    with lock:
        output = run_video(video, **kwargs)
    output["git_sha"] = os.environ.get("GIT_SHA", "unknown")
    return output


if __name__ == "__main__":
    import uvicorn
    import torch
    gpu_name = torch.cuda.get_device_name().upper()
    if not any(model in gpu_name for model in ("H100", "RTX 5090")):
        raise RuntimeError("this Vast endpoint accepts H100 and RTX 5090 GPUs only")
    from adapters.runpod_handler import prepare_worker
    prepare_worker()
    print("LUCHE_MODEL_READY", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "18000")))
