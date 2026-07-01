# Sapiens 2B keypoint serverless image. One image, multiple entrypoints
# (RunPod handler / Vast FastAPI). Weights are baked in for warm cold-starts.
#
# Build for the deploy target's arch (amd64 GPU hosts):
#   docker buildx build --platform linux/amd64 -t ghcr.io/<you>/sapiens-serve:latest --push .
#
# Weights must exist in ./weights/*.pt2 at build time (run ./download_weights.sh
# first, or let CI download them — see .github/workflows/build.yml).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.0 \
    && pip install --no-cache-dir -r requirements.txt

# Bake the model in (the ~4GB COPY that makes FlashBoot worthwhile).
COPY weights/ /app/weights/
COPY core/ /app/core/
COPY adapters/ /app/adapters/
COPY run_local.py /app/

# Default entrypoint = RunPod. Vast overrides CMD to launch the FastAPI worker:
#   python adapters/vast_worker.py
CMD ["python", "-u", "adapters/runpod_handler.py"]
