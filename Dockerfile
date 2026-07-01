# Sapiens2 pose keypoint serverless image — PORTABLE across RunPod + Vast.
# Weights are NOT baked (5B is 20 GB); fetched at startup into $WEIGHTS_DIR and
# cached there (RunPod network volume / Vast instance disk), along with the
# Inductor compile-cache. One image, two entrypoints (RunPod handler / Vast FastAPI).
#
# runtime base (not devel) keeps the image ~10 GB so it builds on a free GH runner.
# torch.compile only needs a C++ compiler (g++) + Triton, not nvcc — so we add
# build-essential and skip the multi-GB CUDA toolkit.
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1
ENV WEIGHTS_DIR=/weights MODEL_SIZE=5b

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sapiens2 (clean deps: torch/torchvision/transformers/timm — no mmcv/mmpose).
RUN git clone --depth 1 https://github.com/facebookresearch/sapiens2.git /opt/sapiens2 \
    && pip install --no-cache-dir -e /opt/sapiens2

RUN pip install --no-cache-dir decord requests huggingface_hub runpod fastapi uvicorn

COPY core/ /app/core/
COPY adapters/ /app/adapters/
COPY run_local.py /app/

# Default entrypoint = RunPod. Vast overrides CMD -> python adapters/vast_worker.py
CMD ["python", "-u", "adapters/runpod_handler.py"]
