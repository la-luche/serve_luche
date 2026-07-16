# Sapiens2 pose keypoint serverless image — PORTABLE across RunPod + Vast.
# The 20 GB Sapiens weights are NOT baked. The much smaller person-detector
# weights ARE baked so the optional top-down crop path never downloads at runtime.
# On RunPod, configure the cached Sapiens model `facebook/sapiens2-pose-5b`.
# The Inductor compile-cache is kept in $WEIGHTS_DIR. One image, two entrypoints
# (RunPod handler / Vast FastAPI).
#
# runtime base (not devel) keeps the image ~10 GB so it builds on a free GH runner.
# torch.compile only needs a C++ compiler (g++) + Triton, not nvcc — so we add
# build-essential and skip the multi-GB CUDA toolkit.
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1
# /app on the path so `import core` resolves when CMD runs adapters/*.py
# (running a script puts ITS dir on sys.path, not the workdir).
ENV PYTHONPATH=/app
ENV WEIGHTS_DIR=/weights MODEL_SIZE=5b
ENV PERSON_DETECTOR_DIR=/opt/models/detr-resnet-101-dc5
ENV PERSON_DETECTOR_MODEL=/opt/models/detr-resnet-101-dc5
ENV PERSON_DETECTOR_NAME=facebook/detr-resnet-101-dc5

# Auth: only the SHA-256 of the static API key is baked in (safe for a public
# image — the key itself never touches the repo/image). Handler checks
# sha256(request.api_key) == API_KEY_SHA256. GIT_SHA lets you verify which build
# a running container is (via the unauthenticated {"ping": true} request).
ENV API_KEY_SHA256=88edc437185f02bf774f458a8f3e3404d6b9e49c89ceb3393a31cd5579f9d446
ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sapiens2 (clean deps: torch/torchvision/transformers/timm — no mmcv/mmpose).
RUN git clone --depth 1 https://github.com/facebookresearch/sapiens2.git /opt/sapiens2 \
    && pip install --no-cache-dir -e /opt/sapiens2

# Pin the detector revision and save only its inference files into the image.
# Download + cache removal happen in one layer so the final image has one copy.
ARG PERSON_DETECTOR_REVISION=96317ca979e231bd960cb3cac31328e0165a3e94
RUN HF_HOME=/tmp/hf-person-build python -c "from transformers import AutoConfig, AutoImageProcessor, DetrForObjectDetection; repo='facebook/detr-resnet-101-dc5'; revision='${PERSON_DETECTOR_REVISION}'; dst='${PERSON_DETECTOR_DIR}'; AutoImageProcessor.from_pretrained(repo, revision=revision, use_fast=False).save_pretrained(dst); config=AutoConfig.from_pretrained(repo, revision=revision); config.use_pretrained_backbone=False; model=DetrForObjectDetection.from_pretrained(repo, revision=revision, config=config); model.save_pretrained(dst, safe_serialization=True)" \
    && rm -rf /tmp/hf-person-build

RUN pip install --no-cache-dir decord kornia requests huggingface_hub runpod fastapi uvicorn

COPY core/ /app/core/
COPY adapters/ /app/adapters/
COPY run_local.py /app/

# Default entrypoint = RunPod. Vast overrides CMD -> python adapters/vast_worker.py
CMD ["python", "-u", "adapters/runpod_handler.py"]
