# Sapiens2 pose keypoint image — portable across RunPod and dedicated GPUs.
# The 20 GB Sapiens weights are NOT baked. The much smaller person-detector
# weights ARE baked so the optional top-down crop path never downloads at runtime.
# On RunPod, configure the cached Sapiens model `facebook/sapiens2-pose-5b`.
# The Inductor compile-cache is kept in $WEIGHTS_DIR. One image, two entrypoints:
# RunPod's handler or our provider-neutral HTTP queue on Vast/the lab RTX 5080.
#
# runtime base (not devel) keeps the image ~10 GB so it builds on a free GH runner.
# torch.compile only needs a C++ compiler (g++) + Triton, not nvcc — so we add
# build-essential and skip the multi-GB CUDA toolkit.
FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime@sha256:72f863fa1fe13d5d87a72d00db2c85fb2d43409ee08dd26bc469de4c8a28b427

LABEL org.opencontainers.image.source="https://github.com/la-luche/serve_luche"

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1
# /app on the path so `import core` resolves when CMD runs adapters/*.py
# (running a script puts ITS dir on sys.path, not the workdir).
ENV PYTHONPATH=/app
ENV WEIGHTS_DIR=/weights MODEL_SIZE=5b
# Production Serverless profile: initialize from CPU so the fp32 checkpoint
# never has to fit in VRAM, run batch 1 so 24 GB GPUs are safe, and prefer eager
# inference so there is no per-host Inductor compile in the cold-start path.
# The worker imports + warms the runtime before RunPod sees it as ready.
ENV MODEL_LOAD_DEVICE=cpu BATCH_SIZE=1 COMPILE=0
ENV PRELOAD_MODEL=1 WARMUP_ON_START=1
ENV SAPIENS_MODEL_REVISION=ada1f29aa1fd454ca28665c700923a0101b6b24f
ENV PERSON_DETECTOR_DIR=/opt/models/detr-resnet-101-dc5
ENV PERSON_DETECTOR_MODEL=/opt/models/detr-resnet-101-dc5
ENV PERSON_DETECTOR_NAME=facebook/detr-resnet-101-dc5

# Auth: only the SHA-256 of the static API key is baked in (safe for a public
# image — the key itself never touches the repo/image). Handler checks
# sha256(request.api_key) == API_KEY_SHA256. GIT_SHA lets you verify which build
# a running container is (via the unauthenticated {"ping": true} request).
ENV API_KEY_SHA256=1e058a1a665275a66f1aac6778c8525de451a42c0e9519b84f8802c3452106a5
ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY constraints.txt /tmp/constraints.txt

# Sapiens2 (clean deps: torch/torchvision/transformers/timm — no mmcv/mmpose).
# Fetch the exact commit rather than allowing upstream HEAD to change this image.
ARG SAPIENS2_REVISION=7e5bae88456ac418ff0e58e74106c9fe192055d4
RUN git init /opt/sapiens2 \
    && git -C /opt/sapiens2 remote add origin https://github.com/facebookresearch/sapiens2.git \
    && git -C /opt/sapiens2 fetch --depth 1 origin ${SAPIENS2_REVISION} \
    && git -C /opt/sapiens2 checkout --detach FETCH_HEAD \
    && pip install --no-cache-dir --constraint /tmp/constraints.txt -e /opt/sapiens2

# Pin the detector revision and save only its inference files into the image.
# Download + cache removal happen in one layer so the final image has one copy.
ARG PERSON_DETECTOR_REVISION=96317ca979e231bd960cb3cac31328e0165a3e94
ENV PERSON_DETECTOR_REVISION=${PERSON_DETECTOR_REVISION}
RUN HF_HOME=/tmp/hf-person-build python -c "from huggingface_hub import hf_hub_download; repo='facebook/detr-resnet-101-dc5'; revision='${PERSON_DETECTOR_REVISION}'; dst='${PERSON_DETECTOR_DIR}'; [hf_hub_download(repo_id=repo, filename=name, revision=revision, local_dir=dst) for name in ('config.json','preprocessor_config.json','model.safetensors')]" \
    && HF_HUB_OFFLINE=1 python -c "from transformers import AutoConfig, AutoImageProcessor, DetrForObjectDetection; dst='${PERSON_DETECTOR_DIR}'; processor=AutoImageProcessor.from_pretrained(dst, local_files_only=True, use_fast=False); config=AutoConfig.from_pretrained(dst, local_files_only=True); config.use_pretrained_backbone=False; model=DetrForObjectDetection.from_pretrained(dst, local_files_only=True, config=config); assert model.config.id2label[1].lower() == 'person'" \
    && rm -rf /tmp/hf-person-build ${PERSON_DETECTOR_DIR}/.cache

RUN pip install --constraint /tmp/constraints.txt --no-cache-dir \
        decord kornia requests huggingface_hub runpod fastapi uvicorn

COPY core/ /app/core/
COPY adapters/ /app/adapters/
COPY tests/ /app/tests/
COPY run_local.py /app/

# Pure tracking tests plus mandatory Sapiens-backed affine/fallback geometry tests.
# REQUIRE_SAPIENS_TESTS makes a missing/broken production import fail the build.
RUN REQUIRE_SAPIENS_TESTS=1 python -m unittest discover -s /app/tests -v

# Default entrypoint = RunPod. A dedicated GPU overrides this with
# `python -u adapters/http_worker.py`.
CMD ["python", "-u", "adapters/runpod_handler.py"]
