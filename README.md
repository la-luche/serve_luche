# sapiens-serve

Whole-body keypoint detection. Send a **presigned R2 GET** for a video, get
**presigned PUT**. Runs the Meta **Sapiens2 5B** pose model (safetensors, bf16)
on a 16–24 GB GPU and returns 308 Sociopticon whole-body keypoints. Optional
baked DETR person detection supplies a tracked top-down crop.
One image runs on **RunPod**, a normal **Vast** instance, or the lab GPU box.

> **API reference: [`docs/API.md`](docs/API.md)** — auth, request/response,
> keypoint indices, and copy-paste curl examples.


## Design in one picture

```
   [ build, no GPU ]                     [ run, warm 16–24 GB GPU ]
GitHub Actions (amd64) ── push ──► GHCR ── pull ──► RunPod or dedicated worker
```

- `core/` — platform-agnostic inference (`run_video`). No RunPod/Vast imports.
- `adapters/runpod_handler.py` — RunPod queue handler.
- `adapters/http_worker.py` — persistent RunPod-compatible HTTP queue for any dedicated GPU.
- `adapters/vast_worker.py` — compatibility alias for the HTTP worker.
- `run_local.py` — on-box benchmark / smoke test (no platform).

**Credential-free by design:** the client passes a presigned GET (video in) and a
presigned PUT (results out), so the container holds **no R2 secrets** and the
same image runs anywhere.

## Request / response (async)

Submit via RunPod `/run`:

```json
{ "input": {
    "api_key": "sk_luche_…",
    "video_url": "https://<r2>/video.mp4?X-Amz-...",
    "result_put_url": "https://<r2>/results/abc.json?X-Amz-...",
    "frame_stride": 2,
    "batch_size": 1,
    "person_detection": true,
    "person_detection_stride": 5,
    "person_box_overflow": 0.25
} }
```

A dedicated worker accepts the exact same asynchronous `/run`, `/status`, and
`/cancel` calls. It also offers synchronous `/infer` with the fields inside
RunPod's outer `input` object.

The job returns a small summary; the big keypoint array is PUT to R2:

```json
{ "status": "ok", "results_location": "https://<r2>/results/abc.json",
  "num_processed_frames": 2700, "num_source_frames": 5400, "source_fps": 30.0,
  "num_keypoints": 308, "infer_seconds": 118.3, "fps_processed": 22.8 }
```

Results JSON on R2: `{ keypoint_format, num_keypoints, input_hw, source_fps,
frame_stride, frames: [ { frame_idx, keypoints: [[x,y,conf], ...308] } ] }` —
keypoints are in **original video pixel coordinates**.

## Build order

1. **Image:** push to GitHub. The workflow builds amd64, reloads the baked DETR
   fully offline, runs tracking and Sapiens affine tests inside the image, and
   pushes only the immutable commit-SHA tag to GHCR.
2. **CPU/API smoke:** verify the immutable SHA image starts and reports its build
   fingerprint. `requirements.txt` is only a local convenience list; the pinned
   Docker base, Sapiens code/checkpoint revisions, and complete production
   `constraints.txt` define production.
3. **GPU canary:** deploy the immutable SHA tag, ping it, then run a real clip
   before updating normal callers. Never deploy `latest` to RunPod.

## Deploy

**RunPod (production):** create a queue-based Serverless endpoint from the GHCR
image. Leave the endpoint's **Model** field empty: the legacy Model Store entry
blocked for more than 20 minutes during the 2026-08-17 rollout, while the direct
Hugging Face download fetched the 20.48 GB checkpoint in 53.6 seconds. Keep the
default container command (`python -u adapters/runpod_handler.py`), enable
FlashBoot, set execution timeout and job TTL to 24 hours, and set **active
workers = 1, max workers = 1**. Prioritize the 24 GB L4 / RTX A5000 / RTX 3090 /
RTX 4090 pools. The batch-1 eager profile fits these cards; requests queue behind
the warm worker instead of starting a second cold worker. Set the template image
to an immutable `ghcr.io/la-luche/serve_luche:<40-character-git-sha>` tag.

**Dedicated RTX 5080 (Vast or lab):** run the same image with port 8000 exposed
and override the command to `python -u adapters/http_worker.py`. Use persistent
paths for `WEIGHTS_DIR` and `JOB_STATE_DIR`, then set:

```bash
MODEL_LOAD_DEVICE=cpu MODEL_SIZE=5b BATCH_SIZE=1 COMPILE=1
PERSON_DETECTOR_DEVICE=cpu WEIGHTS_DIR=/workspace/weights
JOB_STATE_DIR=/workspace/job-state PORT=8000
```

`MODEL_LOAD_DEVICE=cpu` is essential on a 16 GB card: the 20 GB fp32 checkpoint
is restored in system RAM and transferred to CUDA directly as ~10 GB bf16.

## Cold start

The production contract is deliberately **warm, eager, and batch 1**:

- `PRELOAD_MODEL=1` imports the full runtime before registering the handler, so
  RunPod does not report a worker ready while it is still loading 20 GB of weights.
- `WARMUP_ON_START=1` pays CUDA initialization before readiness.
- `COMPILE=0` removes the host-specific Inductor compile from every fresh worker.
  Inference is slower, but startup has a much tighter tail.
- `MODEL_LOAD_DEVICE=cpu` restores fp32 weights in system RAM and moves them to
  CUDA as bf16, allowing the 5B model to start safely on 16–24 GB cards.
- `FAST_META_LOAD=1` builds the 5B module graph on PyTorch's zero-storage
  `meta` device, streams each checkpoint tensor directly to its final bf16 CUDA
  slot, and assigns it without creating a second fp32 model. If a future Sapiens
  architecture cannot use this path, startup falls back to the upstream loader.
- One active worker is the only way to guarantee no scale-to-zero cold start.
  FlashBoot reduces recovery time but is not an availability guarantee. Keep
  `workersMax=1`; a burst queues rather than spawning a cold copy.

The direct checkpoint download is observable in the worker log and was faster
than the legacy Model Store in the measured rollout. A worker replacement still
has to initialize the model, but it is not offered work until initialization and
warmup succeed.

### Measured RunPod replacement profile (2026-08-17)

| Stage | Result |
|---|---:|
| Cache-cold image provision/pull in US-WA-1 | ~137 s |
| Download 20.48 GB checkpoint | 34.2 s |
| Build zero-storage model graph | 0.4 s |
| Stream checkpoint to bf16 CUDA | 58.5 s |
| Assign state tensors | 0.3 s |
| Eager batch-1 warmup | 1.0 s |
| Handler ready | 98.7 s from container start |
| Cache-cold worker record → handler ready | ~236 s |
| Resident / peak warmup VRAM | 9.63 / 9.93 GiB |

The original upstream restore took 376.6 s after container start, including
249.4 s just to build/copy the fp32 model. The optimized path cuts container
preparation by 74% and produced bit-identical keypoints in the production
canary (8,316 values compared, maximum absolute difference 0).

RunPod may label a worker `RUNNING` before the SDK handler has registered. Use
the handler's `worker runtime ready` log line or a successful ping as the
readiness signal. Normal requests avoid this replacement path because production
keeps one active worker.

Production is pinned to `US-WA-1`: repeated `EUR-IS-2` allocations remained in
image initialization while measured US-WA workers progressed normally. Revisit
the pin only with a measured replacement canary in another region.

Inference uses one fixed shape per configured batch size. A short final batch is
padded to `BATCH_SIZE` before the forward and sliced back afterward. Production
uses `BATCH_SIZE=1`; dedicated workers with persistent compile caches may opt
back into `COMPILE=1` and a hardware-appropriate fixed batch.

On an RTX 5080, keep batch size fixed at 1. The compiled graph is cached under
`WEIGHTS_DIR`; the first forward compiles once and subsequent processes reuse it.

### Measured dedicated RTX 5080 profile (5B, bf16, compiled batch 1, 1024×768)

| Metric | Result |
|---|---:|
| Resident model VRAM | 9.54 GiB |
| Peak inference VRAM | 9.84 GiB |
| First uncached compile/forward | 58.5 s |
| First forward after process restart with cache | 8.7 s |
| Warm steady-state throughput | ~2.6 fps |
| 146-frame canary including cached first forward | 65.2 s (2.24 fps) |

These measurements are from an RTX 5080 with torch 2.12.1/CUDA 13.0. The 5B
model therefore fits with more than 5 GiB physical VRAM headroom; no inference
activation checkpointing or quantization is required.

## Measured on one H100 80GB (batch 32, 1080p, stride 2 = 2,700 frames)

| Model | Keypoints | End-to-end fps | Forward-only fps | 3-min video |
|-------|-----------|----------------|------------------|-------------|
| **1B Goliath**   | 308 (dense face) | 22.9 | 26.2 | **118 s** |
| **2B WholeBody** | 133 (21/hand, fingers) | 14.8 | 17.5 | **183 s** |

- Batch 32 ≈ batch 64 (118.2 s vs 118.8 s) → **compute-bound on the ViT forward**;
  bigger batches don't help. Batch 64 used only 38 GB of 80 GB.
- **Decode is NOT the bottleneck here** (initial assumption was wrong): decord CPU
  decode is only ~15 s = **13%** of wall time; the forward is ~87%. GPU/NVDEC decode
  buys little for this model at 1024x768 — model math is the wall.
- There is **no 2B *Goliath* (308-kp)** checkpoint — 2B pose maxes at 133-kp
  WholeBody. 308-kp Goliath tops out at 1B. Both hosted (gated) on
  `noahcao/sapiens-pose-coco`; 1B Goliath also ungated on `facebook/sapiens-pose-1b-torchscript`.

## Compile / speedup attempts — all FAILED on the shipped checkpoints

The public Sapiens `.pt2` are **frozen TorchScript mmpose "topdown" wrappers**, not
clean tensor-in/out modules, so they resist off-the-shelf recompilation:
- `torch.compile` — N/A (needs eager nn.Module, not a ScriptModule).
- **Torch-TensorRT** TS frontend — `Unable to freeze tensor` / graphShapeAnalyzer
  assertion on the ViT graph (TensorRT 10).
- **ONNX export** — fails in the TorchScript interpreter on the mmpose wrapper.

Only remaining path to a TRT/fp8 speedup: rebuild the **eager** model (mmpose +
Sapiens code + the *non*-TorchScript checkpoint), then dynamo-TRT. Multi-hour effort.

## Gotchas

- **RTX 5080 / Blackwell:** use the pinned torch 2.12.1 + CUDA 13 image,
  `MODEL_LOAD_DEVICE=cpu`, bf16, and batch 1. Loading the fp32 checkpoint directly
  on CUDA OOMs before conversion. Activation checkpointing does not reduce
  inference memory because `torch.inference_mode()` retains no backward activations.
- **RGB vs BGR:** `core/postproc.py:INPUT_BGR` defaults to RGB. If step-2
  keypoints look mirrored/garbage, flip it — it's the #1 thing to check.
- **Sub-pixel decode** is a light quarter-pixel Darkpose-style shift; swap in
  full UDP if you need max accuracy.

## Not a fit: Yandex Cloud

Yandex's serverless (Serverless Containers, Cloud Functions) is **CPU-only**, and
Compute tops out at A100 80GB (no H100). No scale-to-zero GPU. If ever needed
(RU data residency), run this same image on a Yandex **Managed K8s GPU node
group** with a small always-on deployment instead of scale-to-zero.
