# sapiens-serve

Serverless keypoint detection. Send a **presigned R2 GET** for a video, get
**presigned PUT**. Runs the Meta **Sapiens2 5B** pose model (safetensors,
`torch.compile`, bf16) on an H100 and returns 308 Sociopticon whole-body
keypoints. Optional baked DETR person detection supplies a tracked top-down crop.
One image, thin adapters for **RunPod** and **Vast**.

> **API reference: [`docs/API.md`](docs/API.md)** — auth, request/response,
> keypoint indices, curl examples. (The request/response snippets below are v1 and
> outdated; docs/API.md is authoritative.)


## Design in one picture

```
   [ build, no GPU ]                     [ run, needs H100 ]
GitHub Actions (amd64) ── push ──► GHCR ── pull ──► RunPod serverless
                                           cached 5B model mount
```

- `core/` — platform-agnostic inference (`run_video`). No RunPod/Vast imports.
- `adapters/runpod_handler.py` — RunPod queue handler.
- `adapters/vast_worker.py` — FastAPI server for a Vast workergroup.
- `run_local.py` — on-box benchmark / smoke test (no platform).

**Credential-free by design:** the client passes a presigned GET (video in) and a
presigned PUT (results out), so the container holds **no R2 secrets** and the
same image runs anywhere.

## Request / response (async)

Submit via RunPod `/run` (or Vast `POST /infer`):

```json
{ "input": {
    "video_url": "https://<r2>/video.mp4?X-Amz-...",
    "result_put_url": "https://<r2>/results/abc.json?X-Amz-...",
    "frame_stride": 2,
    "batch_size": 16,
    "include_conf": true,
    "person_detection": true,
    "person_detection_stride": 5,
    "person_box_overflow": 0.25
} }
```

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

1. **Validate on a GPU box (do this before containerizing):** rent one H100
   (Vast/RunPod pod), `pip install -r requirements.txt`, then
   `python run_local.py --video "$GET_URL" --put-url "$PUT_URL" --stride 2`.
   Confirm keypoints look sane and read the real per-video time.
2. **Image:** push to GitHub → the `build-and-push` workflow builds amd64 and
   pushes to `ghcr.io/<owner>/serve_luche`. No Hugging Face build secret is needed.
3. **Deploy** the same image to both platforms (below).

## Deploy

**RunPod (recommended):** create a queue-based Serverless endpoint from the GHCR
image. In the endpoint's **Model** field select `facebook/sapiens2-pose-5b` so
RunPod mounts the checkpoint from its model cache before starting the worker.
The loader resolves that mount automatically. Keep the default container command
(`python -u adapters/runpod_handler.py`), choose H100 80 GB, enable FlashBoot,
set execution timeout and job TTL to 24 hours, and use one active worker while
you need predictable latency. Set active workers back to zero when cost matters
more than cold-start latency.

**Vast:** create a workergroup from the same image with the start command
overridden to `python adapters/vast_worker.py`, expose `$PORT`, and point the
PyWorker template's forward URL + BenchmarkConfig at `/infer`. Same `core/`.

## Cold start

RunPod's cached model avoids downloading the 20 GB checkpoint inside billed worker
time, but PyTorch must still initialize the model and compile its first forward.
One active worker removes that user-visible cold start. With zero active workers,
FlashBoot helps subsequent starts but the first deployment can still be slow.

Inference uses one fixed shape per configured batch size. A short final batch is
padded to `BATCH_SIZE` before the compiled forward and sliced back afterward, so
it does not trigger a second compilation. The production API does not override
the image's `BATCH_SIZE=16`; changing that value between direct requests can
create another compiled graph.

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

- **RTX 5080 / Blackwell:** needs torch >= 2.7 + CUDA 12.8 (the pinned torch 2.4
  has no `sm_120` kernels). 16 GB fits 1B fine; 0.6b is even lighter
 .
- **RGB vs BGR:** `core/postproc.py:INPUT_BGR` defaults to RGB. If step-2
  keypoints look mirrored/garbage, flip it — it's the #1 thing to check.
- **Sub-pixel decode** is a light quarter-pixel Darkpose-style shift; swap in
  full UDP if you need max accuracy.

## Not a fit: Yandex Cloud

Yandex's serverless (Serverless Containers, Cloud Functions) is **CPU-only**, and
Compute tops out at A100 80GB (no H100). No scale-to-zero GPU. If ever needed
(RU data residency), run this same image on a Yandex **Managed K8s GPU node
group** with a small always-on deployment instead of scale-to-zero.
