# sapiens-serve

Serverless keypoint detection. Send a **presigned R2 GET** for a video, get
**308 Goliath whole-body keypoints per processed frame** written to R2 via a
**presigned PUT**. Runs the Meta **Sapiens 1B** pose model (TorchScript, bf16) on
an H100. One image, thin adapters for **RunPod** and **Vast**.

## Design in one picture

```
   [ build, no GPU ]                 [ run, needs H100 ]
GitHub Actions (amd64) ── push ──►  GHCR  ── pull ──►  RunPod serverless
   COPY weights .pt2                 (one image)       Vast workergroup
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
    "include_conf": true
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

1. **Weights:** accept the Sapiens license, then
   `export HF_TOKEN=hf_xxx && ./download_weights.sh` → `weights/*.pt2`.
2. **Validate on a GPU box (do this before containerizing):** rent one H100
   (Vast/RunPod pod), `pip install -r requirements.txt`, then
   `python run_local.py --video "$GET_URL" --put-url "$PUT_URL" --stride 2`.
   Confirm keypoints look sane and read the real per-video time.
3. **Image:** push to GitHub → the `build-and-push` workflow builds amd64 and
   pushes to `ghcr.io/<owner>/serve_luche`. (Needs repo secret `HF_TOKEN`.)
4. **Deploy** the same image to both platforms (below).

## Deploy

**RunPod:** create a Serverless endpoint from the GHCR image. Container start
command is the image default (`python -u adapters/runpod_handler.py`). Set
GPU = H100. Keep `min_workers=0` for cost (first request after each new image
pays a cold start — FlashBoot snapshots are keyed to the image SHA); set `>=1`
to eliminate cold starts.

**Vast:** create a workergroup from the same image with the start command
overridden to `python adapters/vast_worker.py`, expose `$PORT`, and point the
PyWorker template's forward URL + BenchmarkConfig at `/infer`. Same `core/`.

## Cold start

The model loads at **import** in `core/model.py` (before `serverless.start`), so
RunPod FlashBoot snapshots a *warm* worker. Never move the load into the handler.

## Performance notes / gotchas

- **H100, batch 32-64:** ~1-3 min for a 3-min video (stride 2 ≈ 2,700 frames).
  Tune `batch_size` up until just before OOM — the 1B fits easily in 80 GB.
- **Decode is often the bottleneck**, not the forward pass. We decode strided
  frames on GPU (torchcodec/NVDEC) to avoid H2D copies.
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
