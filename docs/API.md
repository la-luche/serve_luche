# serve_luche API

Whole-body keypoint detection. Send a video (as a presigned R2 GET
URL), get **308 Sapiens2 keypoints per frame** back — either inline (small clips)
or written to R2 (presigned PUT). Same image runs on **RunPod**, a normal Vast
instance, and the lab RTX 5080.

- **Live RunPod endpoint:** `uf9tlbqtd90q1y`
- **Image:** `ghcr.io/la-luche/serve_luche` (public)
- **Model:** Sapiens2-pose-5b, 308-keypoint Sociopticon (body + feet + hands + face)

---

## Auth

Every real request needs a **static API key**. Only the **SHA-256** of the key is
baked into the (public) image — the key itself is never in the repo/image; the
handler compares `sha256(request.api_key)` against the baked hash (constant-time).

- Wrong/missing key → `{"error": "unauthorized"}`.
- On **RunPod** you also need the account's RunPod API key in the HTTP
  `Authorization: Bearer` header (that's RunPod's own gate); `api_key` in the body
  is *our* app gate.

Current key hash (baked): `1e058a1a665275a66f1aac6778c8525de451a42c0e9519b84f8802c3452106a5`
(the key is held out-of-band; rotate by regenerating and rebuilding the image.)

---

## RunPod (async)

Submit a job, poll for the result.

```bash
# submit
curl -X POST https://api.runpod.ai/v2/uf9tlbqtd90q1y/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {
        "api_key": "sk_luche_…",
        "video_url": "https://<r2>/clip.mp4?X-Amz-…",
        "result_put_url": "https://<r2>/out/kp.json?X-Amz-…",
        "frame_stride": 1,
        "batch_size": 1,
        "person_detection": true,
        "person_detection_stride": 5,
        "person_box_overflow": 0.25,
        "person_detection_threshold": 0.3
  }}'
# -> {"id": "<job-id>", "status": "IN_QUEUE"}

# poll
curl https://api.runpod.ai/v2/uf9tlbqtd90q1y/status/<job-id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### Version / health check (unauthenticated)
Use this to confirm which build a container is running:
```bash
curl -X POST https://api.runpod.ai/v2/uf9tlbqtd90q1y/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"ping": true}}'
# -> {"status":"ok","git_sha":"<40-hex>","model_size":"5b",
#     "batch_size":1,"compile":false,"preload_model":true,
#     "warmup_on_start":true,"model_load_device":"cpu",
#     "sapiens_model_revision":"<40-hex>",
#     "person_detector":"facebook/detr-resnet-101-dc5",
#     "person_detector_revision":"<40-hex>"}
```

## Dedicated GPU (Vast or lab machine)

```
GET  /healthz          -> fingerprint + busy/queue state (unauthenticated)
POST /run              -> enqueue the same {"input": {...}} body as RunPod
GET  /status/<job-id>   -> IN_QUEUE / IN_PROGRESS / COMPLETED / FAILED / CANCELLED
POST /cancel/<job-id>   -> cooperative cancellation between GPU batches
POST /runsync           -> synchronous RunPod-shaped body
POST /infer             -> synchronous flat body
```

Run `python -u adapters/http_worker.py`. Job status survives process restarts;
active jobs become `FAILED` after a restart so callers can retry instead of
polling a vanished request forever. API keys and presigned URLs are never
persisted.

---

## Request fields

| field | type | default | notes |
|-------|------|---------|-------|
| `api_key` | string | — | **required** (except `ping`); wrong → `unauthorized` |
| `video_url` | string | — | **required**; presigned R2 **GET** (or any http(s) URL / local path) |
| `result_put_url` | string | none | presigned R2 **PUT**; if omitted, keypoints are returned **inline** |
| `frame_stride` | int | `1` | process every Nth frame (`1` = every frame) |
| `batch_size` | int | `1` | frames per GPU batch |
| `person_detection` | bool | `false` | use a tracked person box instead of the full frame |
| `person_detection_stride` | int | `5` | run DETR every Nth **source** frame, then interpolate boxes |
| `person_box_overflow` | float | `0.25` | extra detected-box width/height added on **each side** (`0.25` gives a 1.5× box before 3:4 aspect correction) |
| `person_detection_threshold` | float | `0.3` | DETR person confidence threshold, in `(0,1]` |
| `ping` | bool | — | if `true`, returns the build fingerprint and skips auth + model |

**Credential-free by design:** the client supplies both presigned URLs, so the
container holds no R2 secrets and the same image runs anywhere.

## Response (the small summary)

```json
{
  "status": "ok",
  "num_processed_frames": 724,
  "num_source_frames": 724,
  "source_fps": 24.0,
  "frame_stride": 1,
  "num_keypoints": 308,
  "model_size": "5b",
  "sapiens_model_revision": "<40-hex>",
  "person_detection": {
    "enabled": true,
    "stride": 5,
    "box_overflow": 0.25,
    "threshold": 0.3,
    "detection_frames": 145,
    "person_detections": 145,
    "selected_track_observations": 145,
    "fallback_full_frame": false,
    "seconds": 2.1
  },
  "infer_seconds": 64.9,
  "fps_processed": 11.15,
  "git_sha": "…",
  "results_location": "https://<r2>/out/kp.json"   // when result_put_url given
  // "results": { …full keypoints… }                // inline, when it wasn't
}
```

## Results JSON (written to R2, or inline under `results`)

```json
{
  "keypoint_format": "sapiens2_keypoints308",
  "git_sha": "<40-hex>",
  "model_size": "5b",
  "sapiens_model_revision": "<40-hex>",
  "person_detector": "facebook/detr-resnet-101-dc5",
  "person_detector_revision": "<40-hex>",
  "num_keypoints": 308,
  "source_fps": 24.0,
  "frame_stride": 1,
  "num_source_frames": 724,
  "num_processed_frames": 724,
  "person_detection": { "enabled": true, "stride": 5, "box_overflow": 0.25, "threshold": 0.3, "fallback_full_frame": false },
  "frames": [
    {
      "frame_idx": 0,
      "person_bbox": [x1, y1, x2, y2],
      "keypoints": [[x, y, conf], … 308 rows]
    },
    …
  ]
}
```
- `keypoints[k] = [x, y, conf]` in **original video pixel coordinates**.
- `person_bbox` is present only on frames supported by the selected track. It is
  the raw interpolated detector box before overflow and 3:4 aspect correction.
  Frames outside that span use the original full-frame transform and omit it.
- 308 = Sociopticon whole-body: body + 6 feet + 2×21 hands + dense face.

### Key body indices (compact Goliath/Sociopticon scheme)
This is **not COCO-17 after the elbows**. The hand roots also serve as wrists.

| joint | left | right |
|-------|------|-------|
| shoulder | 5 | 6 |
| elbow | 7 | 8 |
| wrist / hand root | 62 | 41 |
| hip | 9 | 10 |
| knee | 11 | 12 |
| ankle | 13 | 14 |

Feet are left big toe/small toe/heel `15/16/17` and right `18/19/20`.
Extra elbow/acromion/neck points are `63–69`; dense face points start at `70`.

### Key hand-keypoint indices (compact scheme the model outputs)
The hand joints are indices **21–62**, NOT 92–132. Fingers are `tip → … → base`.

| | thumb-tip | index-tip | middle-tip | ring-tip | pinky-tip | wrist |
|-|-----------|-----------|------------|----------|-----------|-------|
| **left**  | 42 | 46 | 50 | 54 | 58 | 62 |
| **right** | 21 | 25 | 29 | 33 | 37 | 41 |

(e.g. thumb-index tap distance = `dist(kp[42], kp[46])` for the left hand.)

---

## Notes / current characteristics

- **Historical warm throughput (H100, 5B, compiled):** ~11 fps → a 30 s /
  724-frame clip in ~65 s. Eager (no compile) was ~8.6 fps.
- **Warm throughput (RTX 5080, 5B, compiled, batch 1):** ~2.6 fps with 9.84 GiB
  peak VRAM. A 146-frame canary took 65.2 s including an 8.7 s cached first
  forward after process restart.
- **Production latency profile:** one active batch-1 worker, eager inference,
  model preload + first-forward warmup before RunPod readiness. Throughput is
  intentionally lower so the worker can stay resident on a cheaper 24 GB GPU.
- **Cold start:** do not attach the legacy RunPod Model Store entry. In the
  2026-08-17 production rollout it remained in "initializing model files" for
  more than 20 minutes without starting the container. Direct Hugging Face
  download fetched the 20.48 GB checkpoint in 53.6 seconds. FlashBoot reduces
  recovery time; one active worker removes normal scale-to-zero cold starts.
- **Env vars:** `MODEL_SIZE` (default `5b`), `WEIGHTS_DIR` (`/weights`),
  `RUNPOD_HF_CACHE` (`/runpod-volume/huggingface-cache/hub`),
  `BATCH_SIZE` (`1`), `COMPILE` (`0`=default eager, `1`=Inductor compile),
  `MODEL_LOAD_DEVICE` (`cpu`), `PRELOAD_MODEL` (`1`),
  `WARMUP_ON_START` (`1`),
  `JOB_STATE_DIR` (persistent dedicated-worker status directory),
  `SAPIENS_MODEL_REVISION` (pinned pose-checkpoint commit),
  `PERSON_DETECTOR_MODEL` (baked path by default),
  `PERSON_DETECTOR_REVISION` (baked checkpoint commit),
  `PERSON_DETECTION_BATCH_SIZE` (`8`), `API_KEY_SHA256`, `GIT_SHA`.
- If person detection finds no usable track, inference safely falls back to the
  original full-frame transform and reports `fallback_full_frame: true`.
- **Long videos:** pass `result_put_url` so the (potentially large) keypoint JSON
  goes to R2 instead of the HTTP response.
