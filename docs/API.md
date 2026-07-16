# serve_luche API

Serverless whole-body keypoint detection. Send a video (as a presigned R2 GET
URL), get **308 Sapiens2 keypoints per frame** back — either inline (small clips)
or written to R2 (presigned PUT). Same image runs on **RunPod** and **Vast**.

- **Live RunPod endpoint:** `v29lgubwpc998d`
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

Current key hash (baked): `88edc437185f02bf774f458a8f3e3404d6b9e49c89ceb3393a31cd5579f9d446`
(the key is held out-of-band; rotate by regenerating and rebuilding the image.)

---

## RunPod (async)

Submit a job, poll for the result.

```bash
# submit
curl -X POST https://api.runpod.ai/v2/v29lgubwpc998d/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {
        "api_key": "sk_luche_…",
        "video_url": "https://<r2>/clip.mp4?X-Amz-…",        # presigned GET
        "result_put_url": "https://<r2>/out/kp.json?X-Amz-…", # presigned PUT (optional)
        "frame_stride": 1,
        "batch_size": 16,
        "person_detection": true,
        "person_detection_stride": 5,
        "person_box_overflow": 0.25,
        "person_detection_threshold": 0.3
  }}'
# -> {"id": "<job-id>", "status": "IN_QUEUE"}

# poll
curl https://api.runpod.ai/v2/v29lgubwpc998d/status/<job-id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### Version / health check (unauthenticated, no model load)
Use this to confirm which build a container is running:
```bash
curl -X POST https://api.runpod.ai/v2/v29lgubwpc998d/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"ping": true}}'
# -> {"status":"ok","git_sha":"<40-hex>","api_key_sha256":"88edc437…","model_size":"5b"}
```

## Vast (or any HTTP host running `adapters/vast_worker.py`)

```
GET  /healthz   -> {"status":"ok","git_sha":…,"api_key_sha256":…,"model_size":…}   (unauth)
POST /infer     -> body identical to RunPod's "input" object; requires api_key
```

---

## Request fields

| field | type | default | notes |
|-------|------|---------|-------|
| `api_key` | string | — | **required** (except `ping`); wrong → `unauthorized` |
| `video_url` | string | — | **required**; presigned R2 **GET** (or any http(s) URL / local path) |
| `result_put_url` | string | none | presigned R2 **PUT**; if omitted, keypoints are returned **inline** |
| `frame_stride` | int | `1` | process every Nth frame (`1` = every frame) |
| `batch_size` | int | `16` | frames per GPU batch |
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
- `person_bbox` is present only when detection selected a track. It is the raw
  interpolated detector box before overflow and 3:4 aspect correction.
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

- **Warm throughput (H100, 5B, compiled):** ~11 fps → a 30 s / 724-frame clip in
  ~65 s. Eager (no compile) ~8.6 fps.
- **Cold start** (fresh worker): image pull + model initialization + first-forward
  compilation can still take minutes. On RunPod, configure the endpoint Model as
  `facebook/sapiens2-pose-5b` to avoid downloading the 20 GB checkpoint in the
  worker. Use one active worker when predictable latency matters.
- **Env vars:** `MODEL_SIZE` (default `5b`), `WEIGHTS_DIR` (`/weights`),
  `RUNPOD_HF_CACHE` (`/runpod-volume/huggingface-cache/hub`),
  `BATCH_SIZE` (`16`), `COMPILE` (`1`=default Inductor `torch.compile`, `0`=eager),
  `PERSON_DETECTOR_MODEL` (baked path by default),
  `PERSON_DETECTION_BATCH_SIZE` (`8`), `API_KEY_SHA256`, `GIT_SHA`.
- If person detection finds no usable track, inference safely falls back to the
  original full-frame transform and reports `fallback_full_frame: true`.
- **Long videos:** pass `result_put_url` so the (potentially large) keypoint JSON
  goes to R2 instead of the HTTP response.
