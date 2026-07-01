"""run_video(): decode every frame -> Sapiens2 pose -> 308 keypoints -> R2.

Single-person / full-frame: bbox = whole frame. Follows sapiens2's
`process_one_image` (model.pipeline -> data_preprocessor -> model(inputs) ->
codec.decode -> map back to original pixels via bbox meta), batched across frames
for GPU throughput. Keypoints land in ORIGINAL video pixel coordinates.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from . import sink
from .model import DEVICE, FORWARD, MODEL
from .video import Decoder, fetch_to_local

DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
KEYPOINT_FORMAT = "sapiens2_keypoints308"


def _prep(frame_np: np.ndarray):
    h, w = frame_np.shape[:2]
    data_info = {
        "img": frame_np,
        "bbox": np.array([[0, 0, w - 1, h - 1]], dtype=np.float32),
        "bbox_score": np.ones(1, dtype=np.float32),
    }
    data = MODEL.pipeline(data_info)
    data = MODEL.data_preprocessor(data)
    return data["inputs"], data["data_samples"]


def _meta(ds):
    m = ds["meta"] if isinstance(ds, dict) and "meta" in ds else ds
    return (
        np.asarray(m["input_size"], dtype=np.float32),
        np.asarray(m["bbox_center"], dtype=np.float32),
        np.asarray(m["bbox_scale"], dtype=np.float32),
    )


def run_video(
    video_url: str,
    *,
    result_put_url: str | None = None,
    result_local_path: str | None = None,
    frame_stride: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    coord_round: int = 2,
    conf_round: int = 3,
) -> dict:
    t0 = time.time()
    local_path, is_temp = fetch_to_local(video_url)
    try:
        dec = Decoder(local_path)
        frames_out: list[dict] = []
        n = 0

        for indices, frames in dec.iter_batches_np(frame_stride, batch_size):
            inputs_list, samples = [], []
            for f in frames:
                inp, ds = _prep(f)
                inputs_list.append(inp)
                samples.append(ds)
            inputs = torch.cat(inputs_list, dim=0).to(DEVICE, dtype=torch.bfloat16)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = FORWARD(inputs)
            pred = pred.float().cpu().numpy()  # B x K x hH x hW

            for j, fidx in enumerate(indices):
                kps, scores = MODEL.codec.decode(pred[j])  # (1,K,2), (1,K) in input space
                input_size, bbox_center, bbox_scale = _meta(samples[j])
                kps = kps / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
                kps, scores = kps[0], scores[0]
                frames_out.append({
                    "frame_idx": int(fidx),
                    "keypoints": [
                        [round(float(kps[k][0]), coord_round),
                         round(float(kps[k][1]), coord_round),
                         round(float(scores[k]), conf_round)]
                        for k in range(len(scores))
                    ],
                })
            n += len(indices)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        results = {
            "keypoint_format": KEYPOINT_FORMAT,
            "num_keypoints": len(frames_out[0]["keypoints"]) if frames_out else 0,
            "source_fps": dec.fps,
            "frame_stride": frame_stride,
            "num_source_frames": dec.num_frames,
            "num_processed_frames": n,
            "frames": frames_out,
        }
        location = sink.emit(results, put_url=result_put_url, local_path=result_local_path)

        elapsed = round(time.time() - t0, 2)
        return {
            "status": "ok",
            "results_location": location,
            "num_processed_frames": n,
            "num_source_frames": dec.num_frames,
            "source_fps": dec.fps,
            "frame_stride": frame_stride,
            "num_keypoints": results["num_keypoints"],
            "model_size": os.environ.get("MODEL_SIZE", "5b"),
            "infer_seconds": elapsed,
            "fps_processed": round(n / elapsed, 2) if elapsed else None,
        }
    finally:
        if is_temp:
            try:
                os.remove(local_path)
            except OSError:
                pass
