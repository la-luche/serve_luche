"""run_video(): decode -> batched GPU warp -> Sapiens2 -> 308 keypoints -> R2.

Single-person / full-frame. Preprocessing (top-down UDP affine + normalize) is done
batched on the GPU (core.preprocess), NOT the per-frame CPU mmpose pipeline — that
was ~87% of wall time. codec.decode + the constant (input_size, bbox_center,
bbox_scale) back-mapping reproduce the reference exactly. Keypoints land in ORIGINAL
video pixel coordinates.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from . import sink
from .model import DEVICE, FORWARD, MODEL, USE_AUTOCAST
from .preprocess import GpuPreprocessor
from .video import Decoder, fetch_to_local

DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
KEYPOINT_FORMAT = "sapiens2_keypoints308"


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
        pre = GpuPreprocessor(dec.height, dec.width, DEVICE, dtype=torch.bfloat16)
        input_size = np.asarray(pre.meta["input_size"], dtype=np.float32)
        bbox_center = np.asarray(pre.meta["bbox_center"], dtype=np.float32)
        bbox_scale = np.asarray(pre.meta["bbox_scale"], dtype=np.float32)

        frames_out: list[dict] = []
        n = 0
        for indices, frames in dec.iter_batches(frame_stride, batch_size, DEVICE):
            x = pre(frames)  # (B,3,1024,768) bf16 on GPU
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=USE_AUTOCAST
            ):
                pred = FORWARD(x)
            pred = pred.float().cpu().numpy()  # B x K x hH x hW

            for j, fidx in enumerate(indices):
                kps, scores = MODEL.codec.decode(pred[j])  # (1,K,2),(1,K) input space
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

        elapsed = round(time.time() - t0, 2)
        summary = {
            "status": "ok",
            "num_processed_frames": n,
            "num_source_frames": dec.num_frames,
            "source_fps": dec.fps,
            "frame_stride": frame_stride,
            "num_keypoints": results["num_keypoints"],
            "model_size": os.environ.get("MODEL_SIZE", "5b"),
            "infer_seconds": elapsed,
            "fps_processed": round(n / elapsed, 2) if elapsed else None,
        }
        if result_put_url or result_local_path:
            summary["results_location"] = sink.emit(
                results, put_url=result_put_url, local_path=result_local_path
            )
        else:
            summary["results"] = results
        return summary
    finally:
        if is_temp:
            try:
                os.remove(local_path)
            except OSError:
                pass
