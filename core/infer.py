"""The single entrypoint every serverless adapter calls: run_video().

Downloads a video (presigned R2 GET), decodes strided frames on GPU, runs the
Sapiens pose model in batches, decodes keypoints back to original pixels, writes
the full per-frame keypoint JSON to R2 (presigned PUT), and returns a small
summary dict (the big array went to R2, not through the HTTP response).
"""
from __future__ import annotations

import os
import time

import torch

from . import sink
from .model import DEVICE, MODEL, MODEL_DTYPE, USE_AUTOCAST, WEIGHTS_FILE
from .postproc import compute_affine, decode_keypoints, preprocess
from .video import Decoder, fetch_to_local

DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
KEYPOINT_FORMAT = "goliath_308"


def _forward(x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=USE_AUTOCAST
    ):
        return MODEL(x).float()


def run_video(
    video_url: str,
    *,
    result_put_url: str | None = None,
    result_local_path: str | None = None,
    frame_stride: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_conf: bool = True,
    coord_round: int = 2,
    conf_round: int = 3,
) -> dict:
    t0 = time.time()
    local_path, is_temp = fetch_to_local(video_url)
    try:
        dec = Decoder(local_path)
        affine = None
        frames_out: list[dict] = []
        n_processed = 0

        for indices, frames_u8 in dec.iter_batches(frame_stride, batch_size):
            if affine is None:
                affine = compute_affine(frames_u8.shape[2], frames_u8.shape[3])
            x = preprocess(frames_u8, affine, MODEL_DTYPE if MODEL_DTYPE != torch.float32 else torch.float32)
            heatmaps = _forward(x)
            coords, conf = decode_keypoints(heatmaps, affine)
            coords = coords.cpu().tolist()
            conf = conf.cpu().tolist()

            for j, fidx in enumerate(indices):
                if include_conf:
                    kps = [
                        [round(coords[j][k][0], coord_round),
                         round(coords[j][k][1], coord_round),
                         round(conf[j][k], conf_round)]
                        for k in range(len(conf[j]))
                    ]
                else:
                    kps = [
                        [round(coords[j][k][0], coord_round),
                         round(coords[j][k][1], coord_round)]
                        for k in range(len(conf[j]))
                    ]
                frames_out.append({"frame_idx": fidx, "keypoints": kps})
            n_processed += len(indices)

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        results = {
            "keypoint_format": KEYPOINT_FORMAT,
            "num_keypoints": len(frames_out[0]["keypoints"]) if frames_out else 0,
            "input_hw": [1024, 768],
            "source_fps": dec.fps,
            "frame_stride": frame_stride,
            "num_source_frames": dec.num_frames,
            "num_processed_frames": n_processed,
            "frames": frames_out,
        }

        location = sink.emit(
            results, put_url=result_put_url, local_path=result_local_path
        )

        elapsed = round(time.time() - t0, 2)
        return {
            "status": "ok",
            "results_location": location,
            "num_processed_frames": n_processed,
            "num_source_frames": dec.num_frames,
            "source_fps": dec.fps,
            "frame_stride": frame_stride,
            "num_keypoints": results["num_keypoints"],
            "batch_size": batch_size,
            "device": DEVICE,
            "weights": os.path.basename(WEIGHTS_FILE),
            "infer_seconds": elapsed,
            "fps_processed": round(n_processed / elapsed, 2) if elapsed else None,
        }
    finally:
        if is_temp:
            try:
                os.remove(local_path)
            except OSError:
                pass
