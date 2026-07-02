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
from .decode import GpuDecoder
from .log import log
from .model import DEVICE, FORWARD, MODEL
from .preprocess import GpuPreprocessor
from .video import Decoder, fetch_to_local

DEFAULT_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
KEYPOINT_FORMAT = "sapiens2_keypoints308"

# GPU heatmap->keypoint decoder (DARK-UDP on GPU). Falls back to the CPU codec on
# non-cuda (e.g. local mac tests).
_DECODER = (
    GpuDecoder(MODEL.codec.blur_kernel_size, MODEL.codec.input_size, DEVICE)
    if DEVICE == "cuda" else None
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
    log(f"run_video stride={frame_stride} batch={batch_size} — fetching video...")
    local_path, is_temp = fetch_to_local(video_url)
    log(f"video fetched in {time.time() - t0:.1f}s")
    try:
        dec = Decoder(local_path)
        pre = GpuPreprocessor(dec.height, dec.width, DEVICE, dtype=torch.bfloat16)
        input_size = np.asarray(pre.meta["input_size"], dtype=np.float32)
        bbox_center = np.asarray(pre.meta["bbox_center"], dtype=np.float32)
        bbox_scale = np.asarray(pre.meta["bbox_scale"], dtype=np.float32)

        cuda = DEVICE == "cuda"
        nbatches = (len(range(0, dec.num_frames, max(1, frame_stride))) + batch_size - 1) // batch_size
        log(f"video {dec.width}x{dec.height} {dec.fps:.1f}fps {dec.num_frames} frames | "
            f"stride={frame_stride} batch={batch_size} -> {nbatches} batches")

        t_dec = t_pre = t_fwd = t_codec = 0.0
        frames_out: list[dict] = []
        n = 0
        tprev = time.time()
        for bi, (indices, frames) in enumerate(dec.iter_batches(frame_stride, batch_size, DEVICE)):
            t_dec += time.time() - tprev  # decode + H2D (iterator produced this batch)

            t = time.time()
            x = pre(frames)  # (B,3,1024,768) bf16 on GPU (model is bf16 too)
            real = x.shape[0]
            if real < batch_size:
                # pad last partial batch so FORWARD always sees ONE shape
                pad = x[-1:].expand(batch_size - real, -1, -1, -1)
                x = torch.cat([x, pad], dim=0)
            if cuda:
                torch.cuda.synchronize()
            t_pre += time.time() - t

            t = time.time()
            with torch.inference_mode():
                pred = FORWARD(x)[:real].float()  # (real,K,hH,hW) on GPU
            if cuda:
                torch.cuda.synchronize()
            t_fwd += time.time() - t

            t = time.time()
            if _DECODER is not None:
                coords, scores = _DECODER.decode(pred)  # GPU DARK-UDP, (real,K,2),(real,K)
            else:  # cpu fallback
                pn = pred.cpu().numpy()
                cs = [MODEL.codec.decode(pn[j]) for j in range(pn.shape[0])]
                coords = np.stack([c[0][0] for c in cs]); scores = np.stack([c[1][0] for c in cs])
            # input-space -> original pixels (vectorized), then round
            coords = coords / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
            coords = np.round(coords, coord_round)
            scores = np.round(scores, conf_round)
            for j, fidx in enumerate(indices):
                ck, sk = coords[j], scores[j]
                frames_out.append({
                    "frame_idx": int(fidx),
                    "keypoints": [[float(ck[k, 0]), float(ck[k, 1]), float(sk[k])]
                                  for k in range(ck.shape[0])],
                })
            t_codec += time.time() - t
            n += len(indices)
            if bi == 0 or (bi + 1) % 10 == 0 or bi == nbatches - 1:
                el = time.time() - t0
                log(f"batch {bi+1}/{nbatches} frame {n}/{dec.num_frames} | "
                    f"decode {t_dec:.1f} warp {t_pre:.1f} fwd {t_fwd:.1f} codec {t_codec:.1f}s "
                    f"| {n/el:.1f} fps")
            tprev = time.time()

        if cuda:
            torch.cuda.synchronize()
        el = time.time() - t0
        log(f"DONE {n} frames in {el:.1f}s = {n/el:.1f} fps | stages: "
            f"decode {t_dec:.1f}s ({100*t_dec/el:.0f}%) warp {t_pre:.1f}s ({100*t_pre/el:.0f}%) "
            f"fwd {t_fwd:.1f}s ({100*t_fwd/el:.0f}%) codec {t_codec:.1f}s ({100*t_codec/el:.0f}%)")

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
