"""Fetch a video and decode frames on the GPU (NVDEC via torchcodec).

Decode is usually the bottleneck for this pipeline, not the 2B forward pass, so
we (a) decode on-device to avoid H2D copies and (b) only decode the frames we
actually keep (`frame_stride`). Frames come out as uint8 (B, C, H, W) on DEVICE.
"""
from __future__ import annotations

import os
import tempfile
from typing import Iterator
from urllib.parse import urlparse

import requests
import torch

from .model import DEVICE


def fetch_to_local(video_url: str, chunk: int = 1 << 20) -> tuple[str, bool]:
    """Return (local_path, is_temp). Accepts http(s) presigned URLs or a path."""
    parsed = urlparse(video_url)
    if parsed.scheme in ("http", "https"):
        fd, path = tempfile.mkstemp(suffix=os.path.splitext(parsed.path)[1] or ".mp4")
        os.close(fd)
        with requests.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for block in r.iter_content(chunk_size=chunk):
                    f.write(block)
        return path, True
    if not os.path.isfile(video_url):
        raise FileNotFoundError(video_url)
    return video_url, False


class Decoder:
    """Thin wrapper over torchcodec giving strided, batched GPU frame tensors."""

    def __init__(self, local_path: str):
        try:
            from torchcodec.decoders import VideoDecoder
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "torchcodec is required for decoding. `pip install torchcodec` "
                "and ensure ffmpeg is installed."
            ) from e

        # Fall back to CPU decode if the GPU decoder can't init (e.g. no NVDEC).
        try:
            self._dec = VideoDecoder(local_path, device=DEVICE)
        except Exception:
            self._dec = VideoDecoder(local_path, device="cpu")

        md = self._dec.metadata
        self.fps = float(getattr(md, "average_fps", None) or 30.0)
        self.num_frames = int(getattr(md, "num_frames", 0) or len(self._dec))

    def iter_batches(
        self, frame_stride: int, batch_size: int
    ) -> Iterator[tuple[list[int], torch.Tensor]]:
        """Yield (frame_indices, frames uint8 (B,C,H,W) on DEVICE) in order."""
        indices = list(range(0, self.num_frames, max(1, frame_stride)))
        for i in range(0, len(indices), batch_size):
            chunk = indices[i : i + batch_size]
            batch = self._dec.get_frames_at(indices=chunk)  # FrameBatch
            frames = batch.data.to(DEVICE, non_blocking=True)  # (B,C,H,W) uint8
            yield chunk, frames
