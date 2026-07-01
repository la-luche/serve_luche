"""Fetch a video and decode strided, batched frames via decord.

We use decord (CPU decode + H2D copy) rather than torchcodec/NVDEC: torchcodec's
wheels are fragile (CUDA-version-coupled, and some ship ops for only one backend
-> `NotImplementedError: ... scan_all_streams_to_update_metadata ... CPU`).
decord is a single portable wheel that Just Works. Decode is not the bottleneck
for the heavy Sapiens forward pass, so CPU decode is an acceptable trade for
reliability. Frames come out as uint8 (B, C, H, W) RGB on DEVICE.
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
    """Thin wrapper over decord giving strided, batched frame tensors on DEVICE."""

    def __init__(self, local_path: str):
        try:
            from decord import VideoReader, cpu
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("decord is required for decoding: `pip install decord`") from e

        self._vr = VideoReader(local_path, ctx=cpu(0))
        self.fps = float(self._vr.get_avg_fps() or 30.0)
        self.num_frames = len(self._vr)

    def iter_batches(
        self, frame_stride: int, batch_size: int
    ) -> Iterator[tuple[list[int], torch.Tensor]]:
        """Yield (frame_indices, frames uint8 (B,C,H,W) RGB on DEVICE) in order."""
        indices = list(range(0, self.num_frames, max(1, frame_stride)))
        for i in range(0, len(indices), batch_size):
            chunk = indices[i : i + batch_size]
            arr = self._vr.get_batch(chunk).asnumpy()  # (B,H,W,C) uint8 RGB
            frames = (
                torch.from_numpy(arr)
                .permute(0, 3, 1, 2)
                .contiguous()
                .to(DEVICE, non_blocking=True)
            )
            yield chunk, frames
