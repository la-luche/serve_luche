"""Batched GPU preprocessing — replaces the per-frame CPU mmpose pipeline.

For a full-frame bbox on a fixed-size video the top-down UDP affine is a single
constant 2x3 matrix. We get it from sapiens' OWN functions (no reimplementation),
then apply it to a whole batch with kornia.warp_affine on the GPU and normalize —
replacing the 724 serial cv2.warpAffine calls that were ~87% of wall time.

codec.decode + the (input_size, bbox_center, bbox_scale) back-mapping are unchanged.
decord yields RGB and the model wants RGB (reference does bgr_to_rgb on cv2's BGR),
so no channel flip; mean/std are in RGB order.
"""
from __future__ import annotations

import numpy as np
import torch
from kornia.geometry.transform import warp_affine

from sapiens.pose.src.datasets.transforms.bbox_transforms import (
    bbox_xyxy2cs,
    get_udp_warp_matrix,
)

INPUT_W, INPUT_H = 768, 1024
MEAN = (123.675, 116.28, 103.53)  # RGB
STD = (58.395, 57.12, 57.375)


def _fix_aspect(scale: np.ndarray, ar: float) -> np.ndarray:
    w, h = float(scale[0]), float(scale[1])
    return np.array([w, w / ar] if w > h * ar else [h * ar, h], dtype=np.float32)


class GpuPreprocessor:
    def __init__(self, h_img: int, w_img: int, device: str, dtype: torch.dtype):
        bbox = np.array([0, 0, w_img - 1, h_img - 1], dtype=np.float32)
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        scale = _fix_aspect(scale, INPUT_W / INPUT_H)
        M = get_udp_warp_matrix(center, scale, 0.0, (INPUT_W, INPUT_H))  # (2,3)
        self.M = torch.as_tensor(M, dtype=torch.float32, device=device)[None]  # (1,2,3)
        self.meta = {
            "input_size": (INPUT_W, INPUT_H),
            "bbox_center": tuple(center.tolist()),
            "bbox_scale": tuple(scale.tolist()),
        }
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B,3,H,W) uint8 RGB on device
        x = frames.float()
        M = self.M.expand(x.shape[0], -1, -1)
        x = warp_affine(x, M, (INPUT_H, INPUT_W), mode="bilinear")  # (B,3,1024,768)
        return ((x - self.mean) / self.std).to(self.dtype)
