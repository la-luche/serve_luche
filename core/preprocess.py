"""Batched GPU preprocessing — replaces the per-frame CPU mmpose pipeline.

For a full-frame bbox on a fixed-size video the top-down UDP affine is a single
constant 2x3 matrix. With optional person detection it becomes one matrix per
frame. We get both from sapiens' OWN functions (no reimplementation), then apply
them to a whole batch with kornia.warp_affine on the GPU and normalize.

codec.decode + the (input_size, bbox_center, bbox_scale) back-mapping are unchanged.
decord yields RGB and the model wants RGB (reference does bgr_to_rgb on cv2's BGR),
so no channel flip; mean/std are in RGB order.
"""
from __future__ import annotations

import numpy as np
import torch
from kornia.geometry.transform import warp_affine

# NOTE: sapiens/pose/__init__.py merges the physical `src/` dir into the
# `sapiens.pose` namespace, so the canonical path drops `src` (matching how
# `from sapiens.pose.datasets import UDPHeatmap` works).
from sapiens.pose.datasets.transforms.bbox_transforms import (
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
        self.full_M = torch.as_tensor(M, dtype=torch.float32, device=device)[None]
        self.full_center = np.asarray(center, dtype=np.float32)
        self.full_scale = np.asarray(scale, dtype=np.float32)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(
        self,
        frames: torch.Tensor,
        bboxes: np.ndarray | None = None,
        bbox_overflow: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        # frames: (B,3,H,W) uint8 RGB on device
        x = frames.float()
        if bboxes is None:
            M = self.full_M.expand(x.shape[0], -1, -1)
            centers = np.repeat(self.full_center[None], x.shape[0], axis=0)
            scales = np.repeat(self.full_scale[None], x.shape[0], axis=0)
        else:
            if len(bboxes) != x.shape[0]:
                raise ValueError(
                    f"got {len(bboxes)} boxes for {x.shape[0]} input frames"
                )
            centers_list, scales_list, matrices = [], [], []
            # overflow is the fraction added on EACH side: 0.25 -> 1.5x box.
            padding = 1.0 + 2.0 * bbox_overflow
            for bbox in np.asarray(bboxes, dtype=np.float32):
                center, scale = bbox_xyxy2cs(bbox, padding=padding)
                scale = _fix_aspect(scale, INPUT_W / INPUT_H)
                centers_list.append(center)
                scales_list.append(scale)
                matrices.append(
                    get_udp_warp_matrix(
                        center, scale, 0.0, (INPUT_W, INPUT_H)
                    )
                )
            centers = np.asarray(centers_list, dtype=np.float32)
            scales = np.asarray(scales_list, dtype=np.float32)
            M = torch.as_tensor(
                np.asarray(matrices), dtype=torch.float32, device=self.device
            )
        # align_corners=True to match cv2.warpAffine (validated: mean diff 0.08/255)
        x = warp_affine(x, M, (INPUT_H, INPUT_W), mode="bilinear", align_corners=True)
        normalized = ((x - self.mean) / self.std).to(self.dtype)
        return normalized, {
            "input_size": np.asarray((INPUT_W, INPUT_H), dtype=np.float32),
            "bbox_center": centers,
            "bbox_scale": scales,
        }
