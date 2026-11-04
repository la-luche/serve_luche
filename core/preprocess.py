"""Batched GPU preprocessing — replaces the per-frame CPU mmpose pipeline.

The default crop never extends outside the source image. Requested context and
3:4 aspect expansion are clipped to real image pixels, then resized to the
model input. Edge crops may therefore be resized differently along x and y.
``legacy`` retains the old out-of-frame, zero-filled geometry for comparisons.

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
    def __init__(
        self, h_img: int, w_img: int, device: str, dtype: torch.dtype,
        *, crop_policy: str = "in_frame",
    ):
        if crop_policy not in {"in_frame", "legacy"}:
            raise ValueError("crop_policy must be in_frame or legacy")
        if min(h_img, w_img) < 2:
            raise ValueError("source dimensions must be at least 2 pixels")
        self.crop_policy = crop_policy
        self.source_max = np.asarray([w_img - 1, h_img - 1], dtype=np.float32)
        bbox = np.array([0, 0, w_img - 1, h_img - 1], dtype=np.float32)
        if crop_policy == "legacy":
            center, scale = bbox_xyxy2cs(bbox, padding=1.25)
            scale = _fix_aspect(scale, INPUT_W / INPUT_H)
        else:
            center, scale = self.source_max * 0.5, self.source_max.copy()
        M = get_udp_warp_matrix(center, scale, 0.0, (INPUT_W, INPUT_H))  # (2,3)
        self.full_matrix = np.asarray(M, dtype=np.float32)
        self.full_M = torch.as_tensor(M, dtype=torch.float32, device=device)[None]
        self.full_center = np.asarray(center, dtype=np.float32)
        self.full_scale = np.asarray(scale, dtype=np.float32)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        self.device = device
        self.dtype = dtype

    def _person_geometry(self, bbox: np.ndarray, padding: float):
        if self.crop_policy == "in_frame":
            bbox = np.clip(bbox.reshape(2, 2), 0, self.source_max).reshape(4)
            if np.any(bbox[2:] <= bbox[:2]):
                return self.full_center, self.full_scale, self.full_matrix
        center, scale = bbox_xyxy2cs(bbox, padding=padding)
        scale = _fix_aspect(scale, INPUT_W / INPUT_H)
        if self.crop_policy == "in_frame":
            lower = np.maximum(center - 0.5 * scale, 0)
            upper = np.minimum(center + 0.5 * scale, self.source_max)
            center, scale = (lower + upper) * 0.5, upper - lower
        matrix = get_udp_warp_matrix(center, scale, 0.0, (INPUT_W, INPUT_H))
        return center, scale, matrix

    def geometry(
        self,
        batch_size: int,
        bboxes: np.ndarray | None = None,
        bbox_overflow: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """Return per-frame affine matrices and their inverse-map metadata."""
        if not np.isfinite(bbox_overflow) or bbox_overflow < 0:
            raise ValueError("bbox_overflow must be finite and nonnegative")
        if bboxes is None:
            M = self.full_M.expand(batch_size, -1, -1)
            centers = np.repeat(self.full_center[None], batch_size, axis=0)
            scales = np.repeat(self.full_scale[None], batch_size, axis=0)
        else:
            if len(bboxes) != batch_size:
                raise ValueError(
                    f"got {len(bboxes)} boxes for {batch_size} input frames"
                )
            centers_list, scales_list, matrices = [], [], []
            # overflow is the fraction added on EACH side: 0.25 -> 1.5x box.
            padding = 1.0 + 2.0 * bbox_overflow
            for bbox in np.asarray(bboxes, dtype=np.float32):
                valid = (
                    np.isfinite(bbox).all()
                    and bbox[2] > bbox[0]
                    and bbox[3] > bbox[1]
                )
                if valid:
                    center, scale, matrix = self._person_geometry(bbox, padding)
                else:
                    center, scale, matrix = (
                        self.full_center,
                        self.full_scale,
                        self.full_matrix,
                    )
                centers_list.append(center)
                scales_list.append(scale)
                matrices.append(matrix)
            centers = np.asarray(centers_list, dtype=np.float32)
            scales = np.asarray(scales_list, dtype=np.float32)
            M = torch.as_tensor(
                np.asarray(matrices), dtype=torch.float32, device=self.device
            )
        return M, {
            "input_size": np.asarray((INPUT_W, INPUT_H), dtype=np.float32),
            "bbox_center": centers,
            "bbox_scale": scales,
            "crop_policy": self.crop_policy,
        }

    @torch.inference_mode()
    def __call__(
        self,
        frames: torch.Tensor,
        bboxes: np.ndarray | None = None,
        bbox_overflow: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        # frames: (B,3,H,W) uint8 RGB on device
        x = frames.float()
        M, meta = self.geometry(x.shape[0], bboxes, bbox_overflow)
        # align_corners=True to match cv2.warpAffine (validated: mean diff 0.08/255)
        # All in_frame sampling coordinates are inside the source. Border mode
        # only guards floating-point roundoff at the outermost source pixel.
        x = warp_affine(
            x, M, (INPUT_H, INPUT_W), mode="bilinear", align_corners=True,
            padding_mode="border" if self.crop_policy == "in_frame" else "zeros",
        )
        normalized = ((x - self.mean) / self.std).to(self.dtype)
        return normalized, meta
