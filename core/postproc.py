"""Preprocess frames → model input, and decode heatmaps → keypoints.

Single-person / full-frame: the "bounding box" is the whole frame, so we
letterbox each frame (aspect-preserving resize + symmetric pad) to 1024x768 and
record the affine so keypoints can be mapped back to original pixel coordinates.
All frames in one video share the same original size, so the affine is computed
once.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import (
    MODEL_INPUT_H,
    MODEL_INPUT_W,
    NORM_MEAN,
    NORM_STD,
)

# Sapiens' TorchScript demo feeds RGB. If keypoints come out mirrored/garbage in
# the step-2 validation, flip this to True (BGR) — it's the #1 thing to check.
INPUT_BGR = False


@dataclass(frozen=True)
class Affine:
    scale: float
    pad_left: float
    pad_top: float


def _norm_tensors(device, dtype):
    mean = list(NORM_MEAN)
    std = list(NORM_STD)
    if INPUT_BGR:
        mean = mean[::-1]
        std = std[::-1]
    m = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    return m, s


def compute_affine(orig_h: int, orig_w: int) -> Affine:
    scale = min(MODEL_INPUT_H / orig_h, MODEL_INPUT_W / orig_w)
    new_h = round(orig_h * scale)
    new_w = round(orig_w * scale)
    pad_top = (MODEL_INPUT_H - new_h) / 2.0
    pad_left = (MODEL_INPUT_W - new_w) / 2.0
    return Affine(scale=scale, pad_left=pad_left, pad_top=pad_top)


def preprocess(frames_u8: torch.Tensor, affine: Affine, out_dtype) -> torch.Tensor:
    """(B,C,H,W) uint8 -> (B,3,1024,768) normalized, in `out_dtype`."""
    x = frames_u8.float()
    if INPUT_BGR:
        x = x.flip(1)  # RGB -> BGR

    new_h = round(x.shape[2] * affine.scale)
    new_w = round(x.shape[3] * affine.scale)
    x = F.interpolate(
        x, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    pt = int(round(affine.pad_top))
    pl = int(round(affine.pad_left))
    pb = MODEL_INPUT_H - new_h - pt
    pr = MODEL_INPUT_W - new_w - pl
    x = F.pad(x, (pl, pr, pt, pb), value=0.0)  # (left,right,top,bottom)

    m, s = _norm_tensors(x.device, x.dtype)
    x = (x - m) / s
    return x.to(out_dtype)


@torch.inference_mode()
def decode_keypoints(heatmaps: torch.Tensor, affine: Affine):
    """(B,K,hm_h,hm_w) heatmaps -> (coords (B,K,2) in ORIGINAL px, conf (B,K)).

    argmax + a quarter-pixel Darkpose-style shift toward the higher neighbor.
    """
    B, K, hh, hw = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    conf, idx = flat.max(dim=2)
    hm_x = (idx % hw).float()
    hm_y = (idx // hw).float()

    # Quarter-pixel sub-pixel refinement using neighbor gradient sign.
    xi = idx % hw
    yi = idx // hw
    x0 = (xi - 1).clamp(0, hw - 1)
    x1 = (xi + 1).clamp(0, hw - 1)
    y0 = (yi - 1).clamp(0, hh - 1)
    y1 = (yi + 1).clamp(0, hh - 1)
    bk = torch.arange(B, device=heatmaps.device).view(B, 1)
    kk = torch.arange(K, device=heatmaps.device).view(1, K)
    dx = heatmaps[bk, kk, yi, x1] - heatmaps[bk, kk, yi, x0]
    dy = heatmaps[bk, kk, y1, xi] - heatmaps[bk, kk, y0, xi]
    hm_x = hm_x + torch.sign(dx) * 0.25
    hm_y = hm_y + torch.sign(dy) * 0.25

    # heatmap space -> model-input space
    in_x = hm_x * (MODEL_INPUT_W / hw)
    in_y = hm_y * (MODEL_INPUT_H / hh)

    # model-input space -> original pixel space (invert the letterbox)
    orig_x = (in_x - affine.pad_left) / affine.scale
    orig_y = (in_y - affine.pad_top) / affine.scale

    coords = torch.stack([orig_x, orig_y], dim=2)  # (B,K,2)
    return coords, conf
