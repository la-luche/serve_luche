"""GPU heatmap -> keypoint decode (DARK-UDP), batched.

Faithful GPU port of sapiens2's get_heatmap_maximum + refine_keypoints_dark_udp
(argmax -> gaussian-blur modulate -> log -> Taylor sub-pixel offset). The CPU
codec ran this per-frame in numpy and was ~78% of wall time; on GPU (heatmaps
are already there) it's a batched conv + gather, ~100x cheaper.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _cv2_gaussian_1d(ks: int, device) -> torch.Tensor:
    # matches cv2.GaussianBlur((ks,ks), sigmaX=0): sigma derived from ks
    sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
    x = torch.arange(ks, device=device, dtype=torch.float32) - (ks - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    return g / g.sum()


class GpuDecoder:
    def __init__(self, blur_kernel_size: int, input_size, device: str):
        self.ks = int(blur_kernel_size)
        self.pad = (self.ks - 1) // 2
        self.in_w, self.in_h = float(input_size[0]), float(input_size[1])  # (768,1024)
        k = _cv2_gaussian_1d(self.ks, device)
        self.kx = k.view(1, 1, 1, self.ks)
        self.ky = k.view(1, 1, self.ks, 1)

    @torch.inference_mode()
    def decode(self, hm: torch.Tensor):
        """hm: (B,K,H,W) float on GPU -> (coords (B,K,2), scores (B,K)) numpy."""
        B, K, H, W = hm.shape
        flat = hm.reshape(B, K, -1)
        vals, idx = flat.max(dim=2)
        x = (idx % W).to(torch.float32)
        y = (idx // W).to(torch.float32)

        # modulate: separable gaussian blur with per-map max preservation
        h = hm.reshape(B * K, 1, H, W)
        omax = h.amax(dim=(2, 3), keepdim=True)
        h = F.conv2d(F.pad(h, (self.pad, self.pad, 0, 0), mode="replicate"), self.kx)
        h = F.conv2d(F.pad(h, (0, 0, self.pad, self.pad), mode="replicate"), self.ky)
        bmax = h.amax(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        h = (h * (omax / bmax)).reshape(B, K, H, W)
        h = h.clamp(1e-3, 50.0).log()

        # pad edge by 1 and gather the 3x3 neighborhood at each peak
        p = F.pad(h, (1, 1, 1, 1), mode="replicate")  # (B,K,H+2,W+2)
        bi = torch.arange(B, device=hm.device)[:, None]
        ki = torch.arange(K, device=hm.device)[None, :]
        xi = x.long() + 1
        yi = y.long() + 1

        def g(px, py):
            return p[bi, ki, py, px]

        i0 = g(xi, yi)
        ixp = g(xi + 1, yi); ixm = g(xi - 1, yi)
        iyp = g(xi, yi + 1); iym = g(xi, yi - 1)
        ixpyp = g(xi + 1, yi + 1); ixmym = g(xi - 1, yi - 1)

        dx = 0.5 * (ixp - ixm)
        dy = 0.5 * (iyp - iym)
        dxx = ixp - 2 * i0 + ixm
        dyy = iyp - 2 * i0 + iym
        dxy = 0.5 * (ixpyp - ixp - iyp + 2 * i0 - ixm - iym + ixmym)

        eps = torch.finfo(torch.float32).eps
        a = dxx + eps; d = dyy + eps; b = dxy
        det = a * d - b * b
        det = torch.where(det.abs() < eps, torch.full_like(det, eps), det)
        ox = (d * dx - b * dy) / det   # inv(H) @ [dx,dy]
        oy = (-b * dx + a * dy) / det
        # DARK refinement is sub-pixel; clamp so flat/near-singular heatmaps (garbage
        # low-conf keypoints) don't fly off to absurd coords (the CPU codec doesn't
        # clamp and produces huge values there; this is strictly saner, and confident
        # keypoints match to <0.03 px either way).
        ox = ox.clamp(-1.0, 1.0)
        oy = oy.clamp(-1.0, 1.0)
        x = x - ox
        y = y - oy

        # heatmap space -> input space (matches codec: coords/[W-1,H-1]*input_size)
        x = x / (W - 1) * self.in_w
        y = y / (H - 1) * self.in_h

        coords = torch.stack([x, y], dim=2)  # (B,K,2)
        return coords.cpu().numpy(), vals.cpu().numpy()
