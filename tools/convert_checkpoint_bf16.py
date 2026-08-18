"""Download the fp32 Sapiens2 pose checkpoint and write a bf16 copy.

Runs on a free GH runner (16 GB RAM): tensors are streamed one at a time from
the fp32 mmap, so peak memory is the bf16 result (~10 GB for 5b) plus one
fp32 tensor — never two full copies.

The production loader casts to bf16 anyway (core/model.py streams fp32 -> cuda
bf16), so baking bf16 is numerically identical to the current CUDA path.

Usage:
    python tools/convert_checkpoint_bf16.py --out weights/sapiens2_5b_pose.safetensors
"""
from __future__ import annotations

import argparse
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_SIZE = os.environ.get("MODEL_SIZE", "5b")
HF_REPO = f"facebook/sapiens2-pose-{MODEL_SIZE}"
CKPT_NAME = f"sapiens2_{MODEL_SIZE}_pose.safetensors"
REVISION = os.environ.get(
    "SAPIENS_MODEL_REVISION", "ada1f29aa1fd454ca28665c700923a0101b6b24f"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", default="/tmp/fp32-src")
    args = parser.parse_args()

    print(f"downloading {CKPT_NAME} from {HF_REPO}@{REVISION} (fp32, ~20 GB for 5b)")
    src = hf_hub_download(
        repo_id=HF_REPO,
        filename=CKPT_NAME,
        revision=REVISION,
        local_dir=args.work_dir,
        token=os.environ.get("HF_TOKEN") or None,
    )

    converted: dict[str, torch.Tensor] = {}
    src_bytes = 0
    with safe_open(src, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            src_bytes += tensor.numel() * tensor.element_size()
            if tensor.is_floating_point():
                tensor = tensor.to(torch.bfloat16)
            converted[name] = tensor

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_file(converted, args.out, metadata={"source_revision": REVISION})
    out_bytes = os.path.getsize(args.out)
    print(f"converted {src_bytes / 1e9:.2f} GB fp32 -> {out_bytes / 1e9:.2f} GB bf16 at {args.out}")

    # Free the fp32 source immediately so the runner has room for the image build.
    os.remove(src)


if __name__ == "__main__":
    main()
