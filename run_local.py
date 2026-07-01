#!/usr/bin/env python
"""Local / on-box benchmark + smoke test — NO serverless platform involved.

This is step 2 of the build: rent one GPU box, run this against a real video,
confirm keypoints look sane and get the true per-video time before containerizing.

Examples:
  # local file in, local json out
  python run_local.py --video sample.mp4 --out out.json --stride 2

  # presigned R2 GET in, presigned R2 PUT out (exactly what the endpoint does)
  python run_local.py --video "$GET_URL" --put-url "$PUT_URL" --stride 2 --batch 24
"""
import argparse
import json

from core.infer import run_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="presigned GET url or local path")
    ap.add_argument("--out", help="write results JSON to this local path")
    ap.add_argument("--put-url", help="presigned R2 PUT url for results JSON")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if not args.out and not args.put_url:
        args.out = "out.json"

    summary = run_video(
        args.video,
        result_put_url=args.put_url,
        result_local_path=args.out,
        frame_stride=args.stride,
        batch_size=args.batch,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
