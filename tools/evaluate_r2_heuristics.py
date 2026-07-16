#!/usr/bin/env python3
"""Evaluate feral-api keypoint heuristics against labels already staged on R2."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import boto3
import numpy as np
import pandas as pd
from botocore.client import Config as BotoConfig


def required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"missing environment variable; expected one of {names}")


def load_heuristics(feral_api_dir: Path):
    sys.path.insert(0, str(feral_api_dir.resolve()))
    import heuristics  # pylint: disable=import-outside-toplevel

    return heuristics


class R2:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=required_env("R2_ENDPOINT", "R2_ENDPOINT_URL"),
            aws_access_key_id=required_env("AWS_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
            aws_secret_access_key=required_env(
                "AWS_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"
            ),
            region_name="auto",
            config=BotoConfig(signature_version="s3v4"),
        )

    def bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def json(self, key: str) -> dict:
        return json.loads(self.bytes(key))

    def keys(self, prefix: str) -> Iterable[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def presign_get(self, key: str, expires_in: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
            HttpMethod="GET",
        )


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def finite(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def summarize(rows: list[dict]) -> dict:
    target = np.asarray([row["target"] for row in rows], dtype=float)
    prediction = np.asarray([row["prediction"] for row in rows], dtype=float)
    error = prediction - target
    summary = {
        "n": len(rows),
        "subjects": len({row["subject"] for row in rows}),
        "target_min": finite(np.min(target)),
        "target_max": finite(np.max(target)),
        "prediction_min": finite(np.min(prediction)),
        "prediction_max": finite(np.max(prediction)),
        "mae": finite(np.mean(np.abs(error))),
        "rmse": finite(np.sqrt(np.mean(error**2))),
        "pearson": finite(correlation(target, prediction)),
        "spearman": finite(correlation(rankdata(target), rankdata(prediction))),
        "low_confidence": sum(row.get("confidence") == "low" for row in rows),
    }
    if all(row.get("prediction_grade") is not None for row in rows):
        grades = np.asarray([row["prediction_grade"] for row in rows], dtype=float)
        summary.update(
            {
                "grade_mae": finite(np.mean(np.abs(grades - target))),
                "grade_within_1": finite(np.mean(np.abs(grades - target) <= 1.0)),
                "grade_exact_vs_rounded": finite(
                    np.mean(grades == np.rint(target))
                ),
            }
        )
    if target.max() <= 4.0 and all(row.get("severity") is not None for row in rows):
        continuous = 4.0 * np.asarray([row["severity"] for row in rows], dtype=float)
        summary.update(
            {
                "severity_x4_mae": finite(np.mean(np.abs(continuous - target))),
                "severity_x4_pearson": finite(correlation(target, continuous)),
                "severity_x4_spearman": finite(
                    correlation(rankdata(target), rankdata(continuous))
                ),
            }
        )
    return summary


def result_row(
    evaluation: str,
    dataset: str,
    source: str,
    subject: str,
    target: float,
    result: dict,
    prediction_field: str = "grade",
) -> dict:
    prediction = float(result[prediction_field])
    return {
        "evaluation": evaluation,
        "dataset": dataset,
        "source": source,
        "subject": subject,
        "target": float(target),
        "prediction": prediction,
        "prediction_grade": int(result["grade"]) if "grade" in result else None,
        "severity": result.get("severity"),
        "confidence": result.get("confidence"),
        "submetrics": result.get("submetrics", {}),
    }


def evaluate_hubu(
    r2: R2,
    heuristics,
    output_prefix: str,
    split: str = "all",
) -> list[dict]:
    labels = r2.json("hubu-fis/hubu-fis_regression_labels.json")
    canonical_splits = r2.json("hubu-fis/hubu-fis_labels.json")["splits"]
    split_of = {
        source: split_name
        for split_name, sources in canonical_splits.items()
        for source in sources
    }
    allowed = set(canonical_splits.get(split, [])) if split != "all" else None
    mean = float(labels["normalization"]["mean"][0])
    std = float(labels["normalization"]["std"][0])
    rows = []
    prefix = f"{output_prefix.rstrip('/')}/hubu-fis/videos/"
    for key in r2.keys(prefix):
        if not key.endswith(".json"):
            continue
        source = Path(key).stem + ".mp4"
        if source not in labels["labels"]:
            continue
        if allowed is not None and source not in allowed:
            continue
        target = float(labels["labels"][source][0]) * std + mean
        result = heuristics.run("fingerTapping", r2.json(key))
        subject = re.sub(r"_(DCHA|IZDA)$", "", Path(key).stem)
        row = result_row(
                "hubu_finger_tapping_grade",
                "HUBU-FIS",
                source,
                subject,
                target,
                result,
            )
        row["split"] = split_of.get(source)
        rows.append(row)
    return rows


TULIP_TASKS = {
    "7. Finger_tapping_left": [("Finger tapping - Left hand", "fingerTapping")],
    "8. FInger_tapping_right": [("Finger tapping - Right hand", "fingerTapping")],
    "20. Cross_the_arms_and_rise_from_the_chair": [
        ("Arising from chair", "arisingFromChair")
    ],
    "25. Gait": [("Gait", "gait"), ("Freezing of gait", "freezingOfGait")],
}


def tulip_labels(r2: R2) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    base = "tulip/labels_csv_files_202503/labels_csv_files/"
    for key in r2.keys(base):
        if not key.endswith(".csv"):
            continue
        match = re.search(r"subject(\d+)_labels\.csv$", key)
        if not match:
            continue
        subject = str(int(match.group(1)))
        reader = csv.DictReader(io.StringIO(r2.bytes(key).decode("utf-8-sig")))
        item_scores = {}
        for row in reader:
            try:
                scores = [float(row[f"label_clinician{i}"]) for i in (1, 2, 3)]
            except (TypeError, ValueError):
                continue
            item_scores[row["UPDRS_name"]] = float(np.mean(scores))
        out[subject] = item_scores
    return out


def evaluate_tulip(r2: R2, heuristics, output_prefix: str) -> list[dict]:
    labels = tulip_labels(r2)
    rows = []
    prefix = f"{output_prefix.rstrip('/')}/tulip/"
    for key in r2.keys(prefix):
        if not key.endswith(".json"):
            continue
        parts = key.split("/")
        task = next((part for part in parts if part in TULIP_TASKS), None)
        subject_match = re.search(r"/Subject_(\d+)/", key)
        if not task or not subject_match:
            continue
        subject = str(int(subject_match.group(1)))
        keypoints = r2.json(key)
        for label_name, test_type in TULIP_TASKS[task]:
            if label_name not in labels.get(subject, {}):
                continue
            result = heuristics.run(test_type, keypoints)
            rows.append(
                result_row(
                    f"tulip_{test_type}_grade",
                    "TULIP",
                    "/".join(parts[-3:]),
                    subject,
                    labels[subject][label_name],
                    result,
                )
            )
    return rows


def evaluate_ribeiro(r2: R2, heuristics, output_prefix: str) -> list[dict]:
    labels = r2.json("parkinson/parkinson_labels_trainval_negs.json")["labels"]
    rows = []
    prefix = f"{output_prefix.rstrip('/')}/parkinson/videos/"
    for key in r2.keys(prefix):
        if not key.endswith(".json"):
            continue
        source = Path(key).stem + ".mp4"
        frame_labels = labels.get(source)
        if not frame_labels:
            continue
        result = heuristics.run("freezingOfGait", r2.json(key))
        target_percent = 100.0 * float(np.mean(frame_labels))
        predicted_percent = float(result.get("fog_percent", 0.0))
        row = result_row(
            "ribeiro_fog_percent",
            "Ribeiro-FoG",
            source,
            source.split("_")[0],
            target_percent,
            {**result, "fog_percent": predicted_percent},
            prediction_field="fog_percent",
        )
        row["prediction_grade"] = None
        row["labeled_frames"] = len(frame_labels)
        rows.append(row)
    return rows


# Legacy Sapiens-0.3B Goliath starts with COCO-17. Map only joints the current
# gait heuristic uses. The remaining 5B slots stay invalid; there is no safe
# 308-index identity between the legacy and Sapiens2 layouts.
COCO17_TO_5B = {
    0: 0,    # nose
    5: 5, 6: 6,    # shoulders
    7: 7, 8: 8,    # elbows
    9: 62, 10: 41,  # wrists / hand roots
    11: 9, 12: 10,  # hips
    13: 11, 14: 12,  # knees
    15: 13, 16: 14,  # ankles
}


def legacy_coco_parquet_to_array(raw: bytes) -> np.ndarray:
    frame = pd.read_parquet(io.BytesIO(raw))
    if frame.empty:
        raise ValueError("empty legacy pose parquet")
    # TULIP is single-person. This remains deterministic if an accidental
    # second detection exists: keep the highest detector score per frame.
    frame = (
        frame.sort_values(["frame", "det_score"], ascending=[True, False], na_position="last")
        .drop_duplicates("frame")
        .sort_values("frame")
    )
    kp = np.zeros((len(frame), 308, 3), dtype=np.float32)
    for coco_idx, compact_idx in COCO17_TO_5B.items():
        x = frame[f"k{coco_idx}_x"].to_numpy(dtype=np.float32)
        y = frame[f"k{coco_idx}_y"].to_numpy(dtype=np.float32)
        valid = np.isfinite(x) & np.isfinite(y)
        kp[:, compact_idx, 0] = np.where(valid, x, 0.0)
        kp[:, compact_idx, 1] = np.where(valid, y, 0.0)
        kp[:, compact_idx, 2] = valid.astype(np.float32)
    return kp


def rtmpose_coco_parquet_to_array(raw: bytes) -> np.ndarray:
    frame = pd.read_parquet(io.BytesIO(raw)).sort_values("frame")
    kp = np.zeros((len(frame), 308, 3), dtype=np.float32)
    for coco_idx, compact_idx in COCO17_TO_5B.items():
        x = frame[f"k{coco_idx}_x"].to_numpy(dtype=np.float32)
        y = frame[f"k{coco_idx}_y"].to_numpy(dtype=np.float32)
        score = frame[f"k{coco_idx}_s"].to_numpy(dtype=np.float32)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(score)
        kp[:, compact_idx, 0] = np.where(valid, x, 0.0)
        kp[:, compact_idx, 1] = np.where(valid, y, 0.0)
        kp[:, compact_idx, 2] = np.where(valid, score, 0.0)
    return kp


def probe_fps(url: str) -> float:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", url,
        ],
        text=True,
        timeout=60,
    ).strip()
    num, den = raw.split("/")
    fps = float(num) / float(den)
    if fps <= 0:
        raise ValueError(f"invalid fps {raw!r}")
    return fps


def score_gait_array(heuristics, kp: np.ndarray, fps: float) -> dict:
    """Run the production gait scorer without materializing a huge fake JSON."""
    common = heuristics.gait.C
    original_to_array = common.to_array
    original_fps_of = common.fps_of
    try:
        common.to_array = lambda _: kp
        common.fps_of = lambda _: fps
        return heuristics.gait.score({})
    finally:
        common.to_array = original_to_array
        common.fps_of = original_fps_of


def subject_median_rows(rows: list[dict], evaluation: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    out = []
    for subject, group in sorted(grouped.items()):
        prediction = float(np.median([row["prediction"] for row in group]))
        target = float(np.mean([row["target"] for row in group]))
        out.append(
            {
                "evaluation": evaluation,
                "dataset": group[0]["dataset"],
                "source": f"{subject}:median_of_{len(group)}_cameras",
                "subject": subject,
                "target": target,
                "prediction": prediction,
                "prediction_grade": int(round(prediction)),
                "severity": float(np.median([row["severity"] for row in group])),
                "confidence": "synthetic",
                "submetrics": {"camera_count": len(group)},
            }
        )
    return out


def evaluate_tulip_legacy_gait(
    r2: R2,
    heuristics,
    limit: int | None = None,
    cameras: list[int] | None = None,
) -> list[dict]:
    labels = r2.json("tulip/gait_labels.json")
    mean = float(labels["normalization"]["mean"][0])
    std = float(labels["normalization"]["std"][0])
    rows = []
    prefix = "tulip/gait/poses_sapiens03b_goliath/"
    keys = [key for key in r2.keys(prefix) if key.endswith(".parquet")]
    if cameras:
        endings = tuple(f"_camera{camera}.mp4.parquet" for camera in cameras)
        keys = [key for key in keys if key.endswith(endings)]
    if limit:
        keys = keys[:limit]
    for index, key in enumerate(keys, start=1):
        source = Path(key).name.removesuffix(".parquet")
        if source not in labels["labels"]:
            continue
        target = float(labels["labels"][source][0]) * std + mean
        kp = legacy_coco_parquet_to_array(r2.bytes(key))
        result = score_gait_array(heuristics, kp, fps=80.0)
        subject_match = re.match(r"(subject\d+)_camera\d+\.mp4", source)
        subject = subject_match.group(1) if subject_match else source
        row = result_row(
            "tulip_legacy_gait_grade_clip",
            "TULIP legacy Sapiens-0.3B",
            source,
            subject,
            target,
            result,
        )
        row["confidence"] = "synthetic"
        row["adapter"] = "COCO17 named-body map; confidence synthesized from finite coordinates"
        rows.append(row)
        print(f"legacy gait {index}/{len(keys)} {source}", flush=True)
    return rows + subject_median_rows(rows, "tulip_legacy_gait_grade_subject_median")


KOA_PD_TARGET = {"NM": 0.0, "PD_ML": 1.0, "PD_MD": 2.0, "PD_SV": 3.0}


def evaluate_koa_existing_gait(
    r2: R2,
    heuristics,
    limit: int | None = None,
) -> list[dict]:
    labels = r2.json("koa-pd-nm-gait/koa-pd-nm-gait_labels.json")
    prefix = "koa-pd-nm-gait/poses_per_video/"
    keys = [
        key
        for key in r2.keys(prefix)
        if key.endswith(".parquet")
        and labels.get(Path(key).name.removesuffix(".parquet")) in KOA_PD_TARGET
    ]
    if limit:
        keys = keys[:limit]
    rows = []
    for index, key in enumerate(keys, start=1):
        source = Path(key).name.removesuffix(".parquet")
        category = labels[source]
        fps = probe_fps(r2.presign_get(f"koa-pd-nm-gait/videos/{source}"))
        kp = rtmpose_coco_parquet_to_array(r2.bytes(key))
        result = score_gait_array(heuristics, kp, fps)
        subject_match = re.match(r"(\d+)_(NM|PD)_", source)
        subject = "_".join(subject_match.groups()) if subject_match else source
        row = result_row(
            "koa_existing_gait_ordinal_clip",
            "KOA-PD-NM RTMPose COCO-17",
            source,
            subject,
            KOA_PD_TARGET[category],
            result,
        )
        row.update(
            {
                "category": category,
                "fps": fps,
                "adapter": "COCO17 named-body map with native RTMPose confidence",
            }
        )
        rows.append(row)
        print(f"KOA/PD gait {index}/{len(keys)} {source}", flush=True)
    return rows + subject_median_rows(rows, "koa_existing_gait_ordinal_subject_median")


def run(args: argparse.Namespace) -> int:
    r2 = R2(args.bucket)
    heuristics = load_heuristics(args.feral_api_dir)
    rows = []
    if args.dataset in {"all", "hubu"}:
        rows.extend(
            evaluate_hubu(
                r2, heuristics, args.output_prefix, args.hubu_split
            )
        )
    if args.dataset in {"all", "tulip"}:
        rows.extend(evaluate_tulip(r2, heuristics, args.output_prefix))
    if args.dataset in {"all", "ribeiro"}:
        rows.extend(evaluate_ribeiro(r2, heuristics, args.output_prefix))
    if args.dataset in {"all", "tulip-legacy-gait"}:
        rows.extend(
            evaluate_tulip_legacy_gait(
                r2, heuristics, args.limit, args.tulip_camera
            )
        )
    if args.dataset in {"all", "koa-existing-gait"}:
        rows.extend(evaluate_koa_existing_gait(r2, heuristics, args.limit))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["evaluation"]].append(row)
    summaries = {name: summarize(group) for name, group in sorted(grouped.items())}
    payload = {"summaries": summaries, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0 if rows else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="feral")
    parser.add_argument("--output-prefix", default="mds-updrs-keypoints-5b")
    parser.add_argument(
        "--dataset",
        choices=(
            "all", "hubu", "tulip", "ribeiro", "tulip-legacy-gait",
            "koa-existing-gait",
        ),
        default="all",
    )
    parser.add_argument(
        "--feral-api-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "feral-api",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tulip-camera", type=int, action="append", default=[])
    parser.add_argument(
        "--hubu-split",
        choices=("all", "train", "val", "test"),
        default="all",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
