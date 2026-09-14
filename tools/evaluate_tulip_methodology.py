#!/usr/bin/env python3
"""Run the frozen Luche heuristics on an existing TULIP partition.

The staged TULIP label files have train/val membership and empty test arrays.
By default this evaluator treats the untouched ``val`` membership as the
held-out evaluation set requested for the methodology page. It can also read
the existing ``train`` membership for an explicitly in-sample descriptive
table. It never rewrites splits, tunes a threshold, selects a camera from
outcomes, or fills an abstention with a default score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


FINGER_LABELS = "tulip-hand-best-view/tulip-hand-best-view_labels.json"
HAND_LABELS = FINGER_LABELS
LOWER_LABELS = "tulip-lower-limb-best-view/tulip-lower-limb-best-view_labels.json"
CHAIR_LABELS = "tulip-chair-side-views/tulip-chair-side-views_labels.json"
GAIT_LABELS = "tulip/gait_labels.json"
TREMOR_LABELS = "tulip-rest-tremor-best-view/tulip-rest-tremor-best-view_labels.json"

RAW_LABELS_PREFIX = "tulip/labels_csv_files_202503/labels_csv_files"
GENERATED_PREFIX = "mds-updrs-keypoints-5b-methodology"
EXISTING_PREFIX = "mds-updrs-keypoints-5b"


def required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"missing environment variable; expected one of {names}")


class R2:
    def __init__(self, bucket: str):
        import boto3
        from botocore.client import Config as BotoConfig

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

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code")
            if status == 404 or code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise


@dataclass(frozen=True)
class Example:
    task_id: str
    task: str
    test_type: str
    subject: int
    target_name: str
    keypoints_key: str
    side: str | None = None


def subject_of(name: str) -> int:
    match = re.search(r"subject(\d+)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"cannot parse TULIP subject from {name!r}")
    return int(match.group(1))


def split_names(r2: R2, key: str, partition: str) -> list[str]:
    payload = r2.json(key)
    test_names = payload["splits"].get("test", [])
    if test_names:
        raise RuntimeError(
            f"{key} now has a non-empty test split; review the protocol before unsealing it"
        )
    train_names = list(payload["splits"].get("train", []))
    held_out_names = list(payload["splits"]["val"])
    overlap = {subject_of(name) for name in train_names} & {
        subject_of(name) for name in held_out_names
    }
    if overlap:
        raise RuntimeError(
            f"{key} has participants in both train and held-out membership: {sorted(overlap)}"
        )
    return list(payload["splits"][partition])


def raw_ratings(r2: R2, subject: int) -> dict[str, list[float]]:
    key = f"{RAW_LABELS_PREFIX}/subject{subject}_labels.csv"
    text = r2.bytes(key).decode("utf-8-sig")
    out: dict[str, list[float]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        try:
            ratings = [float(row[f"label_clinician{i}"]) for i in (1, 2, 3)]
        except (TypeError, ValueError):
            continue
        out[row["UPDRS_name"]] = ratings
    return out


def build_examples(r2: R2, partition: str) -> list[Example]:
    examples: list[Example] = []

    # Finger tapping uses the canonical hand held-out participants. Left/right
    # camera views are the already-fixed best-side hand views. TULIP tasks 9/10
    # are thumb-index tapping; tasks 7/8 are sequential finger opposition and
    # must not be paired with the clinical thumb-index tapping ratings.
    finger_subjects = sorted(
        {subject_of(name) for name in split_names(r2, FINGER_LABELS, partition)}
    )
    for subject in finger_subjects:
        examples.extend(
            [
                Example(
                    "finger-tapping",
                    "Finger tapping",
                    "fingerTapping",
                    subject,
                    "Finger tapping - Left hand",
                    f"{GENERATED_PREFIX}/tulip/Subject_{subject}/Subject_{subject}/"
                    "9. Index_Finger_tapping_left/Camera3.json",
                    "left",
                ),
                Example(
                    "finger-tapping",
                    "Finger tapping",
                    "fingerTapping",
                    subject,
                    "Finger tapping - Right hand",
                    f"{GENERATED_PREFIX}/tulip/Subject_{subject}/Subject_{subject}/"
                    "10. Index_Finger_tapping_right/Camera4.json",
                    "right",
                ),
            ]
        )

    hand_specs = {
        11: ("hand-movements", "Hand movements", "handMovements", "Hand movements - Left hand", "left"),
        12: ("hand-movements", "Hand movements", "handMovements", "Hand movements - Right hand", "right"),
        13: ("pronation-supination", "Pronation–supination", "pronationSupination", "Pronation-supination - Left hand", "left"),
        14: ("pronation-supination", "Pronation–supination", "pronationSupination", "Pronation-supination - Right hand", "right"),
    }
    for name in split_names(r2, HAND_LABELS, partition):
        match = re.search(r"_task(1[1-4])_", name)
        if not match:
            raise ValueError(f"unexpected hand filename: {name}")
        task_id, task, test_type, target, side = hand_specs[int(match.group(1))]
        examples.append(
            Example(
                task_id,
                task,
                test_type,
                subject_of(name),
                target,
                f"{EXISTING_PREFIX}/tulip-hand-best-view/videos/{Path(name).stem}.json",
                side,
            )
        )

    lower_specs = {
        15: ("leg-agility", "Leg agility", "legAgility", "Leg agility - Left leg", "left"),
        16: ("leg-agility", "Leg agility", "legAgility", "Leg agility - Right leg", "right"),
        17: ("toe-tapping", "Toe tapping", "toeTapping", "Toe tapping - Left foot", "left"),
        18: ("toe-tapping", "Toe tapping", "toeTapping", "Toe tapping - Right foot", "right"),
    }
    for name in split_names(r2, LOWER_LABELS, partition):
        match = re.search(r"_task(1[5-8])_", name)
        if not match:
            raise ValueError(f"unexpected lower-limb filename: {name}")
        task_id, task, test_type, target, side = lower_specs[int(match.group(1))]
        examples.append(
            Example(
                task_id,
                task,
                test_type,
                subject_of(name),
                target,
                f"{EXISTING_PREFIX}/tulip-lower-limb-best-view/videos/{Path(name).stem}.json",
                side,
            )
        )

    for name in split_names(r2, CHAIR_LABELS, partition):
        if not name.endswith("_camera2.mp4"):
            continue
        examples.append(
            Example(
                "arising-from-chair",
                "Arising from a chair",
                "arisingFromChair",
                subject_of(name),
                "Arising from chair",
                f"{GENERATED_PREFIX}/tulip-chair-side-views/videos/{Path(name).stem}.json",
            )
        )

    # The existing gait staging predates the four-subject split used for newer
    # tasks. Preserve its own held-out membership and fixed Camera 4 view.
    gait_subjects = sorted(
        {subject_of(name) for name in split_names(r2, GAIT_LABELS, partition)}
    )
    for subject in gait_subjects:
        examples.append(
            Example(
                "gait",
                "Gait",
                "gait",
                subject,
                "Gait",
                f"{GENERATED_PREFIX}/tulip/gait_videos/subject{subject:02d}_camera4.json",
            )
        )

    for name in split_names(r2, TREMOR_LABELS, partition):
        examples.append(
            Example(
                "rest-tremor",
                "Rest tremor (right upper extremity)",
                "restTremor",
                subject_of(name),
                "Rest tremor amplitude - RUE",
                f"{EXISTING_PREFIX}/tulip-rest-tremor-best-view/videos/{Path(name).stem}.json",
            )
        )

    return sorted(examples, key=lambda item: (item.task_id, item.subject, item.side or ""))


def finite(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def adapt_sampled_frame_indices(payload: dict) -> tuple[dict, bool]:
    """Normalize a regularly sampled extractor payload to the heuristic contract.

    The extractor retains source-video frame numbers (0, stride, 2*stride, ...),
    while the heuristics expect the sampled sequence itself to be numbered
    contiguously and derive its effective FPS from source_fps / frame_stride.
    Only this exact regular-stride case is adapted; keypoints and timing metadata
    are not changed.
    """
    frames = payload.get("frames")
    stride = payload.get("frame_stride")
    if not isinstance(frames, list) or len(frames) < 2:
        return payload, False
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 1:
        return payload, False
    ordered = sorted(frames, key=lambda frame: frame.get("frame_idx", -1))
    indices = [frame.get("frame_idx") for frame in ordered]
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in indices):
        return payload, False
    if len(set(indices)) != len(indices) or any(
        b - a != stride for a, b in zip(indices, indices[1:])
    ):
        return payload, False
    normalized_frames = [
        {**frame, "source_frame_idx": frame["frame_idx"], "frame_idx": index}
        for index, frame in enumerate(ordered)
    ]
    return {**payload, "frames": normalized_frames}, True


def bootstrap_mae_ci(rows: list[dict], field: str, seed: int = 20260822) -> list[float] | None:
    by_subject: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_subject[row["subject"]].append(row)
    subjects = sorted(by_subject)
    if len(subjects) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(5000):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        sample_rows = [row for subject in sampled for row in by_subject[int(subject)]]
        estimates.append(
            np.mean([abs(float(row[field]) - float(row["target_mean"])) for row in sample_rows])
        )
    return [finite(np.quantile(estimates, 0.025)), finite(np.quantile(estimates, 0.975))]


def summarize(rows: list[dict]) -> dict:
    accepted = [row for row in rows if row["scoreable"]]
    summary = {
        "total": len(rows),
        "accepted": len(accepted),
        "participants_total": len({row["subject"] for row in rows}),
        "participants_accepted": len({row["subject"] for row in accepted}),
        "coverage": finite(len(accepted) / len(rows)) if rows else None,
        "quality_failures": dict(
            sorted(
                Counter(
                    failure
                    for row in rows
                    for failure in row.get("quality_failures", [])
                ).items()
            )
        ),
    }
    if not accepted:
        return summary

    target = np.asarray([row["target_mean"] for row in accepted], dtype=float)
    grade = np.asarray([row["grade"] for row in accepted], dtype=float)
    continuous = np.asarray([row["continuous_grade"] for row in accepted], dtype=float)
    clinician_errors = []
    clinician_within_one = []
    for row in accepted:
        ratings = np.asarray(row["clinician_ratings"], dtype=float)
        for index, rating in enumerate(ratings):
            other_mean = float(np.mean(np.delete(ratings, index)))
            clinician_errors.append(abs(float(rating) - other_mean))
            clinician_within_one.append(abs(float(rating) - other_mean) <= 1.0)

    summary.update(
        {
            "target_range": [finite(np.min(target)), finite(np.max(target))],
            "grade_range": [finite(np.min(grade)), finite(np.max(grade))],
            "grade_mae": finite(np.mean(np.abs(grade - target))),
            "grade_rmse": finite(np.sqrt(np.mean((grade - target) ** 2))),
            "grade_within_one": finite(np.mean(np.abs(grade - target) <= 1.0)),
            "grade_mae_participant_bootstrap_95_ci": bootstrap_mae_ci(accepted, "grade"),
            "continuous_grade_mae": finite(np.mean(np.abs(continuous - target))),
            "clinician_leave_one_out_mae": finite(np.mean(clinician_errors)),
            "clinician_leave_one_out_within_one": finite(np.mean(clinician_within_one)),
        }
    )
    return summary


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> int:
    r2 = R2(args.bucket)
    api = args.feral_api_dir.resolve()
    sys.path.insert(0, str(api))
    import heuristics  # pylint: disable=import-outside-toplevel

    examples = build_examples(r2, args.partition)
    missing = [example.keypoints_key for example in examples if not r2.exists(example.keypoints_key)]
    if missing:
        print(json.dumps({"missing_keypoints": missing}, indent=2))
        return 3

    rating_cache: dict[int, dict[str, list[float]]] = {}
    rows = []
    for index, example in enumerate(examples, start=1):
        ratings_by_name = rating_cache.setdefault(
            example.subject, raw_ratings(r2, example.subject)
        )
        ratings = ratings_by_name.get(example.target_name)
        if ratings is None:
            raise RuntimeError(
                f"subject {example.subject} missing target {example.target_name!r}"
            )
        payload = r2.json(example.keypoints_key)
        keypoint_metadata = {
            field: payload.get(field)
            for field in (
                "git_sha",
                "model_size",
                "sapiens_model_revision",
                "source_fps",
                "frame_stride",
                "num_source_frames",
                "num_processed_frames",
                "person_detection",
                "person_detector_revision",
            )
            if field in payload
        }
        payload, frame_index_adapter = adapt_sampled_frame_indices(payload)
        keypoint_metadata["sampled_frame_index_adapter"] = frame_index_adapter
        if example.side:
            payload = {**payload, "evaluated_side": example.side}
        try:
            result = heuristics.run(example.test_type, payload)
            scoreable = bool(result.get("scoreable", True))
            row = {
                **asdict(example),
                "clinician_ratings": ratings,
                "target_mean": float(np.mean(ratings)),
                "keypoint_metadata": keypoint_metadata,
                "scoreable": scoreable,
                "grade": float(result["grade"]) if scoreable else None,
                "continuous_grade": 4.0 * float(result["severity"]) if scoreable else None,
                "confidence": result.get("confidence"),
                "heuristic_version": result.get("heuristic_version", heuristics.VERSION),
                "quality_failures": result.get("submetrics", {}).get("quality_failures", []),
                "error": None,
            }
        except Exception as exc:  # preserve the failed example in the audit artifact
            row = {
                **asdict(example),
                "clinician_ratings": ratings,
                "target_mean": float(np.mean(ratings)),
                "keypoint_metadata": keypoint_metadata,
                "scoreable": False,
                "grade": None,
                "continuous_grade": None,
                "confidence": "error",
                "heuristic_version": None,
                "quality_failures": ["evaluation_exception"],
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(
            f"{index:02d}/{len(examples)} {example.task_id} subject={example.subject:02d} "
            f"scoreable={row['scoreable']} grade={row['grade']}",
            flush=True,
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    summaries = {
        task_id: {
            "task": task_rows[0]["task"],
            **summarize(task_rows),
        }
        for task_id, task_rows in sorted(grouped.items())
    }
    protocol = {
        "dataset": "TULIP v1.0.3",
        "public_release_participants": 12,
        "split": (
            "existing val membership; source test arrays are empty"
            if args.partition == "val"
            else "existing train membership; in-sample descriptive evaluation"
        ),
        "partition": args.partition,
        "camera_policy": {
            "finger_tapping_left": 3,
            "finger_tapping_right": 4,
            "hand_movements_left": 3,
            "hand_movements_right": 4,
            "pronation_supination_left": 3,
            "pronation_supination_right": 4,
            "toe_tapping": 3,
            "leg_agility": 3,
            "arising_from_chair": 2,
            "gait": 4,
            "rest_tremor": 2,
        },
        "camera_selection_basis": "prespecified visibility/app-view choices; no outcome selection",
        "finger_tapping_task_mapping": {
            "left": "9. Index_Finger_tapping_left",
            "right": "10. Index_Finger_tapping_right",
            "correction": "2026-09-14: earlier audits incorrectly used sequential opposition tasks 7/8",
        },
        "target": "mean of three clinician ratings",
        "primary_prediction": "actual heuristic grade on the 0-4 scale",
        "primary_metrics": ["grade_mae", "grade_within_one", "coverage"],
        "abstention_policy": "no score; never imputed",
        "sampled_frame_index_adapter": (
            "regularly sampled extractor frame IDs are renumbered contiguously; "
            "source frame IDs are retained per frame and source_fps/frame_stride is unchanged"
        ),
        "no_posthoc_tuning": True,
    }
    payload = {
        "schema_version": "1.0",
        "protocol": protocol,
        "provenance": {
            "feral_api_git_head": git_head(api),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "summaries": summaries,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="feral")
    parser.add_argument("--partition", choices=("train", "val"), default="val")
    parser.add_argument(
        "--feral-api-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "feral-api",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
