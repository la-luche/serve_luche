#!/usr/bin/env python3
"""Offline paired QC ablation of the frozen 156-row TULIP methodology audit.

Reads existing keypoints from R2; never changes datasets or production scorers.
QC-off keeps confidence-based feature preprocessing. Raw-coordinate mode also
removes confidence masking, to distinguish missing track support from gates.
Neither mode invents features when the estimator cannot compute a finite score.

Historical warning: the default audit used sequential-opposition tasks 7/8 for
finger tapping. Replaying it reproduces that invalid task join. Use
evaluate_tulip_index_correction.py for corrected thumb-index tasks 9/10 results.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import boto3
import numpy as np
from botocore.config import Config

FROZEN_COMMIT = "c4dcf70fe21813e72e202830dab9738a53057f56"


def r2_client():
    # Same local credential source as run_tulip_methodology_extraction.sh.
    # Values stay in this process and are never printed or written to artifacts.
    names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "R2_ENDPOINT")
    values = {name: os.getenv(name) for name in names}
    if not all(values.values()):
        source = Path(os.environ["FERAL_SHARED_DOCS"]) / "shared_claude/skills/r2-download/SKILL.md"
        for line in source.read_text().splitlines():
            match = re.match(r"export (AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|R2_ENDPOINT)=(.*)$", line)
            if match and not values[match[1]]:
                values[match[1]] = shlex.split(match[2])[0]
    if not all(values.values()):
        raise RuntimeError("missing R2 credentials")
    return boto3.client(
        "s3", endpoint_url=values["R2_ENDPOINT"],
        aws_access_key_id=values["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=values["AWS_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(retries={"max_attempts": 3}, max_pool_connections=4),
    )


def cache_path(cache, key):
    return cache / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def download(rows, cache):
    cache.mkdir(parents=True, exist_ok=True)
    keys = sorted({row["keypoints_key"] for row in rows})
    missing = [key for key in keys if not cache_path(cache, key).exists()]
    if not missing:
        print(f"all {len(keys)} keypoint files cached", flush=True)
        return
    client = r2_client()

    def fetch(key):
        data = client.get_object(Bucket="feral", Key=key)["Body"].read()
        json.loads(data)  # Validate before atomic cache insertion.
        dest = cache_path(cache, key)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return key, len(data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, key) for key in missing]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key, size = future.result()
            print(f"download {i}/{len(missing)} {size / 1e6:.1f} MB {key}", flush=True)


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f"frozen source patch not unique: {before!r}")
    return source.replace(before, after, 1)


def patched_source(name, source, raw=False):
    """Explicit edits to frozen gates, with the complete diff saved for review."""
    if name in {"finger_tapping", "hand_movements", "pronation_supination", "toe_tapping", "leg_agility", "gait", "chair"}:
        # These modules have exactly one recording-level gate in score().
        start = source.index("def score(")
        prefix, score = source[:start], source[start:]
        score = replace_once(score, "    if failures:\n", "    if False and failures:  # offline QC ablation\n")
        source = prefix + score
    if name == "chair":
        source = replace_once(source,
            "transitions = [transition for transition, (accepted, _) in zip(candidates, evidence) if accepted]",
            "transitions = list(candidates)  # offline: bypass standing-evidence gate")
    if name == "rest_tremor":
        # Coverage/gap checks may annotate a track but do not exclude it.
        for text in (
            '            failures.append(f"missing_required_joint:{joint}")\n            continue',
            '            failures.append(f"tracking_gap_joint:{joint}")\n            continue',
        ):
            source = replace_once(source, text, text.rsplit("\n", 1)[0])
        source = replace_once(source,
            '    if failures:\n        return features\n',
            '    if any(joint not in tracks for joint in required):\n        return features\n')
        source = replace_once(source,
            '    if anchor_residual > ANCHOR_RESIDUAL_P95_MAX:\n        features["quality_failures"] = ["unstable_body_anchor_reprojection"]\n        return features',
            '    if anchor_residual > ANCHOR_RESIDUAL_P95_MAX:\n        features["quality_failures"] = ["unstable_body_anchor_reprojection"]')
        source = replace_once(source,
            '    if failures:\n        return _target_rejected(TARGETS[0], failures[0], features)',
            '    if "attributed_body_anchored_amplitude" not in features:\n        return _target_rejected(TARGETS[0], failures[0] if failures else "missing_amplitude", features)')
        source = replace_once(source,
            '    if global_failures:\n        return _rejected(global_failures[0], submetrics)',
            '    if False and global_failures:  # offline QC ablation\n        return _rejected(global_failures[0], submetrics)')
    if raw:
        # Use finite reported coordinates, even if the detector is uncertain.
        # No coordinate values or confidence values in the payload are rewritten.
        source = re.sub(r"^CONFIDENCE_MIN = [0-9.]+$", "CONFIDENCE_MIN = -float('inf')", source, flags=re.M)
        if name == "common":
            source = replace_once(source, "CONF_MIN = 0.3", "CONF_MIN = -float('inf')")
        if name == "rest_tremor":
            source = replace_once(source,
                '    if geometry_ratio < ANCHOR_GEOMETRY_RATIO_MIN:',
                '    if geometry_ratio <= 1e-12:  # retain only numerical degeneracy check')
            source = replace_once(source,
                '    if not np.isfinite(scale) or scale < MIN_HAND_SCALE_PX:',
                '    if not np.isfinite(scale) or scale <= 1e-12:')
    return source


def load_package(name, path):
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare_scorers(api, output):
    names = subprocess.check_output(["git", "ls-tree", "--name-only", FROZEN_COMMIT, "heuristics/"], cwd=api, text=True).splitlines()
    root = output / "frozen_source"
    root.mkdir(parents=True, exist_ok=True)
    keypoint_source = subprocess.check_output(["git", "show", f"{FROZEN_COMMIT}:keypoints.py"], cwd=api)
    (root / "keypoints.py").write_bytes(keypoint_source)
    sys.path.insert(0, str(root))
    packages = {}
    patches = []
    for mode in ("baseline", "qc_off", "raw_coordinates"):
        package = root / ("tulip_" + mode)
        package.mkdir(exist_ok=True)
        for filename in names:
            original = subprocess.check_output(["git", "show", f"{FROZEN_COMMIT}:{filename}"], cwd=api, text=True)
            source = original if mode == "baseline" else patched_source(Path(filename).stem, original, mode == "raw_coordinates")
            (package / Path(filename).name).write_text(source)
            if source != original:
                patches.extend(difflib.unified_diff(original.splitlines(True), source.splitlines(True), fromfile=f"baseline/{filename}", tofile=f"{mode}/{filename}"))
        packages[mode] = load_package("tulip_" + mode, package)
    (output / "qc-ablation.patch").write_text("".join(patches))
    return packages


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def metrics(rows, mode):
    scored = [row for row in rows if row[mode]["scoreable"]]
    errors = [abs(row[mode]["grade"] - row["target_mean"]) for row in scored]
    return {"total": len(rows), "scored": len(scored), "mae": float(np.mean(errors)) if errors else None,
            "within_one": float(np.mean(np.asarray(errors) <= 1.0)) if errors else None}


def evaluate(args, source, packages):
    from evaluate_tulip_methodology import adapt_sampled_frame_indices
    rows = []
    for i, original in enumerate(source["rows"], 1):
        path = cache_path(args.output / "keypoints", original["keypoints_key"])
        data = path.read_bytes()
        payload, adapted = adapt_sampled_frame_indices(json.loads(data))
        if original.get("side"):
            payload["evaluated_side"] = original["side"]
        row = copy.deepcopy(original)
        row["keypoints_sha256"] = hashlib.sha256(data).hexdigest()
        row["sampled_frame_index_adapter"] = adapted
        for mode, package in packages.items():
            try:
                result = package.run(original["test_type"], copy.deepcopy(payload))
                if result["scoreable"] and not np.isfinite([result["grade"], result["severity"]]).all():
                    raise RuntimeError("nonfinite score")
                result["evaluation_mode"] = mode
                if mode != "baseline":
                    result["capture_quality"] = "offline_qc_bypass"
                row[mode] = clean(result)
            except Exception as exc:
                row[mode] = {"scoreable": False, "grade": None, "severity": None,
                             "error": f"{type(exc).__name__}: {exc}"}
        baseline = row["baseline"]
        row["baseline_reproduced"] = (
            baseline["scoreable"] == original["scoreable"]
            and baseline["grade"] == original["grade"]
            and baseline.get("heuristic_version") == original["heuristic_version"]
            and (not original["scoreable"] or abs(4.0 * baseline["severity"] - original["continuous_grade"]) < 1e-8)
            and baseline.get("submetrics", {}).get("quality_failures", []) == original["quality_failures"]
        )
        rows.append(row)
        grades = " ".join(f"{m}={row[m]['grade']}" for m in packages)
        print(f"score {i}/{len(source['rows'])} {original['task_id']} subject={original['subject']} {original.get('side')} {grades} reproduced={row['baseline_reproduced']}", flush=True)
        (args.output / "rows.partial.json").write_text(json.dumps(clean(rows), indent=2, allow_nan=False) + "\n")
    summaries = {}
    for partition in ("full", "train", "held_out"):
        selected = [r for r in rows if partition == "full" or r["source_partition"] == partition]
        summaries[partition] = {}
        for task in sorted({r["task_id"] for r in rows}):
            task_rows = [r for r in selected if r["task_id"] == task]
            summaries[partition][task] = {
                mode: metrics(task_rows, mode) for mode in packages
            }
            rejected = [r for r in task_rows if not r["baseline"]["scoreable"]]
            summaries[partition][task]["previously_rejected"] = {
                mode: metrics(rejected, mode) for mode in packages if mode != "baseline"
            }
    artifact = {
        "protocol": {"source": str(args.baseline.resolve()), "source_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                     "frozen_api_commit": FROZEN_COMMIT, "camera_labels_partitions": "identical to source artifact",
                     "full_dataset_includes_training": True, "primary_mode": "qc_off",
                     "qc_off": "bypass recording/artifact/standing-evidence gates; preserve confidence masking, feature formulas and numerical prerequisites",
                     "raw_coordinates": "also ignore confidence thresholds during feature preprocessing; no fabricated coordinates",
                     "production_changed": False, "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "baseline_reproduced": all(r["baseline_reproduced"] for r in rows), "summaries": summaries, "rows": rows,
    }
    (args.output / "results.json").write_text(json.dumps(clean(artifact), indent=2, allow_nan=False) + "\n")
    print(json.dumps(summaries["full"], indent=2), flush=True)
    if not artifact["baseline_reproduced"]:
        raise RuntimeError("baseline mismatch: inspect per-row audit before using results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--baseline", type=Path, default=root / ".batch_runs/methodology-full-evaluation.json")
    parser.add_argument("--feral-api-dir", type=Path, default=root.parent / "feral-api")
    parser.add_argument("--output", type=Path, default=root / ".batch_runs/tulip-qc-ablation-2026-09-14")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.baseline.read_text())
    download(source["rows"], args.output / "keypoints")
    if not args.download_only:
        evaluate(args, source, prepare_scorers(args.feral_api_dir, args.output))


if __name__ == "__main__":
    main()
