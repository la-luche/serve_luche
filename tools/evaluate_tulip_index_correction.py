#!/usr/bin/env python3
"""Correct the TULIP finger-task join while preserving the frozen QC ablation.

Tasks 7/8 are sequential opposition, not the clinical thumb-index task (9/10).
Recompute just those 24 rows; retain all other results, subjects, partitions,
cameras, clinician targets and frozen heuristic coefficients. No split edits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_tulip_methodology import R2, adapt_sampled_frame_indices, build_examples
from evaluate_tulip_qc_ablation import (
    FROZEN_COMMIT, cache_path, clean, download, metrics, prepare_scorers, r2_client,
)


def run(args):
    old = json.loads(args.original.read_text())
    r2 = R2.__new__(R2)
    r2.bucket, r2.client = 'feral', r2_client()
    corrected = {}
    for partition, source_partition in [('train', 'train'), ('val', 'held_out')]:
        for example in build_examples(r2, partition):
            if example.task_id == 'finger-tapping':
                corrected[source_partition, example.subject, example.side] = example
    original_fingers = [r for r in old['rows'] if r['task_id'] == 'finger-tapping']
    assert len(corrected) == len(original_fingers) == 24
    assert set(corrected) == {
        (r['source_partition'], r['subject'], r['side']) for r in original_fingers
    }, 'participant membership changed'
    args.output.mkdir(parents=True, exist_ok=True)
    download([{'keypoints_key': e.keypoints_key} for e in corrected.values()], args.output / 'keypoints')
    packages = prepare_scorers(args.feral_api_dir, args.output)
    rows = []
    for original in old['rows']:
        if original['task_id'] != 'finger-tapping':
            rows.append(copy.deepcopy(original))
            continue
        example = corrected[original['source_partition'], original['subject'], original['side']]
        assert example.target_name == original['target_name']
        assert example.keypoints_key.rsplit('/', 1)[1] == original['keypoints_key'].rsplit('/', 1)[1]
        raw = cache_path(args.output / 'keypoints', example.keypoints_key).read_bytes()
        payload = json.loads(raw)
        old_payload = json.loads(cache_path(args.original.parent / 'keypoints', original['keypoints_key']).read_text())
        provenance_fields = [
            'git_sha', 'model_size', 'sapiens_model_revision', 'person_detector_revision',
            'frame_stride', 'source_fps', 'keypoint_format', 'num_keypoints',
        ]
        assert {k: payload.get(k) for k in provenance_fields} == {
            k: old_payload.get(k) for k in provenance_fields
        }, 'extractor or sampling profile changed'
        # Compare detector configuration, not per-video runtime/count telemetry.
        detector_fields = ['enabled', 'stride', 'box_overflow', 'threshold', 'device']
        assert {k: payload['person_detection'].get(k) for k in detector_fields} == {
            k: old_payload['person_detection'].get(k) for k in detector_fields
        }, 'person detector configuration changed'
        row = copy.deepcopy(original)
        row.pop('baseline_reproduced', None)
        row['superseded_wrong_exercise'] = {
            'keypoints_key': original['keypoints_key'],
            **{mode: original[mode] for mode in packages},
        }
        row['keypoints_key'] = example.keypoints_key
        row['keypoints_sha256'] = hashlib.sha256(raw).hexdigest()
        row['keypoint_metadata'] = {k: v for k, v in payload.items() if k != 'frames'}
        payload, adapted = adapt_sampled_frame_indices(payload)
        row['sampled_frame_index_adapter'] = adapted
        payload['evaluated_side'] = example.side
        for mode, package in packages.items():
            result = package.run(example.test_type, copy.deepcopy(payload))
            if result['scoreable']:
                assert np.isfinite([result['grade'], result['severity']]).all()
            result['evaluation_mode'] = mode
            if mode != 'baseline':
                result['capture_quality'] = 'offline_qc_bypass'
            row[mode] = clean(result)
        baseline = row['baseline']
        row.update(
            scoreable=baseline['scoreable'], grade=baseline['grade'],
            continuous_grade=4 * baseline['severity'] if baseline['scoreable'] else None,
            confidence=baseline.get('confidence'), heuristic_version=baseline.get('heuristic_version'),
            quality_failures=baseline.get('submetrics', {}).get('quality_failures', []), error=None,
        )
        rows.append(row)
        print(f"subject={example.subject:02d} {example.side} target={row['target_mean']:.3f} "
              f"QC={row['baseline']['grade']} QC-off={row['qc_off']['grade']}", flush=True)
    assert len(rows) == 156
    assert all(a == b for a, b in zip(old['rows'], rows) if a['task_id'] != 'finger-tapping')
    summaries = {}
    for partition in ['full', 'train', 'held_out']:
        selected = [r for r in rows if partition == 'full' or r['source_partition'] == partition]
        summaries[partition] = {}
        for task in sorted({r['task_id'] for r in rows}):
            group = [r for r in selected if r['task_id'] == task]
            summaries[partition][task] = {mode: metrics(group, mode) for mode in packages}
    artifact = {
        'protocol': {
            'supersedes': str(args.original.resolve()),
            'supersedes_sha256': hashlib.sha256(args.original.read_bytes()).hexdigest(),
            'correction': 'finger-tapping videos: sequential opposition tasks 7/8 replaced by thumb-index tasks 9/10',
            'frozen_api_commit': FROZEN_COMMIT,
            'unchanged': '156 subject/task/side rows; partitions, cameras, clinical ratings, extractor build/weights/settings, scorer coefficients',
            'full_dataset_includes_training': True,
            'primary_mode': 'qc_off',
            'production_changed': False,
            'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        'other_132_rows_identical': True, 'summaries': summaries, 'rows': rows,
    }
    (args.output / 'results.json').write_text(json.dumps(clean(artifact), indent=2, allow_nan=False) + '\n')
    print(json.dumps({p: summaries[p]['finger-tapping'] for p in summaries}, indent=2))


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=root / '.batch_runs/tulip-qc-ablation-2026-09-14/results.json')
    parser.add_argument('--output', type=Path, default=root / '.batch_runs/tulip-index-correction-2026-09-14')
    parser.add_argument('--feral-api-dir', type=Path, default=root.parent / 'feral-api')
    run(parser.parse_args())
