"""Optional sparse person detection and temporally stable bounding boxes.

The Sapiens pose model is top-down: it expects a reasonably tight person box.
DETR runs on a sparse set of source frames, detections are associated into
tracks, and the selected track is interpolated for every source frame. The
heavy detector objects are loaded lazily so the default full-frame path and
health checks pay no detector startup cost.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .log import log

_BAKED_MODEL = "/opt/models/detr-resnet-101-dc5"
DEFAULT_MODEL = os.environ.get("PERSON_DETECTOR_MODEL") or (
    _BAKED_MODEL
    if os.path.isdir(_BAKED_MODEL)
    else "facebook/detr-resnet-101-dc5"
)
DEFAULT_BATCH_SIZE = int(os.environ.get("PERSON_DETECTION_BATCH_SIZE", "8"))


@dataclass(frozen=True)
class PersonDetection:
    frame_idx: int
    box: np.ndarray
    score: float


@dataclass
class _Track:
    observations: list[PersonDetection] = field(default_factory=list)

    @property
    def last(self) -> PersonDetection:
        return self.observations[-1]


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def _association_score(a: np.ndarray, b: np.ndarray, image_diagonal: float) -> float:
    distance = float(np.linalg.norm(_box_center(a) - _box_center(b))) / image_diagonal
    area_a, area_b = max(_box_area(a), 1.0), max(_box_area(b), 1.0)
    size_change = abs(math.log(area_a / area_b))
    return 2.5 * _iou(a, b) - 2.0 * distance - 0.25 * size_change


def _build_tracks(
    samples: list[tuple[int, list[tuple[np.ndarray, float]]]],
    *,
    width: int,
    height: int,
    max_gap_frames: int,
) -> list[_Track]:
    """Greedily associate detections without switching to an edge bystander."""
    tracks: list[_Track] = []
    image_diagonal = max(math.hypot(width, height), 1.0)

    for frame_idx, candidates in samples:
        detections = [
            PersonDetection(frame_idx, np.asarray(box, dtype=np.float32), float(score))
            for box, score in candidates
        ]
        active = [
            (index, track)
            for index, track in enumerate(tracks)
            if frame_idx - track.last.frame_idx <= max_gap_frames
        ]
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in active:
            for detection_index, detection in enumerate(detections):
                score = _association_score(
                    track.last.box, detection.box, image_diagonal
                )
                # Negative scores are still useful for gradual scale changes, but
                # reject a clearly unrelated person on the other side of the frame.
                center_distance = float(
                    np.linalg.norm(
                        _box_center(track.last.box) - _box_center(detection.box)
                    )
                ) / image_diagonal
                if _iou(track.last.box, detection.box) > 0.01 or center_distance < 0.22:
                    pairs.append((score, track_index, detection_index))

        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for _, track_index, detection_index in sorted(pairs, reverse=True):
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue
            tracks[track_index].observations.append(detections[detection_index])
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index not in assigned_detections:
                tracks.append(_Track([detection]))
    return tracks


def _track_score(
    track: _Track,
    *,
    width: int,
    height: int,
    num_samples: int,
    frame_span: int,
) -> float:
    boxes = np.stack([observation.box for observation in track.observations])
    scores = np.asarray([observation.score for observation in track.observations])
    image_area = max(float(width * height), 1.0)
    image_center = np.asarray([width * 0.5, height * 0.5])
    image_diagonal = max(math.hypot(width, height), 1.0)
    centers = np.stack([_box_center(box) for box in boxes])
    areas = np.asarray([_box_area(box) / image_area for box in boxes])
    centrality = 1.0 - np.clip(
        np.linalg.norm(centers - image_center, axis=1) / (0.5 * image_diagonal),
        0.0,
        1.0,
    )
    motion = (
        float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()) / image_diagonal
        if len(centers) > 1
        else 0.0
    )
    coverage = len(track.observations) / max(num_samples, 1)
    span = (
        track.observations[-1].frame_idx - track.observations[0].frame_idx + 1
    )
    # Persistence dominates; center/area reject partial edge bystanders, while a
    # capped motion bonus helps select the patient in walking clips.
    return (
        4.0 * coverage
        + min(span / max(frame_span, 1), 1.0)
        + float(scores.mean())
        + float(np.sqrt(np.clip(areas.mean(), 0.0, 1.0)))
        + float(centrality.mean())
        + min(motion, 1.0)
    )


def select_person_track(
    samples: list[tuple[int, list[tuple[np.ndarray, float]]]],
    *,
    width: int,
    height: int,
    detection_stride: int,
) -> _Track | None:
    """Return the most persistent, central person track across the whole clip."""
    tracks = _build_tracks(
        samples,
        width=width,
        height=height,
        max_gap_frames=max(3 * detection_stride, 1),
    )
    if not tracks:
        return None
    frame_span = samples[-1][0] - samples[0][0] + 1 if samples else 1
    return max(
        tracks,
        key=lambda track: _track_score(
            track,
            width=width,
            height=height,
            num_samples=len(samples),
            frame_span=frame_span,
        ),
    )


def interpolate_track(track: _Track, num_frames: int) -> np.ndarray:
    """Interpolate a track as center + log-size to keep every box well formed."""
    observations = track.observations
    source = np.asarray([observation.frame_idx for observation in observations])
    boxes = np.stack([observation.box for observation in observations])
    centers = np.stack([_box_center(box) for box in boxes])
    sizes = np.maximum(boxes[:, 2:4] - boxes[:, 0:2], 2.0)

    # A symmetric three-point filter removes detector jitter without temporal lag.
    if len(observations) >= 3:
        centers[1:-1] = (
            0.25 * centers[:-2] + 0.5 * centers[1:-1] + 0.25 * centers[2:]
        )
        log_sizes = np.log(sizes)
        log_sizes[1:-1] = (
            0.25 * log_sizes[:-2]
            + 0.5 * log_sizes[1:-1]
            + 0.25 * log_sizes[2:]
        )
    else:
        log_sizes = np.log(sizes)

    target = np.arange(num_frames)
    center_series = np.column_stack(
        [np.interp(target, source, centers[:, axis]) for axis in range(2)]
    )
    size_series = np.exp(
        np.column_stack(
            [np.interp(target, source, log_sizes[:, axis]) for axis in range(2)]
        )
    )
    return np.concatenate(
        [center_series - 0.5 * size_series, center_series + 0.5 * size_series],
        axis=1,
    ).astype(np.float32)


class PersonDetector:
    def __init__(self, device: str, model_source: str = DEFAULT_MODEL):
        from transformers import AutoImageProcessor, DetrForObjectDetection

        local_only = os.path.isdir(model_source)
        log(f"loading person detector from {model_source} on {device}")
        started = time.time()
        self.processor = AutoImageProcessor.from_pretrained(
            model_source, local_files_only=local_only, use_fast=False
        )
        self.model = DetrForObjectDetection.from_pretrained(
            model_source,
            local_files_only=local_only,
            # The full DETR checkpoint already contains the backbone. Leaving
            # this true makes Transformers try to fetch a second timm ResNet at
            # worker startup even when the DETR directory is baked and offline.
            use_pretrained_backbone=False,
        ).to(device)
        self.model.eval()
        self.device = device
        labels = {
            int(index): str(name).lower()
            for index, name in self.model.config.id2label.items()
        }
        person_ids = [index for index, name in labels.items() if name == "person"]
        if not person_ids:
            raise RuntimeError("person detector config has no 'person' class")
        self.person_id = person_ids[0]
        log(f"person detector loaded in {time.time() - started:.1f}s")

    @torch.inference_mode()
    def detect(
        self, frames: torch.Tensor, threshold: float
    ) -> list[list[tuple[np.ndarray, float]]]:
        """Detect people in uint8 RGB frames shaped (B,3,H,W)."""
        _, _, height, width = frames.shape
        inputs = self.processor(
            images=[frame for frame in frames], return_tensors="pt"
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor(
            [[height, width]] * len(frames), device=self.device
        )
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )
        batches: list[list[tuple[np.ndarray, float]]] = []
        for result in results:
            labels = result["labels"].detach().cpu().numpy()
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            batches.append(
                [
                    (box.astype(np.float32), float(score))
                    for box, score, label in zip(boxes, scores, labels)
                    if int(label) == self.person_id
                ]
            )
        return batches


_DETECTORS: dict[tuple[str, str], PersonDetector] = {}


def _get_detector(device: str) -> PersonDetector:
    key = (device, DEFAULT_MODEL)
    if key not in _DETECTORS:
        _DETECTORS[key] = PersonDetector(device, DEFAULT_MODEL)
    return _DETECTORS[key]


def detect_person_track(
    decoder,
    *,
    device: str,
    detection_stride: int,
    threshold: float,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray | None, dict]:
    """Detect, track, and interpolate one patient box for every source frame."""
    indices = list(range(0, decoder.num_frames, detection_stride))
    if indices[-1] != decoder.num_frames - 1:
        indices.append(decoder.num_frames - 1)
    detector = _get_detector(device)
    samples: list[tuple[int, list[tuple[np.ndarray, float]]]] = []
    started = time.time()
    detection_count = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        frames = decoder.get_batch(chunk, device="cpu")
        candidates = detector.detect(frames, threshold)
        for frame_idx, frame_candidates in zip(chunk, candidates):
            samples.append((frame_idx, frame_candidates))
            detection_count += len(frame_candidates)

    track = select_person_track(
        samples,
        width=decoder.width,
        height=decoder.height,
        detection_stride=detection_stride,
    )
    elapsed = time.time() - started
    stats = {
        "detection_frames": len(indices),
        "person_detections": detection_count,
        "selected_track_observations": len(track.observations) if track else 0,
        "fallback_full_frame": track is None,
        "seconds": round(elapsed, 2),
    }
    log(
        "person detection "
        f"frames={len(indices)} detections={detection_count} "
        f"track={stats['selected_track_observations']} time={elapsed:.1f}s"
    )
    return (interpolate_track(track, decoder.num_frames) if track else None), stats
