"""Strict request parsing that never imports or initializes the pose model."""
from __future__ import annotations

import math


def _positive_integer(value, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a JSON integer")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _bounded_number(
    value,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool,
) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    converted = float(value)
    lower_ok = converted >= lower if lower_inclusive else converted > lower
    if not lower_ok or converted > upper:
        left = "[" if lower_inclusive else "("
        raise ValueError(f"{name} must be in {left}{lower}, {upper}]")
    return converted


def parse_infer_input(inp: dict) -> tuple[str, dict]:
    """Validate all inference fields before the heavyweight model import."""
    if not isinstance(inp, dict):
        raise ValueError("input must be a JSON object")
    video_url = inp.get("video_url")
    if not isinstance(video_url, str) or not video_url.strip():
        raise ValueError("video_url must be a non-empty string")

    result_put_url = inp.get("result_put_url")
    if result_put_url is not None and (
        not isinstance(result_put_url, str) or not result_put_url.strip()
    ):
        raise ValueError("result_put_url must be a non-empty string or null")

    person_detection = inp.get("person_detection", False)
    if type(person_detection) is not bool:
        raise ValueError("person_detection must be a JSON boolean")

    kwargs = {
        "result_put_url": result_put_url,
        "frame_stride": _positive_integer(
            inp.get("frame_stride", 1), name="frame_stride"
        ),
        "person_detection": person_detection,
        "person_detection_stride": _positive_integer(
            inp.get("person_detection_stride", 5),
            name="person_detection_stride",
        ),
        "person_box_overflow": _bounded_number(
            inp.get("person_box_overflow", 0.25),
            name="person_box_overflow",
            lower=0.0,
            upper=2.0,
            lower_inclusive=True,
        ),
        "person_detection_threshold": _bounded_number(
            inp.get("person_detection_threshold", 0.3),
            name="person_detection_threshold",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        ),
    }
    if "batch_size" in inp and inp["batch_size"] is not None:
        kwargs["batch_size"] = _positive_integer(
            inp["batch_size"], name="batch_size"
        )
    return video_url, kwargs
