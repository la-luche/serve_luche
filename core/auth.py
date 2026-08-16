"""Static API-key auth. Only the SHA-256 of the key is baked into the (public)
image; we compare hashes so the key itself is never present in the image/repo.
"""
import hashlib
import hmac
import os

_EXPECTED = os.environ.get("API_KEY_SHA256", "")
GIT_SHA = os.environ.get("GIT_SHA", "unknown")


def _enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def check(api_key: str | None) -> bool:
    if not _EXPECTED or not isinstance(api_key, str):
        return False
    digest = hashlib.sha256(api_key.encode()).hexdigest()
    return hmac.compare_digest(digest, _EXPECTED)  # constant-time


def version() -> dict:
    """Unauthenticated build fingerprint — confirm what a container is running."""
    return {
        "status": "ok",
        "git_sha": GIT_SHA,
        "api_key_sha256": _EXPECTED,
        "model_size": os.environ.get("MODEL_SIZE", "5b"),
        "batch_size": int(os.environ.get("BATCH_SIZE", "1")),
        "compile": _enabled("COMPILE", False),
        "preload_model": _enabled("PRELOAD_MODEL", True),
        "warmup_on_start": _enabled("WARMUP_ON_START", True),
        "model_load_device": os.environ.get("MODEL_LOAD_DEVICE", "auto"),
        "sapiens_model_revision": os.environ.get(
            "SAPIENS_MODEL_REVISION", "unknown"
        ),
        "person_detector": os.environ.get(
            "PERSON_DETECTOR_NAME", "facebook/detr-resnet-101-dc5"
        ),
        "person_detector_revision": os.environ.get(
            "PERSON_DETECTOR_REVISION", "unknown"
        ),
    }
