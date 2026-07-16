"""Static API-key auth. Only the SHA-256 of the key is baked into the (public)
image; we compare hashes so the key itself is never present in the image/repo.
"""
import hashlib
import hmac
import os

_EXPECTED = os.environ.get("API_KEY_SHA256", "")
GIT_SHA = os.environ.get("GIT_SHA", "unknown")


def check(api_key: str | None) -> bool:
    if not _EXPECTED:
        return False
    digest = hashlib.sha256((api_key or "").encode()).hexdigest()
    return hmac.compare_digest(digest, _EXPECTED)  # constant-time


def version() -> dict:
    """Unauthenticated build fingerprint — confirm what a container is running."""
    return {
        "status": "ok",
        "git_sha": GIT_SHA,
        "api_key_sha256": _EXPECTED,
        "model_size": os.environ.get("MODEL_SIZE", "5b"),
        "person_detector": os.environ.get(
            "PERSON_DETECTOR_NAME", "facebook/detr-resnet-101-dc5"
        ),
    }
