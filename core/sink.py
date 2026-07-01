"""Write the results JSON out — presigned R2 PUT, or a local file.

Using a presigned PUT means the container holds NO R2 credentials, so the same
image runs identically on RunPod, Vast, or anywhere else.
"""
from __future__ import annotations

import json
import os

import requests


def emit(obj: dict, *, put_url: str | None = None, local_path: str | None = None) -> str:
    """Serialize `obj` to JSON and send it to the sink. Returns a location string."""
    payload = json.dumps(obj, separators=(",", ":")).encode("utf-8")

    if put_url:
        resp = requests.put(
            put_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        return put_url.split("?", 1)[0]  # strip presign query for logging

    if local_path:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(payload)
        return local_path

    raise ValueError("emit() needs either put_url or local_path")
