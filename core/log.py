"""Tiny stdout logger with elapsed timestamps (RunPod/Vast capture stdout).

Every line is `[serve_luche +<elapsed>s] <msg>`, flushed immediately so logs
show up live in the platform console.
"""
import sys
import time

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[serve_luche +{time.time() - _T0:7.1f}s] {msg}", file=sys.stdout, flush=True)
