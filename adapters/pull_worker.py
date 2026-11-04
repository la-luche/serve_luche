"""One resident GPU consuming jobs from the Pi's authenticated research queue."""
import os
import signal
import threading
import time
import uuid

import requests

from adapters.runpod_handler import prepare_worker
from core.log import log


def main():
    base = os.environ["ROUTER_URL"].rstrip("/")
    backend = os.environ["ROUTER_BACKEND"]
    token = os.environ["ROUTER_WORKER_TOKEN"]
    worker_id = os.environ.get("WORKER_ID", f"{backend}-{uuid.uuid4().hex[:12]}")
    headers = {"Authorization": f"Bearer {token}"}
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    prepare_worker()
    from core.infer import run_video
    from core.request import parse_infer_input
    import torch

    info = {"worker_id": worker_id, "git_sha": os.environ.get("GIT_SHA"),
            "gpu": torch.cuda.get_device_name(), "ready": True}

    def post(path, data):
        response = requests.post(base + path, headers=headers, json=data, timeout=20)
        response.raise_for_status()
        return response.json()

    log(f"pull worker ready: backend={backend} worker={worker_id}")
    while not stopping.is_set():
        try:
            job = post(f"/workers/{backend}/claim", info).get("job")
        except Exception as exc:
            log(f"queue unavailable: {type(exc).__name__}")
            stopping.wait(5)
            continue
        if not job:
            stopping.wait(2)
            continue
        finished = threading.Event()
        cancelled = threading.Event()
        heartbeat_ok = [time.monotonic()]
        claim = {"worker_id": worker_id, "lease": job["lease"]}

        def heartbeat():
            while not finished.wait(5):
                try:
                    status = post(f"/workers/{backend}/jobs/{job['id']}/heartbeat", claim)
                    heartbeat_ok[0] = time.monotonic()
                    if status.get("cancel_requested"):
                        cancelled.set()
                except Exception:
                    # Fail closed before a lost worker can continue indefinitely.
                    if time.monotonic() - heartbeat_ok[0] > 60:
                        cancelled.set()

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            video, kwargs = parse_infer_input(job["input"])
            output = run_video(video, **kwargs, cancel_check=cancelled.is_set)
            output["git_sha"] = os.environ.get("GIT_SHA", "unknown")
            report = {**claim, "status": "CANCELLED" if cancelled.is_set() else "COMPLETED", "output": output}
        except Exception as exc:
            report = {**claim, "status": "CANCELLED" if cancelled.is_set() else "FAILED",
                      "error": type(exc).__name__}
        # Keep ownership heartbeat while retrying acknowledgement; never re-run.
        acknowledged = False
        for _ in range(24):
            try:
                post(f"/workers/{backend}/jobs/{job['id']}/complete", report)
                acknowledged = True
                break
            except Exception:
                time.sleep(5)
        finished.set()
        thread.join(timeout=6)
        if not acknowledged:
            raise RuntimeError("completion unacknowledged; exiting without claiming more work")


if __name__ == "__main__":
    main()
