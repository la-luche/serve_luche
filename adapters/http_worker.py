"""Portable RunPod-compatible HTTP queue for a dedicated GPU worker.

Run this same image on a normal Vast instance or the lab RTX 5080:

  GET  /healthz          build fingerprint, queue depth, and busy state
  POST /run              enqueue {"input": {...}}; return a job id
  GET  /status/{job_id}  RunPod-compatible job status and output
  POST /cancel/{job_id}  cancel queued work or stop between GPU batches
  POST /runsync          synchronous RunPod-compatible request
  POST /infer            synchronous flat request

Only the small status/result record is persisted. API keys and presigned video
URLs remain in memory and are never written to disk. If the process restarts,
previously active jobs become FAILED so the caller can safely retry them.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from core import auth
from core.errors import InferenceCancelled
from core.log import log


class InferRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    api_key: str = Field(min_length=1)
    video_url: str = Field(min_length=1)
    result_put_url: str | None = None
    frame_stride: int = Field(default=1, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    person_detection: StrictBool = False
    person_detection_stride: int = Field(default=5, ge=1)
    person_box_overflow: float = Field(default=0.25, ge=0.0, le=2.0)
    person_detection_threshold: float = Field(default=0.3, gt=0.0, le=1.0)


def _execute(req: InferRequest, cancelled: threading.Event) -> dict:
    from core.infer import DEFAULT_BATCH_SIZE, run_video

    return run_video(
        req.video_url,
        result_put_url=req.result_put_url,
        frame_stride=req.frame_stride,
        batch_size=req.batch_size or DEFAULT_BATCH_SIZE,
        person_detection=req.person_detection,
        person_detection_stride=req.person_detection_stride,
        person_box_overflow=req.person_box_overflow,
        person_detection_threshold=req.person_detection_threshold,
        cancel_check=cancelled.is_set,
    )


class JobManager:
    _ACTIVE = {"IN_QUEUE", "IN_PROGRESS"}

    def __init__(
        self,
        state_dir: str,
        *,
        executor: Callable[[InferRequest, threading.Event], dict] = _execute,
        max_queue: int = 20,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._requests: dict[str, InferRequest] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._done_events: dict[str, threading.Event] = {}
        self._load_state()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="serve-luche-gpu-worker",
            daemon=True,
        )
        self._thread.start()

    def _path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _persist_locked(self, job_id: str) -> None:
        record = self._jobs[job_id]
        path = self._path(job_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, separators=(",", ":")))
        os.replace(tmp, path)

    def _load_state(self) -> None:
        for path in self.state_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text())
                job_id = record["id"]
                if record.get("status") in self._ACTIVE:
                    record["status"] = "FAILED"
                    record["error"] = "worker restarted before job completed"
                    record["updated_at"] = time.time()
                self._jobs[job_id] = record
                self._persist_locked(job_id)
            except Exception as exc:
                log(f"ignoring corrupt HTTP job state {path}: {exc}")

    def submit(self, req: InferRequest) -> str:
        job_id = uuid.uuid4().hex
        record = {
            "id": job_id,
            "status": "IN_QUEUE",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with self._lock:
            self._jobs[job_id] = record
            self._requests[job_id] = req
            self._cancel_events[job_id] = threading.Event()
            self._done_events[job_id] = threading.Event()
            self._persist_locked(job_id)
            try:
                self._queue.put_nowait(job_id)
            except queue.Full:
                record["status"] = "FAILED"
                record["error"] = "worker queue is full"
                record["updated_at"] = time.time()
                self._persist_locked(job_id)
                self._done_events[job_id].set()
                self._requests.pop(job_id, None)
                self._cancel_events.pop(job_id, None)
        return job_id

    def _set_terminal(self, job_id: str, status: str, **fields) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.update(fields)
            record["status"] = status
            record["updated_at"] = time.time()
            self._persist_locked(job_id)
            self._done_events[job_id].set()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                return
            with self._lock:
                record = self._jobs[job_id]
                if record["status"] == "CANCELLED":
                    self._requests.pop(job_id, None)
                    self._cancel_events.pop(job_id, None)
                    self._queue.task_done()
                    continue
                record["status"] = "IN_PROGRESS"
                record["updated_at"] = time.time()
                self._persist_locked(job_id)
                req = self._requests[job_id]
                cancelled = self._cancel_events[job_id]
            try:
                log(f"HTTP job {job_id}: started")
                output = self.executor(req, cancelled)
                if cancelled.is_set():
                    self._set_terminal(job_id, "CANCELLED")
                else:
                    output["git_sha"] = os.environ.get("GIT_SHA", "unknown")
                    self._set_terminal(job_id, "COMPLETED", output=output)
                    log(f"HTTP job {job_id}: completed")
            except Exception as exc:
                if isinstance(exc, InferenceCancelled) or cancelled.is_set():
                    self._set_terminal(job_id, "CANCELLED")
                    log(f"HTTP job {job_id}: cancelled")
                else:
                    error = f"{type(exc).__name__}: {exc}"
                    self._set_terminal(job_id, "FAILED", error=error)
                    log(f"HTTP job {job_id}: failed: {error}")
            finally:
                with self._lock:
                    self._requests.pop(job_id, None)
                    self._cancel_events.pop(job_id, None)
                self._queue.task_done()

    def status(self, job_id: str) -> dict:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return {"id": job_id, "status": "FAILED", "error": "job not found"}
            return dict(record)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return {"id": job_id, "status": "FAILED", "error": "job not found"}
            if record["status"] == "IN_QUEUE":
                record["status"] = "CANCELLED"
                record["updated_at"] = time.time()
                self._persist_locked(job_id)
                self._done_events[job_id].set()
            elif record["status"] == "IN_PROGRESS":
                record["cancellation_requested"] = True
                record["updated_at"] = time.time()
                self._cancel_events[job_id].set()
                self._persist_locked(job_id)
            return dict(record)

    def wait(self, job_id: str) -> dict:
        with self._lock:
            event = self._done_events[job_id]
        event.wait()
        return self.status(job_id)

    def health(self) -> dict:
        with self._lock:
            busy = any(job.get("status") == "IN_PROGRESS" for job in self._jobs.values())
        return {"busy": busy, "queue_depth": self._queue.qsize()}

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)


_STATE_DIR = os.environ.get("JOB_STATE_DIR", "/tmp/serve-luche-jobs")
_MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "20"))
MANAGER = JobManager(_STATE_DIR, max_queue=_MAX_QUEUE)
app = FastAPI(title="serve_luche")


def _parse_wrapped(payload: dict) -> InferRequest | dict:
    inp = payload.get("input") if isinstance(payload, dict) else None
    if not isinstance(inp, dict):
        return {
            "error": "invalid input: input must be a JSON object",
            "git_sha": os.environ.get("GIT_SHA"),
        }
    try:
        return InferRequest.model_validate(inp)
    except ValidationError as exc:
        return {
            "error": f"invalid input: {exc.errors(include_url=False)}",
            "git_sha": os.environ.get("GIT_SHA"),
        }


def _authorized(req: InferRequest) -> bool:
    if auth.check(req.api_key):
        return True
    log("HTTP job rejected: unauthorized")
    return False


@app.get("/healthz")
def healthz():
    return {**auth.version(), **MANAGER.health()}


@app.post("/run")
def run(payload: dict):
    inp = payload.get("input") if isinstance(payload, dict) else None
    if isinstance(inp, dict) and inp.get("ping"):
        return auth.version()
    req = _parse_wrapped(payload)
    if isinstance(req, dict):
        return req
    if not _authorized(req):
        return {"error": "unauthorized"}
    job_id = MANAGER.submit(req)
    return MANAGER.status(job_id)


@app.get("/status/{job_id}")
def status(job_id: str):
    return MANAGER.status(job_id)


@app.post("/cancel/{job_id}")
def cancel(job_id: str):
    return MANAGER.cancel(job_id)


@app.post("/infer")
def infer(req: InferRequest):
    if not _authorized(req):
        return {"error": "unauthorized"}
    state = MANAGER.wait(MANAGER.submit(req))
    return state.get("output") or {"error": state.get("error", state["status"])}


@app.post("/runsync")
def runsync(payload: dict):
    inp = payload.get("input") if isinstance(payload, dict) else None
    if isinstance(inp, dict) and inp.get("ping"):
        return auth.version()
    req = _parse_wrapped(payload)
    if isinstance(req, dict):
        return req
    if not _authorized(req):
        return {"error": "unauthorized"}
    state = MANAGER.wait(MANAGER.submit(req))
    return state.get("output") or {"error": state.get("error", state["status"])}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
