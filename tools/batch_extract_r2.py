#!/usr/bin/env python3
"""Resumable R2 -> RunPod -> R2 Sapiens2 batch extraction.

The controller keeps only object keys and RunPod job IDs in its state file.
Credentials and presigned URLs are never persisted. Re-running the same command
skips verified output objects and resumes submitted jobs instead of paying for
duplicate inference.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import boto3
import httpx
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError


RUNPOD_BASE = "https://api.runpod.ai/v2"
TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMED_OUT"}


def required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"missing environment variable; expected one of {names}")


@dataclass
class Item:
    source_key: str
    output_key: str
    status: str = "pending"
    job_id: str | None = None
    attempts: int = 0
    error: str | None = None
    processed_frames: int | None = None
    source_frames: int | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "Item":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in raw.items() if key in fields})


class State:
    def __init__(self, path: Path, endpoint_id: str):
        self.path = path
        self.endpoint_id = endpoint_id
        self.items: dict[str, Item] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            previous_endpoint = raw.get("endpoint_id")
            if previous_endpoint and previous_endpoint != endpoint_id:
                raise RuntimeError(
                    f"state belongs to endpoint {previous_endpoint}, not {endpoint_id}"
                )
            self.items = {
                item["source_key"]: Item.from_dict(item) for item in raw.get("items", [])
            }

    def add(self, item: Item) -> None:
        previous = self.items.get(item.source_key)
        if previous and previous.output_key != item.output_key:
            raise RuntimeError(
                f"source {item.source_key} already maps to {previous.output_key}, "
                f"not {item.output_key}"
            )
        self.items.setdefault(item.source_key, item)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "endpoint_id": self.endpoint_id,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "items": [asdict(item) for item in self.items.values()],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.path)


class R2:
    def __init__(self, source_bucket: str, output_bucket: str):
        self.source_bucket = source_bucket
        self.output_bucket = output_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=required_env("R2_ENDPOINT", "R2_ENDPOINT_URL"),
            aws_access_key_id=required_env("AWS_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
            aws_secret_access_key=required_env(
                "AWS_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"
            ),
            region_name="auto",
            config=BotoConfig(signature_version="s3v4"),
        )

    def list_videos(self, prefix: str) -> Iterable[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.source_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    yield key

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.output_bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return False
            raise

    def presign_get(self, key: str, ttl: int) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.source_bucket, "Key": key},
            ExpiresIn=ttl,
            HttpMethod="GET",
        )

    def presign_put(self, key: str, ttl: int) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.output_bucket,
                "Key": key,
                "ContentType": "application/json",
            },
            ExpiresIn=ttl,
            HttpMethod="PUT",
        )


class RunPod:
    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id
        self.api_key = required_env("RUNPOD_API_KEY")
        self.app_key = required_env("SERVE_LUCHE_API_KEY")
        self.client = httpx.Client(
            timeout=45.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.client.close()

    def ping(self) -> dict:
        response = self.client.post(
            f"{RUNPOD_BASE}/{self.endpoint_id}/runsync",
            json={"input": {"ping": True}},
        )
        response.raise_for_status()
        return response.json()

    def submit(
        self,
        video_url: str,
        result_put_url: str,
        frame_stride: int,
        batch_size: int,
    ) -> str:
        response = self.client.post(
            f"{RUNPOD_BASE}/{self.endpoint_id}/run",
            json={
                "input": {
                    "api_key": self.app_key,
                    "video_url": video_url,
                    "result_put_url": result_put_url,
                    "frame_stride": frame_stride,
                    "batch_size": batch_size,
                }
            },
        )
        response.raise_for_status()
        raw = response.json()
        job_id = raw.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod submission returned no job id: {raw}")
        return job_id

    def status(self, job_id: str) -> dict:
        response = self.client.get(
            f"{RUNPOD_BASE}/{self.endpoint_id}/status/{job_id}"
        )
        response.raise_for_status()
        return response.json()


def output_key_for(source_key: str, output_prefix: str) -> str:
    stem = source_key.rsplit(".", 1)[0]
    return f"{output_prefix.rstrip('/')}/{stem}.json"


def select_source_keys(args: argparse.Namespace, r2: R2) -> list[str]:
    keys = list(dict.fromkeys(args.source_key))
    if args.source_key_file:
        file_keys = [
            line.strip()
            for line in args.source_key_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        keys.extend(file_keys)
    if args.source_prefix:
        pattern = re.compile(args.include_regex) if args.include_regex else None
        for key in r2.list_videos(args.source_prefix):
            if pattern is None or pattern.search(key):
                keys.append(key)
    keys = list(dict.fromkeys(keys))
    if args.limit:
        keys = keys[: args.limit]
    if not keys:
        raise RuntimeError("selection found no source videos")
    return keys


def print_summary(items: Iterable[Item]) -> None:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    summary = " ".join(f"{name}={counts[name]}" for name in sorted(counts))
    print(f"summary {summary}", flush=True)


def run(args: argparse.Namespace) -> int:
    r2 = R2(args.source_bucket, args.output_bucket)
    state = State(args.state, args.endpoint_id)
    source_keys = select_source_keys(args, r2)
    for source_key in source_keys:
        state.add(
            Item(
                source_key=source_key,
                output_key=output_key_for(source_key, args.output_prefix),
            )
        )

    selected = [state.items[key] for key in source_keys]
    for item in selected:
        if r2.exists(item.output_key):
            item.status = "done"
            item.job_id = None
            item.error = None
    state.save()

    print(f"selected {len(selected)} videos; state={args.state}", flush=True)
    print_summary(selected)
    if args.dry_run:
        for item in selected:
            print(f"{item.status}\t{item.source_key}\t{item.output_key}")
        return 0

    runpod = RunPod(args.endpoint_id)
    try:
        ping = runpod.ping()
        output = ping.get("output", ping)
        print(
            "endpoint "
            f"status={output.get('status')} model={output.get('model_size')} "
            f"git_sha={output.get('git_sha')}",
            flush=True,
        )

        while True:
            unfinished = [item for item in selected if item.status != "done"]
            if not unfinished:
                print_summary(selected)
                return 0

            active = [
                item
                for item in unfinished
                if item.job_id and item.status in {"submitted", "queued", "running"}
            ]
            capacity = max(0, args.max_in_flight - len(active))
            candidates = [
                item
                for item in unfinished
                if not item.job_id and item.attempts < args.max_attempts
            ][:capacity]

            for item in candidates:
                try:
                    item.job_id = runpod.submit(
                        r2.presign_get(item.source_key, args.url_ttl_seconds),
                        r2.presign_put(item.output_key, args.url_ttl_seconds),
                        args.frame_stride,
                        args.batch_size,
                    )
                    item.attempts += 1
                    item.status = "submitted"
                    item.error = None
                    print(
                        f"submit job={item.job_id} attempt={item.attempts} "
                        f"source={item.source_key}",
                        flush=True,
                    )
                    state.save()
                except Exception as exc:
                    item.error = f"submit: {type(exc).__name__}: {exc}"
                    print(f"ERROR {item.source_key}: {item.error}", file=sys.stderr, flush=True)
                    state.save()

            active = [item for item in unfinished if item.job_id]
            for item in active:
                try:
                    raw = runpod.status(item.job_id)
                    status = raw.get("status")
                    if status == "COMPLETED":
                        if not r2.exists(item.output_key):
                            raise RuntimeError("RunPod completed but output object is missing")
                        result = raw.get("output") or {}
                        item.status = "done"
                        item.processed_frames = result.get("num_processed_frames")
                        item.source_frames = result.get("num_source_frames")
                        item.error = None
                        print(
                            f"done job={item.job_id} frames={item.processed_frames} "
                            f"source={item.source_key}",
                            flush=True,
                        )
                        item.job_id = None
                        state.save()
                    elif status in TERMINAL_FAILURES:
                        item.status = "pending" if item.attempts < args.max_attempts else "failed"
                        item.error = json.dumps(raw.get("error") or raw.get("output") or raw)
                        print(
                            f"ERROR job={item.job_id} status={status} source={item.source_key}",
                            file=sys.stderr,
                            flush=True,
                        )
                        item.job_id = None
                        state.save()
                    else:
                        next_status = "running" if status == "IN_PROGRESS" else "queued"
                        if item.status != next_status:
                            item.status = next_status
                            print(
                                f"status job={item.job_id} {status} source={item.source_key}",
                                flush=True,
                            )
                            state.save()
                except Exception as exc:
                    item.error = f"poll: {type(exc).__name__}: {exc}"
                    print(f"WARN {item.source_key}: {item.error}", file=sys.stderr, flush=True)

            exhausted = [
                item
                for item in unfinished
                if not item.job_id and item.attempts >= args.max_attempts
            ]
            if exhausted and len(exhausted) == len(unfinished):
                for item in exhausted:
                    item.status = "failed"
                state.save()
                print_summary(selected)
                return 1
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("interrupted; submitted jobs remain in state and will be resumed", flush=True)
        state.save()
        return 130
    finally:
        runpod.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bucket", default="feral")
    parser.add_argument("--output-bucket", default="feral")
    parser.add_argument("--output-prefix", default="mds-updrs-keypoints-5b")
    parser.add_argument("--endpoint-id", default="v29lgubwpc998d")
    parser.add_argument("--source-key", action="append", default=[])
    parser.add_argument("--source-key-file", type=Path)
    parser.add_argument("--source-prefix")
    parser.add_argument("--include-regex")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-in-flight", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--url-ttl-seconds", type=int, default=172800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.source_key and not args.source_key_file and not args.source_prefix:
        parser.error("provide --source-key, --source-key-file, and/or --source-prefix")
    if args.frame_stride < 1 or args.batch_size < 1 or args.max_in_flight < 1:
        parser.error("frame stride, batch size, and max in-flight must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
