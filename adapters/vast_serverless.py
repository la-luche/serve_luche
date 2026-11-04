"""Native Vast PyWorker, with a loopback-only model server and startup benchmark."""
import os
import secrets
import subprocess
import sys
from pathlib import Path


def main():
    # Vast signs public requests. The model server is reachable only on loopback.
    os.environ.setdefault("WORKER_PORT", "3000")
    os.environ.setdefault("REPORT_ADDR", "https://run.vast.ai")
    os.environ.setdefault("USE_SSL", "true")
    if not os.environ.get("MASTER_TOKEN"):
        raise RuntimeError("Vast must supply MASTER_TOKEN for a serverless worker")
    log_path = Path("/tmp/luche-vast-model.log")
    benchmark_token = secrets.token_urlsafe(32)
    env = {**os.environ, "PORT": "18000", "BENCHMARK_TOKEN": benchmark_token}
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(["/opt/conda/bin/python", "-u", "-m", "adapters.vast_model_server"],
                                env=env, stdout=log_file, stderr=subprocess.STDOUT)
        try:
            from vastai import Worker, WorkerConfig, HandlerConfig, BenchmarkConfig, LogActionConfig
            Worker(WorkerConfig(
                model_server_url="http://127.0.0.1", model_server_port=18000,
                model_log_file=str(log_path),
                handlers=[HandlerConfig(
                    route="/infer", allow_parallel_requests=False, max_queue_time=120,
                    workload_calculator=lambda p: float(p.get("num_frames", 300)),
                    benchmark_config=BenchmarkConfig(
                        dataset=[{"benchmark_token": benchmark_token, "num_frames": 8}],
                        runs=2, concurrency=1),
                )],
                log_action_config=LogActionConfig(on_load=["LUCHE_MODEL_READY"],
                                                  on_error=["Traceback (most recent call last):"]),
            )).run()
        finally:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
