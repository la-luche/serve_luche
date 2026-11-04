"""Native Vast PyWorker, with a loopback-only model server and startup benchmark."""
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path


def prepare_tls():
    """Use Vast's documented instance certificate signing flow."""
    import requests
    if os.environ.get("USE_SSL") != "true":
        raise RuntimeError("Vast worker must use TLS")
    config = Path("/etc/luche-openssl.cnf")
    config.write_text("[req]\ndistinguished_name=dn\nreq_extensions=v3_req\n[dn]\n"
                      "[v3_req]\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\n"
                      "subjectAltName=@alt\n[alt]\nIP.1=0.0.0.0\n")
    subprocess.run(["openssl", "req", "-newkey", "rsa:2048", "-nodes", "-sha256",
                    "-subj", "/C=US/ST=CA/CN=pyworker.vast.ai/", "-keyout", "/etc/instance.key",
                    "-out", "/etc/instance.csr", "-config", str(config)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.chmod("/etc/instance.key", 0o600)
    for attempt in range(5):
        r = requests.post("https://console.vast.ai/api/v0/sign_cert/",
                          params={"instance_id": os.environ["CONTAINER_ID"]},
                          data=Path("/etc/instance.csr").read_bytes(),
                          headers={"Content-Type": "application/octet-stream"}, timeout=30)
        if r.ok:
            Path("/etc/instance.crt").write_bytes(r.content)
            return
        time.sleep(2 ** attempt)
    raise RuntimeError("Vast instance certificate signing failed")


def main():
    # Vast signs public requests. The model server is reachable only on loopback.
    os.environ.setdefault("WORKER_PORT", "3000")
    os.environ.setdefault("REPORT_ADDR", "https://run.vast.ai")
    os.environ.setdefault("USE_SSL", "true")
    if not os.environ.get("MASTER_TOKEN"):
        raise RuntimeError("Vast must supply MASTER_TOKEN for a serverless worker")
    prepare_tls()
    log_path = Path("/tmp/luche-vast-model.log")
    benchmark_token = secrets.token_urlsafe(32)
    env = {**os.environ, "PORT": "18000", "BENCHMARK_TOKEN": benchmark_token}
    with log_path.open("w") as log_file:
        proc = subprocess.Popen(["/usr/local/bin/luche-model-python", "-u", "-m", "adapters.vast_model_server"],
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
                log_action_config=LogActionConfig(on_load=["Uvicorn running on"],
                                                  on_error=["Traceback (most recent call last):"]),
            )).run()
        finally:
            proc.terminate()
            proc.wait(timeout=30)


if __name__ == "__main__":
    main()
