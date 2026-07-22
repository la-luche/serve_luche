import unittest
import threading
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from adapters.http_worker import InferRequest, JobManager, app
from core.errors import InferenceCancelled


class HttpWorkerTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_does_not_load_model(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertIn("git_sha", response.json())
        self.assertFalse(response.json()["busy"])

    def test_runpod_ping_shape(self):
        response = self.client.post("/run", json={"input": {"ping": True}})
        self.assertEqual(response.status_code, 200)
        self.assertIn("git_sha", response.json())

    def test_runpod_wrapped_request_rejects_bad_key_before_model_load(self):
        response = self.client.post(
            "/run",
            json={"input": {"api_key": "wrong", "video_url": "https://example/video"}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "unauthorized"})

    def test_flat_request_rejects_bad_key_before_model_load(self):
        response = self.client.post(
            "/infer",
            json={"api_key": "wrong", "video_url": "https://example/video"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "unauthorized"})

    def test_job_manager_exposes_runpod_compatible_completed_state(self):
        def execute(req, cancelled):
            self.assertFalse(cancelled.is_set())
            return {"status": "ok", "video_url": req.video_url}

        with TemporaryDirectory() as state_dir:
            manager = JobManager(state_dir, executor=execute)
            job_id = manager.submit(
                InferRequest(api_key="test", video_url="https://example/video")
            )
            state = manager.wait(job_id)
            manager.close()

        self.assertEqual(state["status"], "COMPLETED")
        self.assertEqual(state["output"]["status"], "ok")
        self.assertEqual(state["output"]["video_url"], "https://example/video")

    def test_running_job_cancels_cooperatively(self):
        started = threading.Event()

        def execute(req, cancelled):
            started.set()
            cancelled.wait(timeout=2)
            raise InferenceCancelled("cancelled in test")

        with TemporaryDirectory() as state_dir:
            manager = JobManager(state_dir, executor=execute)
            job_id = manager.submit(
                InferRequest(api_key="test", video_url="https://example/video")
            )
            self.assertTrue(started.wait(timeout=2))
            manager.cancel(job_id)
            state = manager.wait(job_id)
            manager.close()

        self.assertEqual(state["status"], "CANCELLED")


if __name__ == "__main__":
    unittest.main()
