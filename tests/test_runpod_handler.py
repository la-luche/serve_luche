import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

try:
    import runpod  # noqa: F401
except ModuleNotFoundError:
    fake_runpod = ModuleType("runpod")
    fake_runpod.serverless = SimpleNamespace(start=Mock())
    sys.modules["runpod"] = fake_runpod

from adapters import runpod_handler


class RunPodStartupTests(unittest.TestCase):
    def setUp(self):
        runpod_handler._RUN_VIDEO = None

    def tearDown(self):
        runpod_handler._RUN_VIDEO = None

    def test_prepare_worker_preloads_and_warms_by_default(self):
        with patch.object(runpod_handler, "_load_runtime") as load_runtime:
            with patch.dict(runpod_handler.os.environ, {}, clear=True):
                runpod_handler.prepare_worker()

        load_runtime.assert_called_once_with(warm=True)

    def test_prepare_worker_can_skip_preload_for_lightweight_smoke(self):
        with patch.object(runpod_handler, "_load_runtime") as load_runtime:
            with patch.dict(
                runpod_handler.os.environ,
                {"PRELOAD_MODEL": "0"},
                clear=True,
            ):
                runpod_handler.prepare_worker()

        load_runtime.assert_not_called()

    def test_prepare_worker_can_preload_without_first_forward(self):
        with patch.object(runpod_handler, "_load_runtime") as load_runtime:
            with patch.dict(
                runpod_handler.os.environ,
                {"PRELOAD_MODEL": "1", "WARMUP_ON_START": "false"},
                clear=True,
            ):
                runpod_handler.prepare_worker()

        load_runtime.assert_called_once_with(warm=False)

    def test_ping_never_loads_runtime(self):
        with patch.object(runpod_handler, "_load_runtime") as load_runtime:
            result = runpod_handler.handler({"input": {"ping": True}})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["batch_size"], 1)
        self.assertFalse(result["compile"])
        self.assertTrue(result["preload_model"])
        self.assertTrue(result["warmup_on_start"])
        load_runtime.assert_not_called()

    def test_valid_job_uses_preloaded_runtime(self):
        fake_run_video = Mock(return_value={"status": "ok"})
        with patch.object(runpod_handler.auth, "check", return_value=True):
            with patch.object(
                runpod_handler,
                "_load_runtime",
                return_value=fake_run_video,
            ):
                result = runpod_handler.handler(
                    {
                        "input": {
                            "api_key": "test",
                            "video_url": "https://example.test/video.mp4",
                        }
                    }
                )

        self.assertEqual(result["status"], "ok")
        fake_run_video.assert_called_once()


if __name__ == "__main__":
    unittest.main()
