import os
import unittest

import numpy as np

try:
    from core.preprocess import GpuPreprocessor, INPUT_H, INPUT_W
except ModuleNotFoundError as exc:
    if os.environ.get("REQUIRE_SAPIENS_TESTS") == "1":
        raise
    GpuPreprocessor = None
    INPUT_H = INPUT_W = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(GpuPreprocessor is None, f"Sapiens unavailable: {_IMPORT_ERROR}")
class PreprocessGeometryTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.pre = GpuPreprocessor(1280, 720, "cpu", dtype=torch.float32)

    def test_nan_box_is_exact_full_frame_fallback(self):
        full_matrix, full_meta = self.pre.geometry(1)
        fallback_matrix, fallback_meta = self.pre.geometry(
            1, np.full((1, 4), np.nan, dtype=np.float32), bbox_overflow=0.25
        )

        np.testing.assert_array_equal(
            fallback_matrix.numpy(), full_matrix.numpy()
        )
        np.testing.assert_array_equal(
            fallback_meta["bbox_center"], full_meta["bbox_center"]
        )
        np.testing.assert_array_equal(
            fallback_meta["bbox_scale"], full_meta["bbox_scale"]
        )

    def test_overflow_is_applied_on_each_side_then_aspect_corrected(self):
        _, meta = self.pre.geometry(
            1,
            np.asarray([[100, 100, 300, 500]], dtype=np.float32),
            bbox_overflow=0.25,
        )

        np.testing.assert_allclose(meta["bbox_center"][0], [200, 300])
        # Raw 200x400 box -> padding 1.5 = 300x600, then 3:4 => 450x600.
        np.testing.assert_allclose(meta["bbox_scale"][0], [450, 600])

    def test_input_to_original_back_map_round_trip(self):
        _, meta = self.pre.geometry(
            1,
            np.asarray([[100, 100, 300, 500]], dtype=np.float32),
            bbox_overflow=0.25,
        )
        original = np.asarray([[200.0, 300.0], [150.0, 450.0]])
        center = meta["bbox_center"][0]
        scale = meta["bbox_scale"][0]
        input_size = np.asarray([INPUT_W, INPUT_H], dtype=np.float32)

        encoded = (original - center + 0.5 * scale) / scale * input_size
        decoded = encoded / input_size * scale + center - 0.5 * scale

        np.testing.assert_allclose(decoded, original, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
