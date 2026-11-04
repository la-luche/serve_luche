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
        import torch
        legacy = GpuPreprocessor(1280, 720, "cpu", torch.float32, crop_policy="legacy")
        _, meta = legacy.geometry(
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

    def test_final_crop_clips_after_padding_and_aspect_expansion(self):
        _, meta = self.pre.geometry(
            1, np.asarray([[100, 100, 300, 500]], dtype=np.float32), 0.25
        )
        center, scale = meta["bbox_center"][0], meta["bbox_scale"][0]
        # Desired legacy crop is [-25, 0, 425, 600]; retain only real pixels.
        np.testing.assert_allclose(center - scale / 2, [0, 0])
        np.testing.assert_allclose(center + scale / 2, [425, 600])

    def test_full_frame_has_no_expansion(self):
        _, meta = self.pre.geometry(1)
        np.testing.assert_array_equal(meta["bbox_center"][0], [359.5, 639.5])
        np.testing.assert_array_equal(meta["bbox_scale"][0], [719, 1279])

    def test_random_crops_stay_inside_and_retain_visible_box(self):
        rng = np.random.default_rng(17)
        boxes = rng.uniform([-200, -200, 720, 1280], [500, 1000, 1000, 1600], (100, 4))
        for padding in [0, 0.10, 0.25, 2.0]:
            _, meta = self.pre.geometry(len(boxes), boxes, padding)
            lower = meta["bbox_center"] - meta["bbox_scale"] / 2
            upper = meta["bbox_center"] + meta["bbox_scale"] / 2
            self.assertTrue((lower >= -1e-4).all())
            self.assertTrue((upper <= np.asarray([719, 1279]) + 1e-4).all())
            visible = np.clip(boxes.reshape(-1, 2, 2), 0, [719, 1279])
            self.assertTrue((lower <= visible[:, 0] + 2e-4).all())
            self.assertTrue((upper >= visible[:, 1] - 2e-4).all())

    def test_outside_only_box_falls_back_to_real_full_frame(self):
        for box in [[800, 10, 900, 100], [-20, -20, -5, -5], [0, 0, 0, 10]]:
            matrix, meta = self.pre.geometry(1, np.asarray([box], dtype=np.float32))
            np.testing.assert_array_equal(matrix.numpy()[0], self.pre.full_matrix)
            np.testing.assert_array_equal(meta["bbox_scale"][0], [719, 1279])

    def test_real_warp_contains_no_black_and_maps_pixel_ramp(self):
        import torch
        from core.preprocess import MEAN, STD

        pre = GpuPreprocessor(80, 40, "cpu", torch.float32)
        yy, xx = torch.meshgrid(torch.arange(80), torch.arange(40), indexing="ij")
        frame = torch.stack([xx + 30, yy + 30, torch.full_like(xx, 200)]).float()[None]
        boxes = np.asarray([[0, 0, 39, 79], [0, 10, 19, 60], [25, 55, 39, 79]], dtype=np.float32)
        output, meta = pre(frame.expand(3, -1, -1, -1), boxes, 0.25)
        pixels = output * torch.tensor(STD)[None, :, None, None] + torch.tensor(MEAN)[None, :, None, None]
        self.assertTrue(torch.allclose(pixels[:, 2], torch.full_like(pixels[:, 2], 200), atol=1e-3))
        for i in range(3):
            lower = meta["bbox_center"][i] - meta["bbox_scale"][i] / 2
            upper = meta["bbox_center"][i] + meta["bbox_scale"][i] / 2
            np.testing.assert_allclose(pixels[i, :2, 0, 0].numpy(), lower + 30, atol=.002)
            np.testing.assert_allclose(pixels[i, :2, -1, -1].numpy(), upper + 30, atol=.002)
            np.testing.assert_allclose(pixels[i, 0, INPUT_H // 2].numpy(), np.linspace(lower[0], upper[0], INPUT_W) + 30, atol=.002)
            np.testing.assert_allclose(pixels[i, 1, :, INPUT_W // 2].numpy(), np.linspace(lower[1], upper[1], INPUT_H) + 30, atol=.002)


if __name__ == "__main__":
    unittest.main()
