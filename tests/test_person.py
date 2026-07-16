import unittest

import numpy as np

from core.person import interpolate_track, select_person_track


class PersonTrackingTests(unittest.TestCase):
    def test_prefers_persistent_center_patient_over_edge_bystander(self):
        samples = []
        for sample in range(10):
            frame_idx = sample * 5
            patient = (
                np.asarray([260 + sample, 150, 460 + sample, 650], dtype=np.float32),
                0.82,
            )
            candidates = [patient]
            if sample < 6:
                candidates.append(
                    (np.asarray([610, 100, 719, 700], dtype=np.float32), 0.99)
                )
            samples.append((frame_idx, candidates))

        track = select_person_track(
            samples, width=720, height=1280, detection_stride=5
        )

        self.assertIsNotNone(track)
        self.assertEqual(len(track.observations), 10)
        self.assertLess(track.observations[0].box[0], 300)

    def test_interpolation_keeps_endpoints_and_positive_box_size(self):
        samples = [
            (0, [(np.asarray([10, 20, 110, 220], dtype=np.float32), 0.9)]),
            (10, [(np.asarray([30, 40, 230, 440], dtype=np.float32), 0.9)]),
        ]
        track = select_person_track(
            samples, width=640, height=480, detection_stride=10
        )

        boxes = interpolate_track(track, num_frames=11)

        np.testing.assert_allclose(boxes[0], samples[0][1][0][0], atol=2e-5)
        np.testing.assert_allclose(boxes[-1], samples[-1][1][0][0], atol=2e-5)
        self.assertTrue(np.all(boxes[:, 2] > boxes[:, 0]))
        self.assertTrue(np.all(boxes[:, 3] > boxes[:, 1]))

    def test_no_detections_returns_no_track(self):
        samples = [(0, []), (5, []), (10, [])]
        track = select_person_track(
            samples, width=720, height=1280, detection_stride=5
        )
        self.assertIsNone(track)


if __name__ == "__main__":
    unittest.main()
