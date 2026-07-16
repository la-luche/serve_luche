import unittest

from core.request import parse_infer_input


class RequestParsingTests(unittest.TestCase):
    def test_valid_person_detection_options(self):
        video_url, kwargs = parse_infer_input(
            {
                "video_url": "https://example.test/video.mp4",
                "frame_stride": 2,
                "batch_size": 8,
                "person_detection": True,
                "person_detection_stride": 5,
                "person_box_overflow": 0.25,
                "person_detection_threshold": 0.3,
            }
        )
        self.assertEqual(video_url, "https://example.test/video.mp4")
        self.assertEqual(kwargs["batch_size"], 8)
        self.assertTrue(kwargs["person_detection"])

    def test_invalid_values_are_rejected_before_model_import(self):
        invalid_fields = {
            "frame_stride": [0, -1, 1.5, "1", True],
            "batch_size": [0, -1, 8.5, "8", False],
            "person_detection_stride": [0, -1, 2.5, "5", True],
            "person_box_overflow": [-0.1, 2.1, "0.25", True, float("nan")],
            "person_detection_threshold": [0.0, 1.1, "0.3", False],
            "person_detection": [0, 1, "true", None],
        }
        for field, values in invalid_fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        parse_infer_input(
                            {
                                "video_url": "https://example.test/video.mp4",
                                field: value,
                            }
                        )

    def test_missing_or_invalid_urls_are_rejected(self):
        for value in (None, "", "   ", 123):
            with self.subTest(video_url=value):
                with self.assertRaises(ValueError):
                    parse_infer_input({"video_url": value})


if __name__ == "__main__":
    unittest.main()
