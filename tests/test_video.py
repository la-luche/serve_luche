"""Exercise real H.264 B-frame decoding and the detector-to-pose rewind."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import torch

from core.video import Decoder


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required for decoder integration")
class VideoDecoderTests(unittest.TestCase):
    def test_detector_and_pose_passes_preserve_every_frame(self):
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root) / "bframes.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=96x64:rate=24", "-frames:v", "49", "-c:v",
                "libx264", "-bf", "3", "-g", "24", "-pix_fmt", "yuv420p",
                str(path),
            ], check=True, timeout=30)
            decoder = Decoder(str(path))
            self.assertEqual(decoder.num_frames, 49)
            self.assertEqual((decoder.width, decoder.height), (96, 64))
            detector = list(decoder.iter_batches(1, 8, "cpu"))
            pose = list(decoder.iter_batches(1, 1, "cpu"))
            self.assertEqual([i for ids, _ in detector for i in ids], list(range(49)))
            self.assertEqual([i for ids, _ in pose for i in ids], list(range(49)))
            detector_pixels = torch.cat([p for _, p in detector])
            pose_pixels = torch.cat([p for _, p in pose])
            self.assertTrue(torch.equal(detector_pixels, pose_pixels))
            self.assertEqual(pose_pixels.dtype, torch.uint8)
            self.assertFalse(torch.equal(pose_pixels[0], pose_pixels[-1]))
            indices = [48, 0, 24, 0]
            self.assertTrue(torch.equal(decoder.get_batch(indices, "cpu"), pose_pixels[indices]))


if __name__ == "__main__":
    unittest.main()
