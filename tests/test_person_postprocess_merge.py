"""Exact output checks against the original reviewed box-merging algorithm."""
import unittest
from unittest.mock import patch
import numpy as np
import core.person_postprocess as pp
from reference_person_merge import reference_merge_duplicate_tracklets


class IndexedMergeTests(unittest.TestCase):
    @staticmethod
    def signature(tracks):
        return [(t.id, [(d.frame, tuple(d.box), d.score, d.sources)
                        for d in t.observations]) for t in tracks]

    def compare(self, tracks, policy=pp.BoxPolicy()):
        self.assertEqual(self.signature(pp._merge_duplicate_tracklets(tracks, policy)),
                         self.signature(reference_merge_duplicate_tracklets(tracks, policy)))

    def test_empty_and_disjoint_lifetimes(self):
        self.compare([])
        self.compare([pp.Tracklet(7, [])])
        self.compare([pp.Tracklet(i, [pp.Detection(i*20+f,
            np.array([10., 10., 90., 190.]), .8, (i,)) for f in range(10)])
            for i in range(150)])

    def test_merge_ties_follow_input_order_and_preserve_sources(self):
        for order in ([9, 2, 7, 1], [1, 7, 2, 9]):
            self.compare([pp.Tracklet(i, [pp.Detection(f,
                np.array([10., 10., 90., 190.]), .8, (i,))
                for f in range(10)]) for i in order])

    def test_random_fragmented_tracks_match_reviewed_reference(self):
        for seed in range(100):
            rng = np.random.default_rng(seed)
            tracks = []
            for slot in range(18):
                first = int(rng.integers(0, 20))
                last = int(rng.integers(first+1, 40))
                identity = int(rng.integers(0, 3))
                dx = float(rng.choice([0, 0, 2, 10, 60]))
                box = np.array([identity*130.+dx, 10., identity*130.+100., 190.])
                observations = [pp.Detection(f, box.copy(),
                    float(rng.choice([.15, .2, .3, .8, .99])), (slot,))
                    for f in range(first, last) if rng.random() > .1]
                tracks.append(pp.Tracklet(100-slot, observations))
            with self.subTest(seed=seed):
                self.compare(tracks)

    def test_full_postprocess_boxes_and_audit_match_reference(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            samples = []
            for f in range(80):
                candidates = []
                for person in range(4):
                    if rng.random() < .25:
                        continue
                    x = 40.+person*190.+f*.5
                    box = np.array([x, 20., x+170., 760.])
                    candidates.append((box, float(rng.choice([.2, .9]))))
                    if rng.random() < .5:
                        candidates.append((box+[10., 40., -10., -140.], .85))
                samples.append((f, candidates))
            kwargs = dict(width=1000, height=800, fps=30, num_frames=len(samples))
            actual = pp.postprocess_person_detections(samples, **kwargs)
            with patch.object(pp, '_merge_duplicate_tracklets', reference_merge_duplicate_tracklets):
                expected = pp.postprocess_person_detections(samples, **kwargs)
            with self.subTest(seed=seed):
                np.testing.assert_equal(actual.pop('boxes'), expected.pop('boxes'))
                np.testing.assert_equal(actual.pop('unsmoothed_boxes'), expected.pop('unsmoothed_boxes'))
                for left, right in zip(actual.pop('deduplicated'), expected.pop('deduplicated')):
                    self.assertEqual(self.signature([pp.Tracklet(0, left)]),
                                     self.signature([pp.Tracklet(0, right)]))
                self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
