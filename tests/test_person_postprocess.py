import unittest
import numpy as np
from core.person_postprocess import BoxPolicy, deduplicate, postprocess_person_detections


class OfflineBoxesTests(unittest.TestCase):
    def process(self, frames, **kwargs):
        return postprocess_person_detections(list(enumerate(frames)),width=1000,height=800,fps=30,num_frames=len(frames),**kwargs)

    def test_outer_body_kept_instead_of_high_score_arm(self):
        outer=np.array([100,50,600,750]);arm=np.array([450,70,580,400])
        kept,removed=deduplicate([(arm,.99),(outer,.75)],frame=0,width=1000,height=800)
        self.assertEqual(len(kept),1);np.testing.assert_equal(kept[0].box,outer)
        self.assertEqual(removed[0]['reason'],'contained')

    def test_weak_oversized_box_does_not_erase_confident_person(self):
        kept,_=deduplicate([(np.array([1,1,998,798]),.16),(np.array([300,100,600,750]),.95)],frame=0,width=1000,height=800)
        self.assertEqual(len(kept),2)

    def test_overlapping_distinct_people_are_not_suppressed(self):
        kept,_=deduplicate([(np.array([100,50,400,700]),.9),(np.array([300,50,600,700]),.85)],frame=0,width=1000,height=800)
        self.assertEqual(len(kept),2)

    def test_short_interior_dropout_is_filled_using_two_sided_support(self):
        box=np.array([300,100,600,700]);frames=[[(box,.9)] for _ in range(15)]
        frames[6]=[];frames[7]=[];r=self.process(frames)
        self.assertEqual(r['status'][6],'interpolated');self.assertEqual(r['support_frames'][6],[5,8])
        self.assertTrue(np.isfinite(r['boxes']).all())

    def test_weak_detection_is_recovered_but_not_for_entire_empty_tail(self):
        box=np.array([300,100,600,700]);frames=[[(box,.9)] for _ in range(4)]+[[(box,.2)] for _ in range(20)]
        r=self.process(frames)
        self.assertEqual(r['status'][5],'weak_supported')
        self.assertTrue(np.isnan(r['boxes'][10:]).all())

    def test_exit_and_reentry_are_reacquired_without_bridge_through_empty_room(self):
        box=np.array([0,100,150,700]);frames=[[(box,.9)] for _ in range(8)]+[[] for _ in range(20)]+[[(box,.9)] for _ in range(8)]
        r=self.process(frames)
        self.assertTrue(np.isnan(r['boxes'][8:28]).all())
        self.assertTrue(np.isfinite(r['boxes'][28:]).all())
        self.assertEqual(len(r['reacquisitions']),1)

    def test_side_gap_limit_is_shorter_than_interior_gap(self):
        box=np.array([0,100,150,700]);frames=[[(box,.9)] for _ in range(6)]+[[] for _ in range(4)]+[[(box,.9)] for _ in range(6)]
        r=self.process(frames)
        self.assertTrue(np.isnan(r['boxes'][6:10]).all())

    def test_no_extrapolation_before_entry_or_after_exit(self):
        box=np.array([300,100,600,700]);frames=[[]]*5+[[(box,.9)] for _ in range(8)]+[[]]*8
        r=self.process(frames)
        self.assertTrue(np.isnan(r['boxes'][:5]).all());self.assertTrue(np.isnan(r['boxes'][13:]).all())

    def test_smoothing_contains_the_current_observed_body(self):
        frames=[[(np.array([300+(i%2)*5,100,600+(i%2)*5,700]),.9)] for i in range(15)]
        r=self.process(frames)
        for i,f in enumerate(frames):
            self.assertTrue(np.all(r['boxes'][i,:2]<=f[0][0][:2]+1e-4))
            self.assertTrue(np.all(r['boxes'][i,2:]>=f[0][0][2:]-1e-4))

    def test_persistent_edge_bystander_does_not_win_over_moving_patient(self):
        frames=[]
        for i in range(60):
            frames.append([(np.array([900,30,998,740]),.99),(np.array([300+i*2,100,550+i*2,700]),.85)])
        r=self.process(frames)
        self.assertTrue(np.all(r['boxes'][:,0]<500))

    def test_empty_video_has_no_fake_boxes(self):
        r=self.process([[] for _ in range(20)])
        self.assertTrue(np.isnan(r['boxes']).all())

    def test_nested_tracklets_with_shared_lifetime_do_not_drop_later_frames(self):
        body=np.array([0,0,700,790]);torso=np.array([0,0,700,550])
        frames=[[(body,.9)] for _ in range(15)]
        frames += [[(body,.2),(torso,.9)] for _ in range(5)]
        frames += [[(torso,.9)] for _ in range(20)]
        r=self.process(frames)
        self.assertTrue(np.isfinite(r['boxes']).all())

    def test_brief_foot_crop_collapse_uses_surrounding_body_extent(self):
        body=np.array([0,0,600,790]);foot=np.array([0,600,130,790])
        frames=[[(body,.9)] for _ in range(15)]+[[(foot,.8)] for _ in range(3)]+[[(body,.9)] for _ in range(15)]
        r=self.process(frames)
        self.assertTrue(r['shrink_guarded'][16])
        self.assertLess(r['boxes'][16,1],50)
        self.assertGreater(r['boxes'][16,2],550)

    def test_one_spurious_detection_does_not_create_a_crop(self):
        frames=[[] for _ in range(20)];frames[10]=[(np.array([100,100,500,700]),.95)]
        self.assertTrue(np.isnan(self.process(frames)['boxes']).all())

    def test_missing_frame_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            postprocess_person_detections([(1,[])],width=1000,height=800,fps=30,num_frames=1)


if __name__=='__main__':unittest.main()
