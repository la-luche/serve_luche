import unittest
import numpy as np
from core.box_smoothing import smooth_large_boxes


class LargeEnvelopeTests(unittest.TestCase):
    def smooth(self,b):
        return smooth_large_boxes(np.asarray(b),width=1000,height=800,fps=30)

    def test_flickering_nested_boxes_prefer_larger_extent(self):
        large=[200,100,800,700];small=[300,200,700,600]
        b=np.array([large if i%2 else small for i in range(90)])
        r=self.smooth(b)
        np.testing.assert_allclose(r[10:-10],np.tile(large,(70,1)),atol=.001)

    def test_true_shrink_is_eventually_followed(self):
        large=[200,100,800,700];small=[300,200,700,600]
        r=self.smooth([large]*30+[small]*60)
        self.assertGreater(r[31,2]-r[31,0],500)
        np.testing.assert_allclose(r[-1],small,atol=.001)

    def test_unsupported_gaps_are_not_smoothed_over(self):
        b=np.array([[200,100,800,700]]*10+[[np.nan]*4]*20+[[400,300,600,500]]*10)
        r=self.smooth(b)
        self.assertTrue(np.isnan(r[10:30]).all())
        np.testing.assert_allclose(r[30],[400,300,600,500])

    def test_smoothing_never_cuts_into_current_measurement(self):
        b=np.array([[200+i*3,100,500+i*3,700] for i in range(60)])
        r=self.smooth(b)
        self.assertTrue(np.all(r[:,:2]<=b[:,:2]+.001))
        self.assertTrue(np.all(r[:,2:]>=b[:,2:]-.001))

    def test_single_large_outlier_is_not_propagated_to_the_whole_window(self):
        b=np.array([[300,200,700,600]]*61);b[30]=[50,50,950,750]
        r=self.smooth(b)
        np.testing.assert_allclose(r[29],[300,200,700,600],atol=.001)
        np.testing.assert_allclose(r[30],b[30],atol=.001)


if __name__=='__main__':unittest.main()
