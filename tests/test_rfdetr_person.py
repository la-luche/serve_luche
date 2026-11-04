import os
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from core import rfdetr_person as rf
from core.person import detect_person_track


class RFBackendTests(unittest.TestCase):
    def setUp(self):
        rf._models.clear()
        self.checkpoint=tempfile.NamedTemporaryFile(suffix='.pth')
        self.env=patch.dict(os.environ,{'PERSON_DETECTOR_BACKEND':'rfdetr','RFDETR_WEIGHTS':self.checkpoint.name,'RFDETR_BATCH_SIZE':'8'})
        self.env.start()
        self.calls=[]
        owner=self
        class Detector:
            def __init__(self,**kwargs):pass
            def inference(self,**kwargs):pass
            def predict(self,images,threshold,include_source_image):
                owner.calls.append(len(images))
                owner.assertFalse(include_source_image)
                return [types.SimpleNamespace(xyxy=np.array([[0,0,99,79],[20,10,60,70]]),confidence=np.array([.99,.9]),class_id=np.array([0,1])) for image in images]
        self.model=patch.dict('sys.modules',{'rfdetr':types.SimpleNamespace(RFDETRLarge=Detector)})
        self.model.start()
        self.indices=[]
        def batch(indices,device):
            self.indices.extend(indices)
            return torch.zeros((len(indices),3,80,100),dtype=torch.uint8)
        self.decoder=types.SimpleNamespace(num_frames=19,width=100,height=80,fps=30,get_batch=batch)

    def tearDown(self):
        self.model.stop();self.env.stop();self.checkpoint.close();rf._models.clear()

    def test_dispatch_detects_every_frame_before_postprocessing_and_filters_class_one(self):
        actual=rf.postprocess_person_detections
        def post(samples,**kwargs):
            self.assertEqual(self.calls,[8,8,3])
            self.assertEqual([i for i,_ in samples],list(range(19)))
            self.assertTrue(all(len(c)==1 for _,c in samples))
            return actual(samples,**kwargs)
        with patch.object(rf,'postprocess_person_detections',side_effect=post), \
             patch.object(rf.time,'perf_counter',side_effect=[1000.,1002.,1002.,1002.25]):
            boxes,stats=detect_person_track(self.decoder,device='cpu',detection_stride=5,threshold=.3)
        self.assertEqual(self.indices,list(range(19)))
        np.testing.assert_allclose(boxes,np.tile([20,10,60,70],(19,1)))
        self.assertEqual(stats['stride'],1)
        self.assertEqual(stats['full_frame_fallback_frames'],0)
        self.assertTrue(stats['revision'].startswith('sha256:'))
        self.assertEqual(stats['batch_size'],8)
        self.assertEqual(stats['precision'],'float32')
        self.assertEqual(stats['detection_seconds'],2.)
        self.assertEqual(stats['postprocess_seconds'],.25)

    def test_cancellation_during_detection_does_not_start_postprocessing(self):
        def cancel():
            if len(self.calls)==2:raise RuntimeError('cancelled')
        with patch.object(rf,'postprocess_person_detections') as post:
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                rf.detect_person_track_rfdetr(self.decoder,device='cpu',cancel_hook=cancel)
            post.assert_not_called()
        self.assertEqual(len(self.calls),2)

    def test_missing_local_weights_fails_without_fetching(self):
        with patch.dict(os.environ,{'RFDETR_WEIGHTS':''}):
            with self.assertRaisesRegex(RuntimeError,'RFDETR_WEIGHTS'):
                rf.detect_person_track_rfdetr(self.decoder,device='cpu')
        self.assertEqual(self.calls,[])

    def test_unknown_backend_is_explicit_error(self):
        with patch.dict(os.environ,{'PERSON_DETECTOR_BACKEND':'typo'}):
            with self.assertRaisesRegex(ValueError,'unknown PERSON_DETECTOR_BACKEND'):
                detect_person_track(self.decoder,device='cpu',detection_stride=1,threshold=.3)


if __name__=='__main__':unittest.main()
