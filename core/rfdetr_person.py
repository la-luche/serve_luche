"""RF-DETR pass -> offline person-box policy -> pose-ready per-frame boxes."""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
import time
from dataclasses import replace
import numpy as np
import torch
from PIL import Image
from .person_postprocess import BoxPolicy, postprocess_person_detections
from .log import log

_models = {}
_revisions = {}


def _get_detector(device):
    checkpoint=os.environ.get('RFDETR_WEIGHTS','')
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError('RFDETR_WEIGHTS must point to the local rf-detr-large-2026.pth checkpoint')
    key=(str(device),str(Path(checkpoint).resolve()))
    if key not in _models:
        from rfdetr import RFDETRLarge
        detector=RFDETRLarge(device=device,pretrain_weights=checkpoint)
        # Export the inference graph once; keep FP32 to preserve the reviewed
        # detections. Half precision changed threshold decisions in the bench.
        detector.inference(compile=False, dtype=torch.float32, inplace=True)
        _models[key]=detector
        with open(checkpoint,'rb') as f:
            _revisions[key]='sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
    return _models[key], _revisions[key]


def warmup_rfdetr(device="cuda"):
    detector, _ = _get_detector(device)
    batch_size = int(os.environ.get('RFDETR_BATCH_SIZE', '8'))
    if batch_size < 1:
        raise ValueError('RFDETR_BATCH_SIZE must be >= 1')
    detector.predict([Image.new('RGB', (640, 640))] * batch_size,
                     threshold=.05, include_source_image=False)


def detect_person_track_rfdetr(decoder, *, device, threshold=.30, cancel_hook=None):
    batch_size = int(os.environ.get('RFDETR_BATCH_SIZE', '8'))
    if batch_size < 1:
        raise ValueError('RFDETR_BATCH_SIZE must be >= 1')
    detector, revision = _get_detector(device)
    samples=[];detection_started=time.perf_counter();last_progress=detection_started
    # Complete the video pass first. The policy uses future evidence too.
    for start in range(0, decoder.num_frames, batch_size):
        if cancel_hook:cancel_hook()
        indices=list(range(start,min(start+batch_size,decoder.num_frames)))
        frames=decoder.get_batch(indices,device='cpu')
        images=[Image.fromarray(frame.permute(1,2,0).numpy()) for frame in frames]
        predictions=detector.predict(images,threshold=min(.05,threshold),include_source_image=False)
        if len(predictions) != len(indices):
            raise RuntimeError('RF-DETR returned a different number of predictions than frames')
        for idx,prediction in zip(indices,predictions):
            candidates=[(np.asarray(b,dtype=np.float32),float(s)) for b,s,label in zip(prediction.xyxy,prediction.confidence,prediction.class_id) if int(label)==1]
            samples.append((idx,candidates))
        now=time.perf_counter()
        if now-last_progress>=30 or len(samples)==decoder.num_frames:
            log(f"RF-DETR {len(samples)}/{decoder.num_frames} frames in {now-detection_started:.1f}s")
            last_progress=now
    detection_seconds=time.perf_counter()-detection_started
    if cancel_hook:cancel_hook()
    policy=replace(BoxPolicy(),high_score=threshold,low_score=min(.15,threshold))
    start=time.perf_counter()
    log(f"person box postprocessing started: {len(samples)} frames")
    result=postprocess_person_detections(samples,width=decoder.width,height=decoder.height,
        fps=decoder.fps,num_frames=decoder.num_frames,policy=policy)
    postprocess_seconds=time.perf_counter()-start
    log(f"person box postprocessing completed in {postprocess_seconds:.2f}s")
    fallback=sum(s=='full_frame_fallback' for s in result['status'])
    return result['boxes'],{'backend':'rfdetr','model':'RF-DETR Large / rf-detr-large-2026.pth',
        'revision':revision,'batch_size':batch_size,'precision':'float32',
        'stride':1,'detection_frames':decoder.num_frames,'detection_seconds':detection_seconds,
        'postprocess_seconds':postprocess_seconds,'postprocess_policy':result['policy'],
        'postprocess_implementation':'indexed_tracklet_merge_v1',
        'suppressed_boxes':sum(len(x) for x in result['suppressed']),
        'interpolated_frames':result['status'].count('interpolated'),
        'weak_supported_frames':result['status'].count('weak_supported'),
        'shrink_guarded_frames':sum(result['shrink_guarded']),
        'full_frame_fallback_frames':fallback,'fallback_full_frame':fallback==decoder.num_frames,
        'reacquisitions':result['reacquisitions']}
