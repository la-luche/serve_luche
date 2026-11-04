"""Offline outer-envelope smoothing; no pose/detector dependencies."""
import numpy as np


def smooth_large_boxes(boxes, *, width, height, fps, envelope_seconds=.30,
                       smooth_seconds=.12, quantile=.90):
    """Prefer repeated larger extents and smooth their path, within valid runs.

    Coordinate quantiles reject isolated extrema. A symmetric filter avoids
    causal lag. The final envelope still contains each current measured box.
    NaN gaps are retained exactly and never contribute to either filter.
    """
    raw=np.asarray(boxes,dtype=np.float64)
    result=raw.copy();valid=np.isfinite(raw).all(axis=1)
    radius=max(1,round(envelope_seconds*fps));blur=max(1,round(smooth_seconds*fps))
    starts=np.flatnonzero(valid & ~np.r_[False,valid[:-1]])
    ends=np.flatnonzero(valid & ~np.r_[valid[1:],False])+1
    for start,end in zip(starts,ends):
        envelope=raw[start:end].copy()
        for j,i in enumerate(range(start,end)):
            context=raw[max(start,i-radius):min(end,i+radius+1)]
            envelope[j,:2]=np.quantile(context[:,:2],1-quantile,axis=0)
            envelope[j,2:]=np.quantile(context[:,2:],quantile,axis=0)
        for j,i in enumerate(range(start,end)):
            lo,hi=max(0,j-blur),min(end-start,j+blur+1)
            weights=np.exp(-.5*((np.arange(lo,hi)-j)/(max(blur/2,1)))**2)
            filtered=np.average(envelope[lo:hi],axis=0,weights=weights)
            result[i]=np.r_[np.minimum(filtered[:2],raw[i,:2]),np.maximum(filtered[2:],raw[i,2:])]
            # Make an edge sticky while it has support inside the envelope.
            for axis,bound in [(0,width-1),(1,height-1)]:
                if result[i,axis]<.035*bound and envelope[j,axis]<.01*bound:result[i,axis]=0
                if result[i,axis+2]>.965*bound and envelope[j,axis+2]>.99*bound:result[i,axis+2]=bound
        result[start:end]=np.clip(result[start:end],[0,0,0,0],[width-1,height-1,width-1,height-1])
    return result.astype(np.float32)

