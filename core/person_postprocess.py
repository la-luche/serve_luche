"""Offline person boxes for a top-down pose model. No detector or GPU imports.

Consume the complete, frame-indexed detector pass before calling this module.
NaN rows mean full-frame pose fallback, never 'skip pose'. This is geometric
association, not identity recognition: ambiguous reacquisition is left separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
import numpy as np


@dataclass(frozen=True)
class BoxPolicy:
    high_score: float = .30
    low_score: float = .15
    containment: float = .95
    duplicate_iou: float = .90
    gap_seconds: float = .25
    side_gap_seconds: float = .10
    weak_support_seconds: float = .20
    reacquire_seconds: float = 2.0
    smoothing_seconds: float = .10  # radius; symmetric, no causal lag
    smoothing_method: str = 'large_envelope'
    envelope_seconds: float = .30
    envelope_smooth_seconds: float = .12
    envelope_quantile: float = .90
    shrink_guard_seconds: float = .25  # radius for brief body-part box collapses
    side_margin: float = .035
    min_strong_observations: int = 2


@dataclass
class Detection:
    frame: int
    box: np.ndarray
    score: float
    sources: tuple[int, ...]


@dataclass
class Tracklet:
    id: int
    observations: list[Detection] = field(default_factory=list)


def area(b):
    return max(0., float(b[2]-b[0])) * max(0., float(b[3]-b[1]))


def intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    inter = intersection(a, b)
    return float(inter / max(area(a)+area(b)-inter, 1.))


def center(b):
    return (b[:2]+b[2:])*.5


def side_flags(b, width, margin):
    return (bool(b[0] <= width*margin), bool(b[2] >= width*(1-margin)))


def deduplicate(candidates, *, frame, width, height, policy=BoxPolicy()):
    """Prefer a supported outer box; overlap alone does not imply containment.

    A low-confidence, oversized box cannot suppress a confident inner box.
    Record every suppressed raw index so the decision is inspectable.
    """
    clean = []
    for i, (box, score) in enumerate(candidates):
        b = np.asarray(box, dtype=np.float64)
        if b.shape != (4,) or not np.isfinite(b).all() or not math.isfinite(score):
            continue
        if not policy.low_score <= score <= 1:
            continue
        b = np.clip(b, [0, 0, 0, 0], [width-1, height-1, width-1, height-1])
        if b[2]-b[0] >= 2 and b[3]-b[1] >= 2:
            clean.append(Detection(frame, b, float(score), (i,)))
    clean.sort(key=lambda d: (-area(d.box), -d.score, d.sources))
    kept, removed = [], []
    for d in clean:
        parent = None
        reason = None
        for outer in kept:
            covered = intersection(outer.box, d.box) / max(area(d.box), 1.)
            supported = outer.score >= policy.high_score or outer.score >= .8*d.score
            if covered >= policy.containment and supported:
                parent, reason = outer, 'contained'
                break
            if iou(outer.box, d.box) >= policy.duplicate_iou and supported:
                parent, reason = outer, 'near_duplicate'
                break
        if parent is None:
            kept.append(d)
        else:
            removed.append({'raw_index': d.sources[0], 'kept_raw_index': parent.sources[0], 'reason': reason})
    return kept, removed


def _match(a, b, width, policy, prediction=None):
    predicted = a if prediction is None else prediction
    sizes = np.maximum((a[2:]-a[:2]+b[2:]-b[:2])*.5, 2.)
    distance = float(np.linalg.norm((center(predicted)-center(b))/sizes))
    scale = float(np.max(np.abs(np.log(np.maximum(b[2:]-b[:2],2)/np.maximum(a[2:]-a[:2],2)))))
    overlap = max(iou(a,b), iou(predicted,b))
    same_side = any(x and y for x,y in zip(side_flags(a,width,policy.side_margin),side_flags(b,width,policy.side_margin)))
    if scale > (2.1 if same_side else 1.2):
        return None
    if not (overlap >= .12 or (distance < .5 and scale < .75) or (same_side and distance < 1.2 and overlap > .025)):
        return None
    return 3*overlap-distance-.25*scale


def _predict(track, frame):
    last = track.observations[-1]
    if len(track.observations)<2:
        return last.box
    recent = track.observations[-4:]
    velocities = [(center(b.box)-center(a.box))/(b.frame-a.frame) for a,b in zip(recent,recent[1:])]
    velocity = np.median(velocities, axis=0)
    shift = velocity*(frame-last.frame)
    limit = np.maximum(last.box[2:]-last.box[:2],2)*.5
    shift = np.clip(shift,-limit,limit)
    return last.box + np.tile(shift,2)


def _tracklets(per_frame, fps, width, policy):
    tracks = []
    max_gap = max(1, math.floor(policy.gap_seconds*fps)) + 1
    for frame, detections in enumerate(per_frame):
        pairs = []
        for t in tracks:
            if frame-t.observations[-1].frame > max_gap:
                continue
            for j,d in enumerate(detections):
                score = _match(t.observations[-1].box, d.box, width, policy, _predict(t,frame))
                if score is not None:
                    pairs.append((score+.10*d.score,t.id,j))
        used_tracks,used_detections = set(),set()
        for _,tid,j in sorted(pairs,reverse=True):
            if tid not in used_tracks and j not in used_detections:
                tracks[tid].observations.append(detections[j]);used_tracks.add(tid);used_detections.add(j)
        for j,d in enumerate(detections):
            if j not in used_detections:
                tracks.append(Tracklet(len(tracks),[d]))
    supported = []
    max_weak = max(1, math.floor(policy.weak_support_seconds*fps))
    for t in tracks:
        strong = [d.frame for d in t.observations if d.score >= policy.high_score]
        if len(strong) < policy.min_strong_observations:
            continue
        t.observations = [d for d in t.observations if d.score>=policy.high_score or min(abs(d.frame-f) for f in strong)<=max_weak]
        supported.append(t)
    return supported


def _merge_duplicate_tracklets(tracks, policy):
    """Join fragmented representations of one body, not adjacent people.

    A detector can alternate between nested boxes, briefly producing both.
    Requiring non-overlapping lifetimes would lose the entire later segment.
    All shared observations must mostly agree spatially before merging.
    """
    tracks = list(tracks)
    while True:
        best = None
        for i, a in enumerate(tracks):
            da = {d.frame: d for d in a.observations}
            for j in range(i+1, len(tracks)):
                b = tracks[j];db = {d.frame: d for d in b.observations}
                shared = sorted(da.keys() & db.keys())
                if not shared:continue
                overlaps = np.array([iou(da[f].box,db[f].box) for f in shared])
                # A single common boundary frame needs especially close agreement.
                threshold = .60 if len(shared)==1 else .50
                agrees=[];strong_disagreement=False
                for f,overlap in zip(shared,overlaps):
                    x,y=da[f],db[f]
                    contained=intersection(x.box,y.box)/max(min(area(x.box),area(y.box)),1.)
                    size=np.maximum((x.box[2:]-x.box[:2]+y.box[2:]-y.box[:2])*.5,2)
                    distance=np.linalg.norm((center(x.box)-center(y.box))/size)
                    agree=overlap>=threshold or (len(shared)>1 and contained>=.90 and distance<.65)
                    agrees.append(agree)
                    if not agree and min(x.score,y.score)>=policy.high_score:strong_disagreement=True
                if strong_disagreement or np.mean(agrees)<.75:continue
                quality = float(np.median(overlaps))
                if best is None or quality > best[0]:best=(quality,i,j)
        if best is None:return tracks
        _,i,j=best;a,b=tracks[i],tracks[j]
        merged={d.frame:d for d in a.observations}
        for d in b.observations:
            old=merged.get(d.frame)
            if old is None or (area(d.box),d.score)>(area(old.box),old.score):merged[d.frame]=d
        tracks[i]=Tracklet(min(a.id,b.id),[merged[f] for f in sorted(merged)])
        del tracks[j]


def _importance(track, width, height, n):
    obs=track.observations;b=np.stack([d.box for d in obs]);centers=(b[:,:2]+b[:,2:])*.5
    central=1-np.clip(np.linalg.norm((centers-[width*.5,height*.5])/[width*.5,height*.5],axis=1),0,1)
    areas=np.array([area(x) for x in b])/(width*height)
    motion=np.linalg.norm(np.diff(centers,axis=0),axis=1).sum()/math.hypot(width,height)
    return 1.5*len(obs)/n + np.mean([d.score for d in obs]) + 2*np.sqrt(areas.mean()) + 3*central.mean() + min(5*motion,1.)


def _select_chain(tracks,width,height,n,fps,policy):
    if not tracks:
        return [],[]
    main=max(tracks,key=lambda t:(_importance(t,width,height,n),-t.id))
    selected=[main];links=[]
    # Join only non-overlapping tracklets at the ends. Never switch to a
    # concurrently visible bystander just because the patient is briefly lost.
    for direction in [-1,1]:
        while True:
            edge=selected[0] if direction<0 else selected[-1]
            candidates=[]
            for t in tracks:
                if any(t.id==x.id for x in selected):continue
                left,right=(t,edge) if direction<0 else (edge,t)
                a,b=left.observations[-1],right.observations[0]
                gap=(b.frame-a.frame-1)/fps
                if gap<0 or gap>policy.reacquire_seconds:continue
                score=_match(a.box,b.box,width,policy)
                if score is None:continue
                candidates.append((score-.2*gap,t.id,t,gap))
            candidates.sort(key=lambda x:(-x[0],x[1]))
            if not candidates or (len(candidates)>1 and candidates[0][0]-candidates[1][0]<.20):break
            _,_,t,gap=candidates[0]
            links.append({'from_track':t.id if direction<0 else edge.id,'to_track':edge.id if direction<0 else t.id,'gap_seconds':gap})
            if direction<0:selected.insert(0,t)
            else:selected.append(t)
    return selected,links


def postprocess_person_detections(samples, *, width, height, fps, num_frames, policy=BoxPolicy()):
    """Return Nx4 pose boxes plus per-frame audit information.

    Exact contiguous frame coverage is required. Missing boxes stay NaN outside
    bounded, two-sided interpolation. Reacquisition does not fill an exit gap.
    """
    if width<3 or height<3 or not math.isfinite(fps) or fps<=0 or num_frames<1:
        raise ValueError('invalid video geometry')
    if not 0<policy.low_score<=policy.high_score<=1:
        raise ValueError('invalid confidence thresholds')
    if [i for i,_ in samples] != list(range(num_frames)):
        raise ValueError('postprocessing requires every source frame, in order')
    per_frame=[];removals=[]
    for frame,candidates in samples:
        kept,removed=deduplicate(candidates,frame=frame,width=width,height=height,policy=policy)
        per_frame.append(kept);removals.append(removed)
    tracks=_merge_duplicate_tracklets(_tracklets(per_frame,fps,width,policy),policy)
    selected,links=_select_chain(tracks,width,height,num_frames,fps,policy)
    observations=sorted([d for t in selected for d in t.observations],key=lambda d:d.frame)
    boxes=np.full((num_frames,4),np.nan,dtype=np.float64)
    status=['full_frame_fallback']*num_frames
    confidence=[None]*num_frames;anchors=[None]*num_frames
    for d in observations:
        boxes[d.frame]=d.box;confidence[d.frame]=d.score
        status[d.frame]='observed' if d.score>=policy.high_score else 'weak_supported'
        anchors[d.frame]=[d.frame]
    for a,b in zip(observations,observations[1:]):
        missing=b.frame-a.frame-1
        if missing<1:continue
        side=any(side_flags(a.box,width,policy.side_margin)) or any(side_flags(b.box,width,policy.side_margin))
        limit=policy.side_gap_seconds if side else policy.gap_seconds
        if missing/fps>limit+1e-9 or _match(a.box,b.box,width,policy) is None:continue
        for frame in range(a.frame+1,b.frame):
            t=(frame-a.frame)/(b.frame-a.frame)
            c=(1-t)*center(a.box)+t*center(b.box)
            size=np.exp((1-t)*np.log(a.box[2:]-a.box[:2])+t*np.log(b.box[2:]-b.box[:2]))
            boxes[frame]=np.r_[c-size*.5,c+size*.5]
            status[frame]='interpolated';anchors[frame]=[a.frame,b.frame]
    unsmoothed=boxes.copy();valid=np.isfinite(boxes).all(axis=1)
    radius=max(0,round(policy.smoothing_seconds*fps))
    edge_extended=[False]*num_frames
    shrink_guarded=[False]*num_frames
    guard_radius=max(radius,round(policy.shrink_guard_seconds*fps))
    # Smooth each supported run separately: no smoothing across an empty room.
    starts=np.flatnonzero(valid & ~np.r_[False,valid[:-1]])
    ends=np.flatnonzero(valid & ~np.r_[valid[1:],False])+1
    if policy.smoothing_method == 'legacy':
        for start,end in zip(starts,ends):
            for frame in range(start,end):
                lo,hi=max(start,frame-radius),min(end,frame+radius+1)
                ids=np.arange(lo,hi);weights=radius+1-np.abs(ids-frame)
                smooth=np.average(unsmoothed[lo:hi],axis=0,weights=weights)
                # The final crop must still contain today's measured visible body.
                boxes[frame]=np.r_[np.minimum(smooth[:2],unsmoothed[frame,:2]),np.maximum(smooth[2:],unsmoothed[frame,2:])]
                context=unsmoothed[max(start,frame-guard_radius):min(end,frame+guard_radius+1)]
                sizes=context[:,2:]-context[:,:2]
                if np.any((unsmoothed[frame,2:]-unsmoothed[frame,:2]) < .5*np.median(sizes,axis=0)):
                    # An ankle/arm crop can remain confident. Preserve a supported
                    # surrounding body envelope, without creating boxes in gaps.
                    envelope=np.r_[np.quantile(context[:,:2],.15,axis=0),np.quantile(context[:,2:],.85,axis=0)]
                    boxes[frame]=np.r_[np.minimum(boxes[frame,:2],envelope[:2]),np.maximum(boxes[frame,2:],envelope[2:])]
                    shrink_guarded[frame]=True
                for axis,bound in [(0,width-1),(1,height-1)]:
                    lower,upper=axis,axis+2;margin=bound*policy.side_margin
                    nearby=unsmoothed[lo:hi]
                    if boxes[frame,lower]<margin and np.min(nearby[:,lower])<bound*.01:
                        edge_extended[frame] |= boxes[frame,lower]>0;boxes[frame,lower]=0
                    if boxes[frame,upper]>bound-margin and np.max(nearby[:,upper])>bound*.99:
                        edge_extended[frame] |= boxes[frame,upper]<bound;boxes[frame,upper]=bound
                boxes[frame]=np.clip(boxes[frame],[0,0,0,0],[width-1,height-1,width-1,height-1])
    if policy.smoothing_method == 'large_envelope':
        from .box_smoothing import smooth_large_boxes
        boxes = smooth_large_boxes(unsmoothed, width=width, height=height, fps=fps,
            envelope_seconds=policy.envelope_seconds,
            smooth_seconds=policy.envelope_smooth_seconds,
            quantile=policy.envelope_quantile)
        edge_extended = [bool(ok and (np.any(b[:2] < u[:2]) or np.any(b[2:] > u[2:]))
            and (b[0]==0 or b[1]==0 or b[2]==width-1 or b[3]==height-1))
            for b,u,ok in zip(boxes,unsmoothed,valid)]
        shrink_guarded = list(map(bool, valid & np.any((boxes[:,2:]-boxes[:,:2]) > 2*(unsmoothed[:,2:]-unsmoothed[:,:2]), axis=1)))
    elif policy.smoothing_method != 'legacy':
        raise ValueError('unknown smoothing method')
    assert np.all(boxes[valid,2:] > boxes[valid,:2])
    return {'boxes':boxes.astype(np.float32),'unsmoothed_boxes':unsmoothed.astype(np.float32),
        'status':status,'confidence':confidence,'support_frames':anchors,'edge_extended':list(map(bool,edge_extended)),
        'shrink_guarded':shrink_guarded,
        'deduplicated':per_frame,'suppressed':removals,'selected_track_ids':[t.id for t in selected],
        'reacquisitions':links,'tracklets':len(tracks),'policy':asdict(policy)}
