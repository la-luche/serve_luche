"""Frozen reviewed merge implementation for exact differential regression checks."""
import numpy as np
from core.person_postprocess import Tracklet, area, intersection, iou, center

def reference_merge_duplicate_tracklets(tracks, policy):
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

