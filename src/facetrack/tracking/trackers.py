"""The three association strategies benchmarked against each other.

They share ``BaseTracker``'s lifecycle and differ only in how live tracks are matched to
detections, so a difference in the benchmark is a difference in association.
"""

from __future__ import annotations

import numpy as np

from facetrack.tracking.base import Association, BaseTracker, Track, TrackState
from facetrack.tracking.kalman import SortKalman, XyahKalman
from facetrack.tracking.matching import greedy, hungarian, iou_matrix, normalized_center_distance
from facetrack.types import Detection


class IoUTracker(BaseTracker):
    """No motion model: each track is matched at its last observed box, greedily by IoU,
    with a centre-distance fallback for small fast-moving faces whose boxes no longer
    overlap between frames."""

    name = "iou"

    def _associate(self, tracks: list[Track], dets: list[Detection]) -> Association:
        tb, db = self._boxes(tracks), self._det_boxes(dets)
        iou = iou_matrix(tb, db)
        cd = normalized_center_distance(tb, db)
        gated = (iou >= self.cfg.iou_threshold) | (cd <= self.cfg.center_distance_threshold)
        # Rank by overlap, break ties (and score non-overlapping pairs) by closeness.
        score = np.where(gated, iou + 0.1 * (1 - np.minimum(cd, 1)), -np.inf)
        matches, _, unmatched = greedy(score, min_score=-1e9)
        return Association(matches, unmatched)


class SortTracker(BaseTracker):
    """SORT: constant-velocity Kalman on (centre, area, aspect), one Hungarian assignment
    on IoU between predicted boxes and detections."""

    name = "sort"
    uses_motion = True

    def _new_motion(self, box: np.ndarray) -> SortKalman:
        return SortKalman(box)

    def _associate(self, tracks: list[Track], dets: list[Detection]) -> Association:
        iou = self._match_iou(tracks, self._det_boxes(dets))
        matches, _, unmatched = hungarian(1.0 - iou, max_cost=1.0 - self.cfg.iou_threshold)
        return Association(matches, unmatched)


class ByteTracker(BaseTracker):
    """ByteTrack: Kalman on (centre, aspect, height) and BYTE association --

    1. high-score detections vs confirmed + lost tracks (IoU fused with score),
    2. low-score detections vs confirmed tracks still unmatched (recovers faces whose
       score dipped from blur, occlusion or a head turn),
    3. remaining high-score detections vs tentative tracks.

    Only high-score detections may start a track. This only helps if the detector is run
    with a threshold at or below ``low_det_threshold``.
    """

    name = "bytetrack"
    uses_motion = True

    def _new_motion(self, box: np.ndarray) -> XyahKalman:
        return XyahKalman(box)

    def _min_det_score(self) -> float:
        return self.cfg.low_det_threshold

    def _associate(self, tracks: list[Track], dets: list[Detection]) -> Association:
        cfg = self.cfg
        scores = np.array([d.confidence for d in dets])
        high = [i for i, s in enumerate(scores) if s >= cfg.det_threshold]
        low = [i for i, s in enumerate(scores) if s < cfg.det_threshold]
        active = [i for i, t in enumerate(tracks) if t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        tentative = [i for i, t in enumerate(tracks) if t.state is TrackState.TENTATIVE]
        tb, db = self._boxes(tracks), self._det_boxes(dets)
        matches: list[tuple[int, int]] = []

        # Stage 1: high-score detections, IoU fused with detection score (ByteTrack's fuse_score).
        iou = self._match_iou([tracks[i] for i in active], db[high])
        fused = 1.0 - iou * scores[high][None, :] if len(high) else iou
        # A pair must clear the raw-IoU gate too, or fusing could accept a 0.2-IoU match
        # on the strength of a confident detection.
        cost = np.where(iou >= cfg.iou_threshold, fused, np.inf) if iou.size else fused
        m, ut, ud = hungarian(np.nan_to_num(cost, posinf=1e6), max_cost=1.0 - cfg.iou_threshold * cfg.det_threshold)
        matches += [(active[r], high[c]) for r, c in m]
        rem_active = [active[r] for r in ut]
        rem_high = [high[c] for c in ud]

        # Stage 2: low-score detections may only extend tracks that are currently confirmed.
        confirmed_left = [i for i in rem_active if tracks[i].state is TrackState.CONFIRMED]
        if confirmed_left and low:
            iou = iou_matrix(tb[confirmed_left], db[low])
            m, _, _ = hungarian(1.0 - iou, max_cost=1.0 - cfg.low_iou_threshold)
            matches += [(confirmed_left[r], low[c]) for r, c in m]

        # Stage 3: tentative tracks against leftover high-score detections.
        if tentative and rem_high:
            iou = iou_matrix(tb[tentative], db[rem_high])
            m, _, ud = hungarian(1.0 - iou, max_cost=1.0 - cfg.iou_threshold)
            matches += [(tentative[r], rem_high[c]) for r, c in m]
            rem_high = [rem_high[c] for c in ud]

        return Association(matches, rem_high)


TRACKERS: dict[str, type[BaseTracker]] = {"iou": IoUTracker, "sort": SortTracker, "bytetrack": ByteTracker}


def create_tracker(name: str, config=None) -> BaseTracker:
    from facetrack.tracking.base import TrackerConfig

    if name not in TRACKERS:
        raise KeyError(f"unknown tracker {name!r}; choose from {', '.join(TRACKERS)}")
    cfg = config if isinstance(config, TrackerConfig) else TrackerConfig.from_dict(config)
    return TRACKERS[name](cfg)
