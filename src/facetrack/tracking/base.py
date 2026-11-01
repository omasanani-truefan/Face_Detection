"""Track lifecycle shared by every tracker; subclasses only decide *association*.

States::

    TENTATIVE --(min_hits observations)--> CONFIRMED <--(matched)-- LOST
        |                                     |  (missed)            |
        | unseen > tentative_timeout_ms       +--------------------> |
        v                                                            | unseen > max_lost_ms
    (dropped silently: never announced)                              v
                                                                TERMINATED (track_ended)

All expiry is measured on the timestamps the caller passes in, never on frame counts:
a phone dropping to 12 fps must not keep a vanished face alive 2.5x longer than a
desktop at 30 fps.

Track IDs are allocated from a per-tracker counter and never reused, so a person who
re-enters after their track terminated gets a new ID. Recognising them as the same
person is a separate, later stage (``facetrack.identity``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

from facetrack.tracking.matching import iou_matrix
from facetrack.types import Detection


class TrackState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    TERMINATED = "terminated"


class EventType(str, Enum):
    STARTED = "track_started"
    UPDATED = "track_updated"
    LOST = "track_lost"
    RECOVERED = "track_recovered"
    ENDED = "track_ended"


@dataclass
class TrackerConfig:
    # Detections below this never start a track (and, outside ByteTrack, are ignored).
    det_threshold: float = 0.5
    # ByteTrack only: detections in [low_det_threshold, det_threshold) may extend an
    # existing confirmed track but never start one.
    low_det_threshold: float = 0.1
    # Minimum score to *start* a track; None = det_threshold.
    new_track_threshold: float | None = None
    min_hits: int = 3
    tentative_timeout_ms: float = 300.0
    max_lost_ms: float = 1000.0
    # Primary association gate: minimum IoU between a track's (predicted) box and a detection.
    iou_threshold: float = 0.3
    # ByteTrack second stage (low-score detections) gate.
    low_iou_threshold: float = 0.5
    # IoU tracker fallback: match boxes that don't overlap if centres are within this
    # many box-diagonals (catches fast motion between frames).
    center_distance_threshold: float = 0.6
    # Frame interval at which the published Kalman tunings apply (see kalman.py).
    ref_frame_ms: float = 1000.0 / 30.0
    emit_updates: bool = True
    # Observation-centric recovery (after OC-SORT, Cao et al. 2023): a LOST track may also
    # match on its last *observed* box, and its motion model restarts from the recovering
    # detection. Fixes the measured failure where a partly occluded, stationary face
    # shrinks sideways, the Kalman filter reads that as velocity, and the prediction
    # drifts off the face before it reappears. Off = the published SORT / ByteTrack.
    lost_match_last_observation: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TrackerConfig":
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown tracker config keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class Track:
    track_id: int
    state: TrackState
    bbox_xyxy: np.ndarray
    confidence: float
    first_seen_ms: float
    last_seen_ms: float
    keypoints: list[tuple[float, float]] | None = None
    hits: int = 1
    confirmed_ms: float | None = None
    matched: bool = True  # matched a detection at the latest detector update
    motion: Any = None  # Kalman model, or None for the IoU tracker
    last_detection: Detection | None = None

    def snapshot(self) -> "Track":
        """Immutable-by-convention copy for outputs: callers that keep results (UI,
        exporters, metrics) must not see later updates rewrite earlier frames."""
        return replace(self, bbox_xyxy=self.bbox_xyxy.copy(), motion=None)

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(v) for v in self.bbox_xyxy)
        return x1, y1, x2 - x1, y2 - y1

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": [round(v, 2) for v in self.bbox_xywh],
            "confidence": round(self.confidence, 4),
            "status": self.state.value,
            "first_seen": round(self.first_seen_ms, 3),
            "last_seen": round(self.last_seen_ms, 3),
            "hits": self.hits,
        }


@dataclass
class TrackEvent:
    type: EventType
    track_id: int
    timestamp_ms: float
    frame_id: int
    track: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.type.value, "track_id": self.track_id, "timestamp_ms": round(self.timestamp_ms, 3),
                "frame_id": self.frame_id, "track": self.track}


@dataclass
class TrackerOutput:
    frame_id: int
    timestamp_ms: float
    tracks: list[Track]  # every live track (tentative, confirmed, lost)
    events: list[TrackEvent] = field(default_factory=list)
    detector_ran: bool = True

    def visible(self, include_lost: bool = False) -> list[Track]:
        """Confirmed tracks matched at this update (plus lost ones if asked) -- what a UI
        or a MOT evaluation should report."""
        return [t for t in self.tracks if (t.state is TrackState.CONFIRMED and t.matched)
                or (include_lost and t.state is TrackState.LOST)]

    def to_dict(self) -> dict[str, Any]:
        return {"frame_id": self.frame_id, "timestamp_ms": round(self.timestamp_ms, 3),
                "tracks": [t.to_dict() for t in self.tracks if t.state is not TrackState.TENTATIVE]}


@dataclass
class Association:
    matches: list[tuple[int, int]]  # (track index, detection index)
    new_track_candidates: list[int]  # detection indices allowed to start tracks


class BaseTracker(ABC):
    name = "base"
    uses_motion = False

    def __init__(self, config: TrackerConfig | None = None):
        self.cfg = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self._last_ts: float | None = None

    # -- subclass hooks -------------------------------------------------------------
    @abstractmethod
    def _associate(self, tracks: list[Track], dets: list[Detection]) -> Association:
        """Match live tracks (boxes already advanced to the current time) to detections."""

    def _new_motion(self, box: np.ndarray) -> Any:
        return None

    # -- public API -----------------------------------------------------------------
    @property
    def ids_issued(self) -> int:
        return self._next_id - 1

    def reset(self) -> None:
        """Forget all tracks, e.g. on a video restart. IDs keep increasing so a reset can
        never hand an old ID to a different face."""
        self.tracks = []
        self._last_ts = None

    def update(self, detections: list[Detection], timestamp_ms: float, frame_id: int = -1) -> TrackerOutput:
        """Advance to ``timestamp_ms`` and fold in this frame's detections."""
        events: list[TrackEvent] = []
        self._advance(timestamp_ms)
        cfg = self.cfg
        dets = [d for d in detections if d.confidence >= self._min_det_score()]

        assoc = self._associate(self.tracks, dets)
        matched_tracks = set()
        for ti, di in assoc.matches:
            t, d = self.tracks[ti], dets[di]
            matched_tracks.add(ti)
            box = np.asarray(d.xyxy, dtype=np.float64)
            if t.motion is not None:
                if t.state is TrackState.LOST and cfg.lost_match_last_observation:
                    t.motion = self._new_motion(box)  # discard the velocity that drifted
                else:
                    t.motion.update(box)
                t.bbox_xyxy = t.motion.box()
            else:
                t.bbox_xyxy = box
            t.confidence, t.keypoints, t.last_detection = d.confidence, d.keypoints, d
            t.last_seen_ms, t.matched = timestamp_ms, True
            t.hits += 1
            if t.state is TrackState.TENTATIVE and t.hits >= cfg.min_hits:
                self._confirm(t, timestamp_ms, frame_id, events)
            elif t.state is TrackState.LOST:
                t.state = TrackState.CONFIRMED
                events.append(self._event(EventType.RECOVERED, t, timestamp_ms, frame_id))
            elif t.state is TrackState.CONFIRMED and cfg.emit_updates:
                events.append(self._event(EventType.UPDATED, t, timestamp_ms, frame_id))

        for ti, t in enumerate(self.tracks):
            if ti in matched_tracks:
                continue
            t.matched = False
            if t.state is TrackState.CONFIRMED:
                # One miss never ends a track: it goes LOST and gets the grace period.
                t.state = TrackState.LOST
                events.append(self._event(EventType.LOST, t, timestamp_ms, frame_id))

        new_thr = cfg.new_track_threshold if cfg.new_track_threshold is not None else cfg.det_threshold
        for di in assoc.new_track_candidates:
            d = dets[di]
            if d.confidence < new_thr:
                continue
            box = np.asarray(d.xyxy, dtype=np.float64)
            t = Track(self._next_id, TrackState.TENTATIVE, box, d.confidence, timestamp_ms, timestamp_ms,
                      d.keypoints, motion=self._new_motion(box), last_detection=d)
            self._next_id += 1
            self.tracks.append(t)
            if cfg.min_hits <= 1:
                self._confirm(t, timestamp_ms, frame_id, events)

        self._expire(timestamp_ms, frame_id, events)
        return TrackerOutput(frame_id, timestamp_ms, [t.snapshot() for t in self.tracks], events, detector_ran=True)

    def predict(self, timestamp_ms: float, frame_id: int = -1) -> TrackerOutput:
        """Advance motion models without a detector result (frame skipping). No lifecycle
        changes: a frame the detector did not look at is not evidence a face is gone."""
        self._advance(timestamp_ms)
        return TrackerOutput(frame_id, timestamp_ms, [t.snapshot() for t in self.tracks], [], detector_ran=False)

    def finish(self, timestamp_ms: float, frame_id: int = -1) -> list[TrackEvent]:
        """End every live track (stream stopped). Emits track_ended for announced tracks."""
        events: list[TrackEvent] = []
        for t in self.tracks:
            if t.state in (TrackState.CONFIRMED, TrackState.LOST):
                t.state = TrackState.TERMINATED
                events.append(self._event(EventType.ENDED, t, timestamp_ms, frame_id))
        self.tracks = []
        return events

    # -- internals ------------------------------------------------------------------
    def _min_det_score(self) -> float:
        return self.cfg.det_threshold

    def _advance(self, timestamp_ms: float) -> None:
        if self._last_ts is not None and timestamp_ms < self._last_ts:
            raise ValueError(f"timestamps must not go backwards ({timestamp_ms} < {self._last_ts}); call reset() on seek")
        dt = 1.0 if self._last_ts is None else (timestamp_ms - self._last_ts) / self.cfg.ref_frame_ms
        self._last_ts = timestamp_ms
        if dt <= 0:
            return
        for t in self.tracks:
            if t.motion is not None:
                t.bbox_xyxy = t.motion.predict(dt)

    def _confirm(self, t: Track, ts: float, frame_id: int, events: list[TrackEvent]) -> None:
        t.state = TrackState.CONFIRMED
        t.confirmed_ms = ts
        events.append(self._event(EventType.STARTED, t, ts, frame_id))

    def _expire(self, ts: float, frame_id: int, events: list[TrackEvent]) -> None:
        keep: list[Track] = []
        for t in self.tracks:
            unseen = ts - t.last_seen_ms
            if t.state is TrackState.TENTATIVE and not t.matched and unseen > self.cfg.tentative_timeout_ms:
                continue  # never announced, so it ends silently
            if t.state is TrackState.LOST and unseen > self.cfg.max_lost_ms:
                t.state = TrackState.TERMINATED
                events.append(self._event(EventType.ENDED, t, ts, frame_id))
                continue
            keep.append(t)
        self.tracks = keep

    @staticmethod
    def _event(kind: EventType, t: Track, ts: float, frame_id: int) -> TrackEvent:
        return TrackEvent(kind, t.track_id, ts, frame_id, t.to_dict())

    def _match_iou(self, tracks: list[Track], det_boxes: np.ndarray) -> np.ndarray:
        """IoU of each track's current (predicted) box with each detection; for LOST
        tracks under observation-centric recovery, the better of predicted and last
        observed box."""
        iou = iou_matrix(self._boxes(tracks), det_boxes)
        if self.cfg.lost_match_last_observation and len(det_boxes):
            for i, t in enumerate(tracks):
                if t.state is TrackState.LOST and t.last_detection is not None:
                    obs = np.asarray([t.last_detection.xyxy], dtype=np.float64)
                    iou[i] = np.maximum(iou[i], iou_matrix(obs, det_boxes)[0])
        return iou

    @staticmethod
    def _boxes(tracks: list[Track]) -> np.ndarray:
        return np.array([t.bbox_xyxy for t in tracks], dtype=np.float64).reshape(-1, 4)

    @staticmethod
    def _det_boxes(dets: list[Detection]) -> np.ndarray:
        return np.array([d.xyxy for d in dets], dtype=np.float64).reshape(-1, 4)
