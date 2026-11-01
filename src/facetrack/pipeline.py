"""Source-agnostic frame pipeline: detector -> tracker -> observers.

The pipeline takes a ``Frame`` and never asks where it came from, so the uploaded-video
path and the live-camera path run identical detection and tracking code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from facetrack.detectors.base import FaceDetector
from facetrack.identity import TrackObserver
from facetrack.tracking.base import BaseTracker, TrackerOutput, TrackEvent, TrackState
from facetrack.types import Detection, Frame, StageTimings


def _as_bgr(image: np.ndarray | None) -> np.ndarray | None:
    """Coerce grayscale / BGRA frames to 3-channel BGR; None for anything unusable."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0 or min(image.shape[:2]) < 8:
        return None
    if image.dtype != np.uint8:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    return None


@dataclass
class PipelineResult:
    frame: Frame
    detections: list[Detection]
    tracks: TrackerOutput
    detector_timings: StageTimings | None
    track_ms: float
    total_ms: float
    events: list[TrackEvent] = field(default_factory=list)

    @property
    def detector_ran(self) -> bool:
        return self.detector_timings is not None

    def to_dict(self) -> dict:
        t = self.detector_timings
        return {
            "frame_id": self.frame.frame_id,
            "timestamp_ms": round(self.frame.timestamp_ms, 3),
            "camera_id": self.frame.source_id,
            "detector_ran": self.detector_ran,
            "detections": [d.to_dict() for d in self.detections],
            "tracks": self.tracks.to_dict()["tracks"],
            "events": [e.to_dict() for e in self.events],
            "timings_ms": {
                "preprocess": round(t.preprocess_ms, 3) if t else None,
                "inference": round(t.inference_ms, 3) if t else None,
                "postprocess": round(t.postprocess_ms, 3) if t else None,
                "detector_fused": t.fused if t else None,
                "tracking": round(self.track_ms, 3),
                "total": round(self.total_ms, 3),
            },
        }


class FacePipeline:
    def __init__(self, detector: FaceDetector, tracker: BaseTracker, detect_every: int = 1,
                 observers: Iterable[TrackObserver] = ()):
        if detect_every < 1:
            raise ValueError("detect_every must be >= 1")
        self.detector = detector
        self.tracker = tracker
        self.detect_every = detect_every
        self.observers = list(observers)
        self._n = 0

    def process(self, frame: Frame) -> PipelineResult:
        t0 = time.perf_counter()
        frame.image = _as_bgr(frame.image)
        # An unusable frame is treated like a skipped one: tracks coast, nothing expires early.
        run_detector = self._n % self.detect_every == 0 and frame.image is not None
        self._n += 1
        if run_detector:
            dets, timings = self.detector.detect(frame.image)
            t1 = time.perf_counter()
            out = self.tracker.update(dets, frame.timestamp_ms, frame.frame_id)
        else:
            dets, timings = [], None
            t1 = time.perf_counter()
            out = self.tracker.predict(frame.timestamp_ms, frame.frame_id)
        t2 = time.perf_counter()
        self._notify(frame, out)
        return PipelineResult(frame, dets, out, timings, (t2 - t1) * 1e3, (time.perf_counter() - t0) * 1e3, out.events)

    def finish(self, timestamp_ms: float, frame_id: int = -1) -> list[TrackEvent]:
        """Stream ended: close every live track so consumers see track_ended."""
        events = self.tracker.finish(timestamp_ms, frame_id)
        for e in events:
            for o in self.observers:
                o.on_event(e)
        return events

    def reset(self) -> None:
        """Restart (e.g. video rewound): drop live tracks; IDs keep counting up."""
        self.tracker.reset()
        self._n = 0

    def _notify(self, frame: Frame, out: TrackerOutput) -> None:
        if frame.image is None:
            return
        if not self.observers:
            return
        for t in out.tracks:
            if t.state is TrackState.CONFIRMED and t.matched and out.detector_ran and t.last_detection is not None:
                for o in self.observers:
                    o.on_observation(frame, t, t.last_detection)
        for e in out.events:
            for o in self.observers:
                o.on_event(e)
