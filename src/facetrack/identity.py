"""Phase 3 extension points -- persistent identity. **Interfaces only; nothing here is
implemented or called with real models in Phases 1-2.**

The recognition stage hangs off the tracker's events without the tracker knowing:

1. While a track is visible (``on_observation``), an ``EvidenceCollector`` scores each
   face crop with a ``FaceQualityScorer`` and keeps the best few, embedding them with an
   ``EmbeddingExtractor`` (e.g. an ArcFace-family model) *while the face is on screen*.
2. When the tracker emits ``track_ended``, the collector hands the accumulated
   embeddings to an ``IdentityMatcher``.
3. The matcher compares them against an ``IdentityStore`` of enrolled people and returns
   a ``person_id`` or ``UNKNOWN``. Different Track IDs (one person leaving and coming
   back) then map to the same ``person_id``.

Never try to recognise a face from a frame taken after the track ended: by then the
face is gone. Evidence must be captured during the track.

Keypoint note: YuNet and SCRFD emit the 5 points ArcFace alignment uses (eyes, nose,
mouth corners). BlazeFace emits eyes, nose, mouth *centre* and ear tragions -- no mouth
corners -- so a MediaPipe-based pipeline needs a different alignment template or a
landmark model before embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from facetrack.tracking.base import Track, TrackEvent
from facetrack.types import Detection, Frame

UNKNOWN = "UNKNOWN"


@dataclass
class FaceEvidence:
    track_id: int
    timestamp_ms: float
    quality: float
    bbox: tuple[float, float, float, float]
    keypoints: list[tuple[float, float]] | None
    embedding: np.ndarray | None = None  # filled by an EmbeddingExtractor in Phase 3


@dataclass
class IdentityDecision:
    track_id: int
    person_id: str  # enrolled id, or UNKNOWN
    score: float
    evidence_count: int
    extra: dict = field(default_factory=dict)


class FaceQualityScorer(Protocol):
    def score(self, frame: Frame, detection: Detection) -> float:
        """Higher is better (size, sharpness, frontal pose, detector confidence)."""


class EmbeddingExtractor(Protocol):
    def embed(self, frame: Frame, detection: Detection) -> np.ndarray:
        """Align with ``detection.keypoints`` and return an L2-normalised embedding."""


class IdentityStore(Protocol):
    def search(self, embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Nearest enrolled persons as (person_id, similarity)."""


class IdentityMatcher(Protocol):
    def decide(self, track_id: int, evidence: list[FaceEvidence]) -> IdentityDecision: ...


@runtime_checkable
class TrackObserver(Protocol):
    """Anything that wants per-frame access to tracks -- the hook Phase 3 plugs into.

    ``FacePipeline`` calls ``on_observation`` for every confirmed track matched by a
    detection on that frame (so the frame pixels are available), and ``on_event`` for
    every lifecycle event, in order.
    """

    def on_observation(self, frame: Frame, track: Track, detection: Detection) -> None: ...

    def on_event(self, event: TrackEvent) -> None: ...
