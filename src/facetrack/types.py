"""Data contracts shared by every stage.

Conventions, fixed here so no stage has to guess:

* Images are ``uint8`` BGR ``H x W x 3`` arrays (OpenCV order). A detector that needs
  RGB converts internally and counts the conversion as its own preprocessing.
* Boxes cross module boundaries as ``[x, y, w, h]`` in *original frame* pixels.
* Every timestamp is in **milliseconds**, taken from the video's presentation time or
  a monotonic clock for cameras -- never derived from a frame counter, because browsers
  and phones do not deliver frames at a fixed rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 5-point order shared by YuNet and SCRFD, and the order ArcFace-style alignment expects
# (subject's right eye first, so it appears on the *left* of a frontal image).
KEYPOINTS_5 = ("right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner")
# MediaPipe BlazeFace emits 6 points; it has no mouth corners, only a mouth centre.
KEYPOINTS_6_BLAZEFACE = ("right_eye", "left_eye", "nose_tip", "mouth_center", "right_ear_tragion", "left_ear_tragion")


@dataclass(slots=True)
class Frame:
    image: np.ndarray
    frame_id: int
    timestamp_ms: float
    source_id: str = "default"


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]  # x, y, w, h in frame pixels
    confidence: float
    keypoints: list[tuple[float, float]] | None = None
    keypoint_names: tuple[str, ...] | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.bbox
        return x, y, x + w, y + h

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"bbox": [round(v, 2) for v in self.bbox], "confidence": round(float(self.confidence), 4)}
        d["keypoints"] = [[round(x, 2), round(y, 2)] for x, y in self.keypoints] if self.keypoints else []
        return d


@dataclass(slots=True)
class StageTimings:
    """Wall-clock milliseconds per stage of one detector call.

    Backends that fuse stages (OpenCV's FaceDetectorYN, MediaPipe Tasks) cannot report
    a split; they put everything in ``inference_ms`` and set ``fused=True`` so reports
    never present a fused number as a pure forward-pass time.
    """

    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    fused: bool = False

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(slots=True)
class DetectionResult:
    frame_id: int
    timestamp_ms: float
    source_id: str
    detections: list[Detection]
    timings: StageTimings = field(default_factory=StageTimings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": round(self.timestamp_ms, 3),
            "camera_id": self.source_id,
            "detections": [d.to_dict() for d in self.detections],
        }
