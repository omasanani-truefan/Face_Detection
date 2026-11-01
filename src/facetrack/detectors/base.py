"""The detector contract every backend implements.

The tracker only ever sees ``list[Detection]``, so a backend can be swapped without
touching tracking code, and the benchmark drives every backend through the same call.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from facetrack.types import Detection, StageTimings


@dataclass
class DetectorConfig:
    model_path: str
    score_threshold: float = 0.5
    nms_threshold: float = 0.4
    # Longer side of the tensor fed to the model; 0 = native frame size (padded to 32).
    # Ignored by MediaPipe, whose graph resizes to a fixed 128/192 square itself.
    input_long_side: int = 640
    # 0 = let the runtime decide (ORT: physical cores).
    num_threads: int = 0
    top_k: int = 5000
    extra: dict[str, Any] = field(default_factory=dict)


class FaceDetector(ABC):
    name: str = "base"
    runtime: str = "unknown"
    precision: str = "fp32"
    keypoint_names: tuple[str, ...] | None = None

    def __init__(self, config: DetectorConfig):
        self.config = config
        if not Path(config.model_path).is_file():
            raise FileNotFoundError(
                f"{self.name}: model file not found at {config.model_path}. Run `python scripts/download_models.py`."
            )

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], StageTimings]:
        """Detect faces in one BGR frame. Boxes are returned in that frame's pixels."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass

    def describe(self) -> dict[str, Any]:
        """Everything a benchmark row needs to be reproducible."""
        path = Path(self.config.model_path)
        return {
            "detector": self.name,
            "model_file": path.name,
            "model_sha256": sha256_file(path),
            "model_size_bytes": path.stat().st_size,
            "runtime": self.runtime,
            "precision": self.precision,
            "input_long_side": self.config.input_long_side,
            "score_threshold": self.config.score_threshold,
            "nms_threshold": self.config.nms_threshold,
            "num_threads": self.config.num_threads,
            "keypoints": list(self.keypoint_names or ()),
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_and_build(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    kps: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    keypoint_names: tuple[str, ...] | None,
) -> list[Detection]:
    """Clip boxes to the frame and wrap them as ``Detection``s.

    Boxes are clipped rather than dropped: a face half out of frame is still a face the
    tracker must follow as it leaves.
    """
    out: list[Detection] = []
    if len(scores) == 0:
        return out
    b = boxes_xyxy.astype(np.float64, copy=True)
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, frame_w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, frame_h)
    for i in range(len(scores)):
        x1, y1, x2, y2 = b[i]
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        pts = [(float(kps[i, j, 0]), float(kps[i, j, 1])) for j in range(kps.shape[1])] if kps is not None else None
        out.append(Detection((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), float(scores[i]), pts, keypoint_names))
    return out
