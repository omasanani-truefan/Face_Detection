"""Lightweight face detection and temporary multi-face tracking.

The package is split so each stage can be swapped without touching the others:

    sources  -> preprocessing (inside each detector) -> detectors -> tracking -> events / viz

Track IDs produced here are *temporary*: they live only while a face is continuously
tracked. Persistent person identity is a later phase (see ``facetrack.identity``).
"""

from facetrack.types import Detection, DetectionResult, Frame, StageTimings

__all__ = ["Detection", "DetectionResult", "Frame", "StageTimings"]
__version__ = "0.1.0"
