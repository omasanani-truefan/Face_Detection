"""Detector registry: one name per (weights, runtime) combination that we benchmark.

Defaults follow each model's official demo (score / NMS thresholds) so a backend starts
from its authors' operating point; the benchmark then tunes thresholds on the dev split.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from facetrack.detectors.base import DetectorConfig, FaceDetector
from facetrack.models import model_path


def _yunet_ort(cfg: DetectorConfig) -> FaceDetector:
    from facetrack.detectors.yunet import YuNetORT

    return YuNetORT(cfg)


def _yunet_cv(cfg: DetectorConfig) -> FaceDetector:
    from facetrack.detectors.yunet import YuNetOpenCV

    return YuNetOpenCV(cfg)


def _scrfd(cfg: DetectorConfig) -> FaceDetector:
    from facetrack.detectors.scrfd import SCRFD

    return SCRFD(cfg)


def _mediapipe(name: str) -> Callable[[DetectorConfig], FaceDetector]:
    def build(cfg: DetectorConfig) -> FaceDetector:
        from facetrack.detectors.mediapipe_blazeface import MediaPipeFaceDetector

        return MediaPipeFaceDetector(cfg, name=name)

    return build


_REGISTRY: dict[str, tuple[Callable[[DetectorConfig], FaceDetector], DetectorConfig]] = {
    "yunet": (_yunet_ort, DetectorConfig(str(model_path("yunet_2023mar_dyn.onnx")), 0.6, 0.3)),
    "yunet_int8": (_yunet_ort, DetectorConfig(str(model_path("yunet_2023mar_int8_dyn.onnx")), 0.6, 0.3)),
    "yunet_cv": (_yunet_cv, DetectorConfig(str(model_path("yunet_2023mar.onnx")), 0.6, 0.3)),
    "scrfd_500m_kps": (_scrfd, DetectorConfig(str(model_path("scrfd_500m_kps.onnx")), 0.5, 0.4)),
    "mediapipe_short": (
        _mediapipe("mediapipe_short"),
        DetectorConfig(str(model_path("blaze_face_short_range.tflite")), 0.5, 0.3, input_long_side=0),
    ),
    "mediapipe_full": (
        _mediapipe("mediapipe_full"),
        DetectorConfig(str(model_path("blaze_face_full_range.tflite")), 0.5, 0.3, input_long_side=0),
    ),
}

DETECTOR_NAMES = tuple(_REGISTRY)


def create_detector(name: str, **overrides: Any) -> FaceDetector:
    """Build a registered detector, overriding any ``DetectorConfig`` field by keyword.

    ``None`` means "keep the default", so native input resolution is requested with
    ``input_long_side=0``, not ``None``.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; choose from {', '.join(_REGISTRY)}")
    build, defaults = _REGISTRY[name]
    return build(replace(defaults, **{k: v for k, v in overrides.items() if v is not None}))


__all__ = ["DETECTOR_NAMES", "DetectorConfig", "FaceDetector", "create_detector"]
