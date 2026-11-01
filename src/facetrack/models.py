"""Pretrained model manifest: where every weight file comes from and what it may be used for.

Checksums pin the exact bytes benchmarked. ``latest`` URLs (MediaPipe) can change
underneath us, which the checksum turns into a loud failure instead of a silent change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class ModelSpec:
    file: str
    url: str | None
    sha256: str | None
    weights_license: str
    # For files we derive locally from a downloaded one (see scripts/download_models.py).
    derived_from: str | None = None
    archive_member: str | None = None


MODELS: dict[str, ModelSpec] = {
    "yunet_2023mar.onnx": ModelSpec(
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "MIT (opencv_zoo)",
    ),
    "yunet_2023mar_int8.onnx": ModelSpec(
        "face_detection_yunet_2023mar_int8.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx",
        "321aa5a6afabf7ecc46a3d06bfab2b579dc96eb5c3be7edd365fa04502ad9294",
        "MIT (opencv_zoo)",
    ),
    # Same weights, with only the I/O shape metadata relaxed from 640x640 to symbolic
    # H/W so ONNX Runtime accepts any stride-32 input (OpenCV ignores the metadata).
    "yunet_2023mar_dyn.onnx": ModelSpec(
        "face_detection_yunet_2023mar_dyn.onnx", None, None, "MIT (opencv_zoo)", derived_from="yunet_2023mar.onnx"
    ),
    "yunet_2023mar_int8_dyn.onnx": ModelSpec(
        "face_detection_yunet_2023mar_int8_dyn.onnx", None, None, "MIT (opencv_zoo)", derived_from="yunet_2023mar_int8.onnx"
    ),
    "scrfd_500m_kps.onnx": ModelSpec(
        "scrfd_500m_kps.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
        "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
        "InsightFace pretrained models: non-commercial research only",
        archive_member="det_500m.onnx",
    ),
    "blaze_face_short_range.tflite": ModelSpec(
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        "Apache-2.0 (MediaPipe; see model card)",
    ),
    "blaze_face_full_range.tflite": ModelSpec(
        "blaze_face_full_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
        "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b",
        "Apache-2.0 (MediaPipe; see model card)",
    ),
}


def model_path(key: str) -> Path:
    return MODELS_DIR / MODELS[key].file
