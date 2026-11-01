"""MediaPipe Tasks Face Detector (BlazeFace short-range / full-range).

The Tasks graph resizes, letterboxes, runs, decodes and suppresses inside one call, so
only the BGR->RGB conversion and the result unpacking are separable; the rest is
reported as fused inference. ``input_long_side`` is ignored -- the model's input is a
fixed square (128 px short-range, 192 px full-range) chosen by the graph.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from facetrack.detectors.base import DetectorConfig, FaceDetector
from facetrack.types import KEYPOINTS_6_BLAZEFACE, Detection, StageTimings


class MediaPipeFaceDetector(FaceDetector):
    keypoint_names = KEYPOINTS_6_BLAZEFACE

    def __init__(self, config: DetectorConfig, name: str = "mediapipe"):
        super().__init__(config)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as e:  # pragma: no cover - optional backend
            raise ImportError("MediaPipe backend needs `pip install facetrack[mediapipe]`") from e
        self.name = name
        self.runtime = f"mediapipe-tasks {mp.__version__} (XNNPACK CPU)"
        # The published .tflite files live under .../float16/...: fp16 weights, run by XNNPACK.
        self.precision = "fp16 weights"
        self._mp = mp
        opts = vision.FaceDetectorOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=config.model_path, delegate=mp_tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=config.score_threshold,
            min_suppression_threshold=config.nms_threshold,
        )
        self.det = vision.FaceDetector.create_from_options(opts)

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], StageTimings]:
        fh, fw = image_bgr.shape[:2]
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        t1 = time.perf_counter()
        result = self.det.detect(mp_image)
        t2 = time.perf_counter()
        dets: list[Detection] = []
        for d in result.detections:
            bb = d.bounding_box
            x1, y1 = max(0.0, float(bb.origin_x)), max(0.0, float(bb.origin_y))
            x2, y2 = min(float(fw), float(bb.origin_x + bb.width)), min(float(fh), float(bb.origin_y + bb.height))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            kps = [(k.x * fw, k.y * fh) for k in d.keypoints] if d.keypoints else None
            dets.append(Detection((x1, y1, x2 - x1, y2 - y1), float(d.categories[0].score), kps, self.keypoint_names))
        t3 = time.perf_counter()
        return dets, StageTimings((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, fused=True)

    def close(self) -> None:
        self.det.close()
