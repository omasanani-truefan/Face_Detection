"""SCRFD (InsightFace), the ``det_500m.onnx`` export = SCRFD-500M with 5 keypoints.

Anchor-based head with 2 anchors per cell at strides 8/16/32; the exported graph already
applies the sigmoid. Decoding follows InsightFace ``python-package/insightface/model_zoo/scrfd.py``:
RGB input normalised as ``(x - 127.5) / 128``, boxes and keypoints are distances from
the anchor centre in units of the stride.

Licence: the InsightFace *code* is MIT, but its pretrained weights are released for
non-commercial research only -- see docs/REPORT.md before shipping this backend.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from facetrack.detectors.base import DetectorConfig, FaceDetector, clip_and_build
from facetrack.detectors.ort_common import make_session
from facetrack.preprocess import letterbox, nms
from facetrack.types import KEYPOINTS_5, Detection, StageTimings

STRIDES = (8, 16, 32)


class SCRFD(FaceDetector):
    name = "scrfd_500m_kps"
    runtime = "onnxruntime-cpu"
    keypoint_names = KEYPOINTS_5

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.session = make_session(config.model_path, config.num_threads)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        if len(outputs) != 9:
            raise ValueError(f"expected the 9-output KPS export of SCRFD, got {len(outputs)} outputs")
        self.num_anchors = 2
        self._anchor_cache: dict[tuple[int, int], list[np.ndarray]] = {}

    def _anchors(self, w: int, h: int) -> list[np.ndarray]:
        key = (w, h)
        if key not in self._anchor_cache:
            grids = []
            for s in STRIDES:
                rows, cols = h // s, w // s
                c, r = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
                centers = np.stack([c.ravel(), r.ravel()], axis=1) * s
                grids.append(np.repeat(centers, self.num_anchors, axis=0))
            self._anchor_cache[key] = grids
        return self._anchor_cache[key]

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], StageTimings]:
        fh, fw = image_bgr.shape[:2]
        t0 = time.perf_counter()
        long_side = self.config.input_long_side or max(fh, fw)
        padded, lb = letterbox(image_bgr, long_side)
        # blobFromImage does BGR->RGB, mean/std and HWC->NCHW in one C++ pass.
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 128.0, None, (127.5, 127.5, 127.5), swapRB=True)
        t1 = time.perf_counter()
        outs = self.session.run(None, {self.input_name: blob})
        t2 = time.perf_counter()

        # Group outputs by role from their shapes, not their (auto-generated) names.
        by_rows: dict[int, dict[int, np.ndarray]] = {}
        for o in outs:
            o = o.reshape(-1, o.shape[-1])
            by_rows.setdefault(o.shape[0], {})[o.shape[1]] = o
        thr = self.config.score_threshold
        boxes, scores, kpss = [], [], []
        for s, centers in zip(STRIDES, self._anchors(lb.input_w, lb.input_h)):
            group = by_rows[len(centers)]
            score, dist, kd = group[1][:, 0], group[4], group[10]
            keep = score >= thr
            if not keep.any():
                continue
            c, d = centers[keep], dist[keep] * s
            boxes.append(np.concatenate([c - d[:, :2], c + d[:, 2:]], axis=1))
            kpss.append(c[:, None, :] + kd[keep].reshape(-1, 5, 2) * s)
            scores.append(score[keep])
        if scores:
            b, sc, kp = np.concatenate(boxes), np.concatenate(scores), np.concatenate(kpss)
            keep = nms(b, sc, self.config.nms_threshold, self.config.top_k)
            b, sc, kp = b[keep] / lb.scale, sc[keep], kp[keep] / lb.scale
            dets = clip_and_build(b, sc, kp, fw, fh, self.keypoint_names)
        else:
            dets = []
        t3 = time.perf_counter()
        return dets, StageTimings((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
