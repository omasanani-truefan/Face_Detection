"""YuNet (OpenCV Zoo, 2023mar).

Two backends over the same weights:

* ``YuNetORT`` -- ONNX Runtime with our own decoder. Stage timings are separable, and
  ORT is also the runtime the browser build uses (ORT Web), so this is the variant the
  cross-model comparison uses.
* ``YuNetOpenCV`` -- ``cv2.FaceDetectorYN``, the official path. Its pre/post-processing
  is fused inside ``detect()``, so it reports end-to-end time only.

The head is anchor-free: one prior per cell at strides 8/16/32. Decoding mirrors
OpenCV's ``modules/objdetect/src/face_detect.cpp``: input is raw BGR (no mean/std),
``score = sqrt(clamp(cls) * clamp(obj))``, centre offsets and log-size are in units of
the stride.
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


class YuNetORT(FaceDetector):
    name = "yunet"
    runtime = "onnxruntime-cpu"
    keypoint_names = KEYPOINTS_5

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.session = make_session(config.model_path, config.num_threads)
        self.input_name = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].type != "tensor(float)":
            raise ValueError("YuNet expects a float input tensor")
        if "int8" in config.model_path:
            self.precision = "int8"
        names = [o.name for o in self.session.get_outputs()]
        self._order = [names.index(f"{kind}_{s}") for s in STRIDES for kind in ("cls", "obj", "bbox", "kps")]
        self._prior_cache: dict[tuple[int, int], list[np.ndarray]] = {}

    def _priors(self, w: int, h: int) -> list[np.ndarray]:
        key = (w, h)
        if key not in self._prior_cache:
            grids = []
            for s in STRIDES:
                rows, cols = h // s, w // s
                c, r = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
                grids.append(np.stack([c.ravel(), r.ravel()], axis=1))
            self._prior_cache[key] = grids
        return self._prior_cache[key]

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], StageTimings]:
        fh, fw = image_bgr.shape[:2]
        t0 = time.perf_counter()
        long_side = self.config.input_long_side or max(fh, fw)
        padded, lb = letterbox(image_bgr, long_side)
        # YuNet takes raw BGR 0..255: blobFromImage only does the HWC->NCHW float copy.
        blob = cv2.dnn.blobFromImage(padded)
        t1 = time.perf_counter()
        outs = self.session.run(None, {self.input_name: blob})
        t2 = time.perf_counter()

        thr = self.config.score_threshold
        boxes, scores, kpss = [], [], []
        for si, (s, grid) in enumerate(zip(STRIDES, self._priors(lb.input_w, lb.input_h))):
            cls, obj, bbox, kps = (outs[self._order[si * 4 + k]][0] for k in range(4))
            score = np.sqrt(np.clip(cls[:, 0], 0, 1) * np.clip(obj[:, 0], 0, 1))
            keep = score >= thr
            if not keep.any():
                continue
            g, bb, kp = grid[keep], bbox[keep], kps[keep]
            cxcy = (g + bb[:, :2]) * s
            wh = np.exp(bb[:, 2:4]) * s
            boxes.append(np.concatenate([cxcy - wh / 2, cxcy + wh / 2], axis=1))
            kpss.append((kp.reshape(-1, 5, 2) + g[:, None, :]) * s)
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


class YuNetOpenCV(FaceDetector):
    name = "yunet_cv"
    runtime = f"opencv-dnn {cv2.__version__}"
    keypoint_names = KEYPOINTS_5

    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        if config.num_threads:
            cv2.setNumThreads(config.num_threads)
        self.det = cv2.FaceDetectorYN.create(
            config.model_path, "", (320, 320), config.score_threshold, config.nms_threshold, config.top_k
        )
        self._size: tuple[int, int] | None = None

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], StageTimings]:
        fh, fw = image_bgr.shape[:2]
        t0 = time.perf_counter()
        long_side = self.config.input_long_side or max(fh, fw)
        scale = long_side / max(fh, fw)
        img = image_bgr
        if scale != 1:
            img = cv2.resize(image_bgr, (round(fw * scale), round(fh * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        size = (img.shape[1], img.shape[0])
        if size != self._size:
            self.det.setInputSize(size)
            self._size = size
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            dets = []
        else:
            boxes = faces[:, :4].astype(np.float64).copy()
            boxes[:, 2:] += boxes[:, :2]
            kps = faces[:, 4:14].reshape(-1, 5, 2) / scale
            dets = clip_and_build(boxes / scale, faces[:, 14], kps, fw, fh, self.keypoint_names)
        t1 = time.perf_counter()
        # FaceDetectorYN does its own blob/pad/decode/NMS internally: report as fused.
        return dets, StageTimings(0.0, (t1 - t0) * 1e3, 0.0, fused=True)
