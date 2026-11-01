"""Aspect-preserving resize + padding shared by the grid-based detectors."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class LetterboxInfo:
    scale: float  # model pixels per original pixel
    input_w: int  # padded tensor width fed to the model
    input_h: int


def letterbox(image: np.ndarray, long_side: int, stride: int = 32) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize so the longer side equals ``long_side``, then zero-pad right/bottom to a
    multiple of ``stride``.

    Padding only right/bottom keeps the inverse mapping a pure scale (no offset), and
    padding to the stride rather than to a square avoids spending compute on a 640x640
    tensor when the frame is 640x360.
    """
    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    pad_w = -(-new_w // stride) * stride
    pad_h = -(-new_h // stride) * stride
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp) if scale != 1 else image
    if (pad_w, pad_h) != (new_w, new_h):
        resized = cv2.copyMakeBorder(resized, 0, pad_h - new_h, 0, pad_w - new_w, cv2.BORDER_CONSTANT, value=0)
    return resized, LetterboxInfo(scale=scale, input_w=pad_w, input_h=pad_h)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float, top_k: int = 5000) -> np.ndarray:
    """Indices kept by greedy NMS, highest score first. Delegates to OpenCV's C++ NMS."""
    if len(scores) == 0:
        return np.empty(0, dtype=np.int64)
    xywh = boxes_xyxy.copy()
    xywh[:, 2:] -= xywh[:, :2]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=iou_threshold, top_k=top_k)
    return np.asarray(keep, dtype=np.int64).reshape(-1)
