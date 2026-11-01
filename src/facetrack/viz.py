"""Overlay drawing for the demo. Pure function of a ``PipelineResult``: no state."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from facetrack.pipeline import PipelineResult
from facetrack.tracking.base import TrackState


def track_color(track_id: int) -> tuple[int, int, int]:
    # Golden-ratio hue walk: neighbouring IDs get clearly different colours.
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _dashed_rect(img: np.ndarray, p1, p2, color, thickness: int = 1, dash: int = 6) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def _label(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw(result: PipelineResult, hud: dict[str, str] | None = None, show_raw: bool = False,
         show_tentative: bool = False) -> np.ndarray:
    img = result.frame.image.copy()
    h, w = img.shape[:2]
    scale = max(0.4, min(w, h) / 1100)
    thick = max(1, round(min(w, h) / 400))

    if show_raw:
        for d in result.detections:
            x, y, bw, bh = (int(round(v)) for v in d.bbox)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (160, 160, 160), 1)

    for t in result.tracks.tracks:
        if t.state is TrackState.TENTATIVE and not show_tentative:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in t.bbox_xyxy)
        color = track_color(t.track_id)
        if t.state is TrackState.LOST or (t.state is TrackState.CONFIRMED and not t.matched):
            _dashed_rect(img, (x1, y1), (x2, y2), color, thick)
        elif t.state is TrackState.TENTATIVE:
            _dashed_rect(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick + 1)
            if t.keypoints:
                for kx, ky in t.keypoints:
                    cv2.circle(img, (int(kx), int(ky)), max(1, thick), color, -1)
        _label(img, f"ID {t.track_id} {t.confidence:.2f} {t.state.value}", (x1, y1 - 3), color, scale)

    if hud:
        lines = [f"{k}: {v}" for k, v in hud.items()]
        pad = int(8 * scale / 0.5)
        lh = int(22 * scale / 0.5)
        box_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for s in lines) + 2 * pad
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, lh * len(lines) + pad), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        for i, s in enumerate(lines):
            cv2.putText(img, s, (pad, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img
