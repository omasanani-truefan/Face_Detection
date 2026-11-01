from __future__ import annotations

from facetrack.types import Detection


def det(x: float, y: float, w: float = 60, h: float = 70, conf: float = 0.9) -> Detection:
    return Detection((float(x), float(y), float(w), float(h)), conf)


def run(tracker, frames: list[list[Detection]], fps: float = 30.0, start_ms: float = 0.0):
    """Feed one detection list per frame at a fixed rate; return (outputs, all events)."""
    outs, events = [], []
    for i, dets in enumerate(frames):
        out = tracker.update(dets, start_ms + i * 1000.0 / fps, i)
        outs.append(out)
        events += out.events
    return outs, events


def visible_ids(out) -> set[int]:
    return {t.track_id for t in out.visible()}
