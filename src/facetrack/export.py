"""Result exporters: JSONL (everything) and MOTChallenge text (tracks, for evaluators)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from facetrack.pipeline import PipelineResult
from facetrack.tracking.base import TrackEvent


class JsonlExporter:
    """One JSON object per line: a header, then one record per frame, then a footer.

    Lifecycle events are inside each frame record (``events``), so a consumer replaying
    the file sees exactly what a live subscriber saw, in the same order.
    """

    def __init__(self, path: str | Path, header: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f: IO[str] = open(self.path, "w", encoding="utf-8")
        self._write({"type": "header", **header})

    def _write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def add(self, result: PipelineResult) -> None:
        self._write({"type": "frame", **result.to_dict()})

    def close(self, final_events: list[TrackEvent] | None = None, summary: dict | None = None) -> None:
        if self._f.closed:
            return
        self._write({"type": "footer", "events": [e.to_dict() for e in final_events or []], "summary": summary or {}})
        self._f.close()


class MotExporter:
    """MOTChallenge format: ``frame,id,x,y,w,h,conf,-1,-1,-1`` (1-based frame numbers).

    Writes the tracks a viewer would see: confirmed and matched on that frame.
    """

    def __init__(self, path: str | Path, include_lost: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w", encoding="utf-8")
        self.include_lost = include_lost

    def add(self, result: PipelineResult) -> None:
        for t in result.tracks.visible(self.include_lost):
            x, y, w, h = t.bbox_xywh
            self._f.write(f"{result.frame.frame_id + 1},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.confidence:.4f},-1,-1,-1\n")

    def close(self) -> None:
        self._f.close()
