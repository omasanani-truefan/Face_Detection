"""Frame sources: the only place that knows where frames come from.

``VideoFileSource`` and ``CameraSource`` both yield ``Frame`` objects (BGR image +
millisecond timestamp), so the detector and tracker cannot tell them apart.

Timestamps:
* video files use the container's presentation time (``CAP_PROP_POS_MSEC``), so
  variable-frame-rate files keep their real timing; frame_index / fps is only a fallback
  when the backend reports no time.
* cameras use a monotonic clock sampled right after each frame is grabbed.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from facetrack.types import Frame

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class SourceError(RuntimeError):
    pass


@dataclass
class SourceInfo:
    kind: str
    uri: str
    width: int
    height: int
    fps: float | None  # nominal rate reported by the container/driver; may be absent
    frame_count: int | None = None
    duration_ms: float | None = None
    codec: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class FrameSource(ABC):
    info: SourceInfo
    source_id: str

    @abstractmethod
    def read(self) -> Frame | None:
        """Next frame, or None at end of stream."""

    @abstractmethod
    def release(self) -> None:
        """Free the device / file handle. Safe to call more than once."""

    def __iter__(self) -> Iterator[Frame]:
        while (f := self.read()) is not None:
            yield f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def _fourcc(cap: cv2.VideoCapture) -> str | None:
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    s = "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")
    return s or None


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path, source_id: str | None = None):
        # An http(s) URL is read as a network stream (see download.resolve_stream_url).
        self.is_url = str(path).startswith(("http://", "https://"))
        self.path = path if self.is_url else Path(path)
        if not self.is_url:
            if not self.path.is_file():
                raise SourceError(f"video not found: {self.path}")
            if self.path.suffix.lower() not in VIDEO_EXTENSIONS:
                log.warning("unrecognised video extension %s; trying anyway", self.path.suffix)
        self.source_id = source_id or ("stream" if self.is_url else self.path.stem)
        self._open()
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = fps if fps and np.isfinite(fps) and fps > 0 else None
        self.info = SourceInfo(
            "video", "stream" if self.is_url else str(self.path), int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps, n if n > 0 else None,
            n * 1000.0 / fps if fps and n > 0 else None, _fourcc(self._cap),
        )
        self.invalid_frames = 0

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise SourceError(f"cannot decode {self.path} (unsupported container or codec?)")
        self._index = 0
        self._last_ts = -1.0

    def read(self) -> Frame | None:
        while True:
            ok, img = self._cap.read()
            if not ok:
                return None
            if img is None or img.size == 0 or img.ndim != 3:
                self.invalid_frames += 1
                self._index += 1
                continue
            break
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        # Some backends report 0 / non-increasing times; fall back to the nominal rate.
        if not np.isfinite(ts) or ts <= self._last_ts:
            ts = max(self._index * 1000.0 / (self.info.fps or 30.0), self._last_ts + 1e-3)
        self._last_ts = ts
        frame = Frame(img, self._index, float(ts), self.source_id)
        self._index += 1
        return frame

    def restart(self) -> None:
        """Rewind to the first frame (reopen: seeking is unreliable for some codecs)."""
        self._cap.release()
        self._open()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class CameraSource(FrameSource):
    """Live camera with automatic reconnection after transient failures (USB hiccup,
    another app grabbing the device, laptop lid closed and reopened)."""

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None,
                 fps: float | None = None, source_id: str | None = None,
                 max_consecutive_failures: int = 10, reconnect_attempts: int = 5, backend: int | None = None):
        self.index = index
        self.source_id = source_id or f"camera{index}"
        self._req = (width, height, fps)
        self._backend = backend if backend is not None else (cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY)
        self._max_fail = max_consecutive_failures
        self._reconnects = reconnect_attempts
        self._cap: cv2.VideoCapture | None = None
        self._open()
        self._t0 = time.monotonic()
        self._index = 0
        self._failures = 0

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.index, self._backend)
        if not cap.isOpened():
            cap.release()
            raise SourceError(f"camera {self.index} unavailable (not connected, in use, or permission denied)")
        w, h, fps = self._req
        if w:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        # Keep latency low: we want the newest frame, not a queue of stale ones.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        rep_fps = cap.get(cv2.CAP_PROP_FPS)
        self.info = SourceInfo("camera", f"camera:{self.index}", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), rep_fps if rep_fps > 0 else None,
                               codec=_fourcc(cap))

    def _reconnect(self) -> bool:
        for attempt in range(1, self._reconnects + 1):
            log.warning("camera %s: reconnect attempt %d/%d", self.index, attempt, self._reconnects)
            if self._cap is not None:
                self._cap.release()
            time.sleep(min(0.25 * 2**attempt, 4.0))
            try:
                self._open()
                return True
            except SourceError:
                continue
        return False

    def read(self) -> Frame | None:
        while True:
            if self._cap is None:
                return None
            ok, img = self._cap.read()
            if ok and img is not None and img.size:
                self._failures = 0
                ts = (time.monotonic() - self._t0) * 1000.0
                f = Frame(img, self._index, ts, self.source_id)
                self._index += 1
                return f
            self._failures += 1
            if self._failures >= self._max_fail:
                self._failures = 0
                if not self._reconnect():
                    raise SourceError(f"camera {self.index} stopped delivering frames and could not be reopened")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def list_cameras(max_index: int = 5) -> list[int]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY)
        if cap.isOpened() and cap.read()[0]:
            found.append(i)
        cap.release()
    return found


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"
