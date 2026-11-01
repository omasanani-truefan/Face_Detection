"""Frame sources and the source-agnostic pipeline."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from facetrack.pipeline import FacePipeline
from facetrack.sources import CameraSource, SourceError, VideoFileSource
from facetrack.tracking import TrackerConfig, create_tracker
from facetrack.types import Detection, Frame, StageTimings


def write_video(path, n=30, fps=15.0, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), i * 8 % 255, np.uint8)
        w.write(img)
    w.release()


@pytest.mark.parametrize("fps", [15.0, 25.0, 60.0])
def test_video_timestamps_follow_container_rate(tmp_path, fps):
    p = tmp_path / f"v{int(fps)}.avi"
    write_video(p, n=20, fps=fps)
    with VideoFileSource(p) as src:
        assert src.info.fps == pytest.approx(fps, rel=1e-3)
        assert (src.info.width, src.info.height) == (160, 120)
        frames = list(src)
    assert len(frames) == 20
    ts = np.array([f.timestamp_ms for f in frames])
    assert np.all(np.diff(ts) > 0)
    assert np.diff(ts).mean() == pytest.approx(1000 / fps, rel=0.02)


def test_video_restart_rewinds(tmp_path):
    p = tmp_path / "v.avi"
    write_video(p, n=10)
    src = VideoFileSource(p)
    first = [src.read().frame_id for _ in range(5)]
    src.restart()
    assert src.read().frame_id == 0 and first == [0, 1, 2, 3, 4]
    src.release()
    src.release()  # idempotent


def test_missing_and_undecodable_video(tmp_path):
    with pytest.raises(SourceError):
        VideoFileSource(tmp_path / "nope.mp4")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(SourceError):
        VideoFileSource(bad)


def test_unavailable_camera_raises_cleanly():
    with pytest.raises(SourceError):
        CameraSource(index=57)


class FlakyCapture:
    """Stands in for cv2.VideoCapture: delivers frames, drops out, comes back on reopen."""

    script: list[str] = []  # per-open behaviour: "ok" / "fail_after:<n>" / "dead"
    opens = 0

    def __init__(self, *_a):
        FlakyCapture.opens += 1
        self.mode = FlakyCapture.script.pop(0) if FlakyCapture.script else "ok"
        self.n = 0

    def isOpened(self):
        return self.mode != "dead"

    def set(self, *_a):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_FPS: 30}.get(prop, 0)

    def read(self):
        self.n += 1
        if self.mode.startswith("fail_after:") and self.n > int(self.mode.split(":")[1]):
            return False, None
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        pass


def test_camera_interruption_reconnects_and_keeps_monotonic_time(monkeypatch):
    import facetrack.sources as sources

    FlakyCapture.script, FlakyCapture.opens = ["fail_after:5", "ok"], 0
    monkeypatch.setattr(sources.cv2, "VideoCapture", FlakyCapture)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    cam = CameraSource(0, max_consecutive_failures=3, reconnect_attempts=2)
    frames = [cam.read() for _ in range(12)]
    assert all(f is not None for f in frames) and FlakyCapture.opens == 2
    ts = [f.timestamp_ms for f in frames]
    assert all(b >= a for a, b in zip(ts, ts[1:]))
    assert [f.frame_id for f in frames] == list(range(12))
    cam.release()


def test_camera_that_never_returns_raises_instead_of_hanging(monkeypatch):
    import facetrack.sources as sources

    FlakyCapture.script, FlakyCapture.opens = ["fail_after:2", "dead", "dead"], 0
    monkeypatch.setattr(sources.cv2, "VideoCapture", FlakyCapture)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    cam = CameraSource(0, max_consecutive_failures=3, reconnect_attempts=2)
    assert cam.read() is not None and cam.read() is not None
    with pytest.raises(SourceError, match="could not be reopened"):
        cam.read()


class FakeDetector:
    """Returns a scripted list of detections per call; records what it was given."""

    name = "fake"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def detect(self, image):
        self.calls.append(image.shape)
        return self.script.pop(0) if self.script else [], StageTimings(0, 0, 0)


def frames(n, fps=30.0, image=None):
    img = np.zeros((48, 64, 3), np.uint8) if image is None else image
    return [Frame(img, i, i * 1000 / fps, "test") for i in range(n)]


def test_pipeline_detect_every_n_runs_detector_on_schedule():
    d = Detection((10, 10, 20, 20), 0.9)
    det = FakeDetector([[d]] * 100)
    pipe = FacePipeline(det, create_tracker("sort", TrackerConfig(min_hits=1)), detect_every=3)
    results = [pipe.process(f) for f in frames(9)]
    assert [r.detector_ran for r in results] == [True, False, False] * 3
    assert len(det.calls) == 3
    assert all(r.tracks.tracks and r.tracks.tracks[0].track_id == 1 for r in results)


def test_pipeline_coerces_gray_and_skips_invalid_frames():
    det = FakeDetector([])
    pipe = FacePipeline(det, create_tracker("iou"))
    pipe.process(Frame(np.zeros((48, 64), np.uint8), 0, 0.0))
    pipe.process(Frame(np.zeros((48, 64, 4), np.uint8), 1, 33.0))
    r = pipe.process(Frame(np.zeros((0, 0, 3), np.uint8), 2, 66.0))
    assert det.calls == [(48, 64, 3), (48, 64, 3)]
    assert not r.detector_ran


def test_same_pipeline_code_for_video_and_camera_shaped_sources(tmp_path):
    # A "camera" here is anything yielding Frames with monotonic ms timestamps; the
    # pipeline must accept both without knowing which it got.
    p = tmp_path / "v.avi"
    write_video(p, n=8)
    det = FakeDetector([[Detection((5, 5, 30, 30), 0.9)]] * 100)
    pipe = FacePipeline(det, create_tracker("bytetrack", TrackerConfig(min_hits=2)))
    with VideoFileSource(p) as src:
        video_ids = {t.track_id for f in src for t in pipe.process(f).tracks.visible()}
    pipe.reset()
    cam_ids = {t.track_id for f in frames(8) for t in pipe.process(f).tracks.visible()}
    assert video_ids == {1} and cam_ids == {2}


class Recorder:
    def __init__(self):
        self.obs, self.events = [], []

    def on_observation(self, frame, track, detection):
        self.obs.append((frame.frame_id, track.track_id))

    def on_event(self, event):
        self.events.append(event.type.value)


def test_observers_see_observations_and_full_lifecycle():
    d = Detection((10, 10, 20, 20), 0.9)
    det = FakeDetector([[d]] * 5 + [[]] * 60)
    rec = Recorder()
    pipe = FacePipeline(det, create_tracker("sort", TrackerConfig(min_hits=2, max_lost_ms=500)), observers=[rec])
    for f in frames(40):
        pipe.process(f)
    assert rec.obs == [(i, 1) for i in range(1, 5)]
    assert rec.events[0] == "track_started" and "track_lost" in rec.events and rec.events[-1] == "track_ended"
