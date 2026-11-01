"""Tracker lifecycle and association edge cases, run against every tracker."""

from __future__ import annotations

import numpy as np
import pytest

from facetrack.tracking import EventType, TrackerConfig, TrackState, create_tracker
from tests.helpers import det, run, visible_ids

ALL = ["iou", "sort", "bytetrack"]
MOTION = ["sort", "bytetrack"]


def make(name, **kw):
    cfg = dict(min_hits=3, max_lost_ms=1000, tentative_timeout_ms=300)
    cfg.update(kw)
    return create_tracker(name, TrackerConfig(**cfg))


def kinds(events, kind):
    return [e for e in events if e.type is kind]


@pytest.mark.parametrize("name", ALL)
def test_no_faces(name):
    outs, events = run(make(name), [[] for _ in range(30)])
    assert all(not o.tracks for o in outs) and not events


@pytest.mark.parametrize("name", ALL)
def test_one_face_confirmed_after_min_hits_and_keeps_id(name):
    outs, events = run(make(name), [[det(100, 100)] for _ in range(60)])
    assert not visible_ids(outs[0]) and not visible_ids(outs[1])  # still tentative
    assert visible_ids(outs[2]) == {1}
    assert all(visible_ids(o) == {1} for o in outs[2:])
    assert len(kinds(events, EventType.STARTED)) == 1


@pytest.mark.parametrize("name", ALL)
def test_multiple_faces_get_distinct_ids(name):
    frames = [[det(50, 50), det(250, 60), det(450, 80)] for _ in range(20)]
    outs, _ = run(make(name), frames)
    assert visible_ids(outs[-1]) == {1, 2, 3}


@pytest.mark.parametrize("name", ALL)
def test_new_face_entering_gets_new_id(name):
    frames = [[det(50, 50)] for _ in range(10)] + [[det(50, 50), det(400, 60)] for _ in range(10)]
    outs, events = run(make(name), frames)
    assert visible_ids(outs[-1]) == {1, 2}
    assert [e.track_id for e in kinds(events, EventType.STARTED)] == [1, 2]


@pytest.mark.parametrize("name", ALL)
def test_moving_face_keeps_id(name):
    frames = [[det(50 + 6 * i, 100 + 2 * i)] for i in range(60)]
    outs, _ = run(make(name), frames)
    assert all(visible_ids(o) == {1} for o in outs[2:])


@pytest.mark.parametrize("name", ALL)
def test_single_missed_detection_does_not_terminate(name):
    frames = [[det(100, 100)] for _ in range(10)] + [[]] + [[det(100, 100)] for _ in range(10)]
    outs, events = run(make(name), frames)
    assert outs[10].tracks[0].state is TrackState.LOST
    assert visible_ids(outs[-1]) == {1}
    assert len(kinds(events, EventType.LOST)) == 1 and len(kinds(events, EventType.RECOVERED)) == 1
    assert not kinds(events, EventType.ENDED)


@pytest.mark.parametrize("name", ALL)
def test_temporary_occlusion_within_grace_keeps_id(name):
    # 0.5 s gap at 30 fps, grace period 1 s.
    frames = [[det(100, 100)] for _ in range(15)] + [[] for _ in range(15)] + [[det(104, 100)] for _ in range(15)]
    outs, events = run(make(name), frames)
    assert visible_ids(outs[-1]) == {1}
    assert not kinds(events, EventType.ENDED)


@pytest.mark.parametrize("name", ALL)
def test_face_leaving_terminates_after_grace_period_by_time(name):
    frames = [[det(100, 100)] for _ in range(15)] + [[] for _ in range(60)]
    outs, events = run(make(name, max_lost_ms=1000), frames)
    ended = kinds(events, EventType.ENDED)
    assert len(ended) == 1
    last_seen = 14 * 1000 / 30
    # Ends on the first update more than 1 s after the last observation.
    assert 1000 < ended[0].timestamp_ms - last_seen <= 1000 + 1000 / 30 + 1e-6
    assert not outs[-1].tracks


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("fps", [10, 30, 60])
def test_termination_delay_is_frame_rate_independent(name, fps):
    n_seen = int(fps)  # 1 s of presence
    frames = [[det(100, 100)] for _ in range(n_seen)] + [[] for _ in range(int(3 * fps))]
    _, events = run(make(name, max_lost_ms=800), frames, fps=fps)
    ended = kinds(events, EventType.ENDED)
    assert len(ended) == 1
    delay = ended[0].timestamp_ms - (n_seen - 1) * 1000 / fps
    assert 800 < delay <= 800 + 1000 / fps + 1e-6


@pytest.mark.parametrize("name", ALL)
def test_reentry_after_termination_gets_new_id(name):
    frames = [[det(100, 100)] for _ in range(15)] + [[] for _ in range(45)] + [[det(100, 100)] for _ in range(15)]
    outs, events = run(make(name, max_lost_ms=1000), frames)
    assert visible_ids(outs[-1]) == {2}
    assert [e.track_id for e in kinds(events, EventType.ENDED)] == [1]


@pytest.mark.parametrize("name", MOTION)
def test_crossing_faces_keep_ids_with_motion_model(name):
    # Two faces move towards each other and pass with partial overlap.
    frames = [[det(40 + 8 * i, 100), det(500 - 8 * i, 130)] for i in range(60)]
    outs, _ = run(make(name), frames)
    first = {t.track_id: t.bbox_xyxy[0] for t in outs[5].visible()}
    last = {t.track_id: t.bbox_xyxy[0] for t in outs[-1].visible()}
    assert set(first) == set(last) == {1, 2}
    # Whichever ID started on the left must end on the right.
    left_id = min(first, key=first.get)
    assert last[left_id] == max(last.values())


@pytest.mark.parametrize("name", ALL)
def test_no_duplicate_active_ids_under_random_detections(name):
    rng = np.random.default_rng(0)
    trk = make(name)
    for i in range(300):
        n = rng.integers(0, 6)
        dets = [det(*rng.uniform(0, 600, 2), conf=float(rng.uniform(0.3, 1))) for _ in range(n)]
        out = trk.update(dets, i * 33.3, i)
        ids = [t.track_id for t in out.tracks]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", ALL)
def test_frame_skipping_predict_does_not_change_lifecycle(name):
    # Stationary face: this checks lifecycle bookkeeping, not motion prediction.
    trk = make(name)
    for i in range(10):
        trk.update([det(100, 100)], i * 33.3, i)
    for i in range(10, 40):  # 1 s with no detector runs at all
        out = trk.predict(i * 33.3, i)
        assert out.tracks[0].state is TrackState.CONFIRMED and not out.events
    out = trk.update([det(100, 100)], 40 * 33.3, 40)
    assert visible_ids(out) == {1}


@pytest.mark.parametrize("name", MOTION)
def test_lastobs_recovers_stationary_face_after_partial_occlusion(name):
    """The measured failure: an occluder slides over a still face, the detected box
    shrinks towards one side, the Kalman filter learns a spurious velocity, and during
    the gap the prediction drifts away. With observation-centric recovery the face is
    re-matched on its last observed box and keeps its ID."""
    frames = [[det(300, 100, 80, 96)] for _ in range(20)]
    frames += [[det(300, 100, 80 - 6 * k, 96)] for k in range(1, 9)]  # covered from the right
    frames += [[] for _ in range(18)]  # 0.6 s fully hidden (< 1 s grace)
    frames += [[det(300, 100, 80, 96)] for _ in range(10)]
    outs, events = run(make(name, lost_match_last_observation=True), frames)
    assert visible_ids(outs[-1]) == {1}
    assert any(e.type is EventType.RECOVERED for e in events)


def test_bytetrack_low_score_detections_hold_track():
    frames = [[det(100, 100, conf=0.9)] for _ in range(10)] + [[det(100, 100, conf=0.35)] for _ in range(10)]
    cfg = dict(det_threshold=0.6, low_det_threshold=0.3)
    outs_b, ev_b = run(make("bytetrack", **cfg), frames)
    outs_s, ev_s = run(make("sort", **cfg), frames)
    assert visible_ids(outs_b[-1]) == {1} and not kinds(ev_b, EventType.LOST)
    assert not visible_ids(outs_s[-1]) and kinds(ev_s, EventType.LOST)


@pytest.mark.parametrize("name", ALL)
def test_low_score_detections_never_start_tracks(name):
    outs, events = run(make(name, det_threshold=0.6, low_det_threshold=0.3), [[det(100, 100, conf=0.4)] for _ in range(20)])
    assert not events and not outs[-1].tracks


@pytest.mark.parametrize("name", ALL)
def test_backwards_timestamps_rejected_and_reset_keeps_ids_increasing(name):
    trk = make(name)
    run(trk, [[det(100, 100)] for _ in range(5)])
    with pytest.raises(ValueError):
        trk.update([det(100, 100)], -5.0, 99)
    trk.reset()
    outs, _ = run(trk, [[det(100, 100)] for _ in range(5)])
    assert visible_ids(outs[-1]) == {2}


@pytest.mark.parametrize("name", ALL)
def test_irregular_frame_intervals(name):
    rng = np.random.default_rng(1)
    trk = make(name)
    t = 0.0
    for i in range(100):
        t += float(rng.uniform(15, 120))  # 8-66 fps jitter, as on a busy phone
        out = trk.update([det(100 + 0.1 * t, 100)], t, i)
    assert visible_ids(out) == {1}


@pytest.mark.parametrize("name", ALL)
def test_finish_ends_announced_tracks_only(name):
    trk = make(name)
    run(trk, [[det(100, 100)] for _ in range(5)] + [[det(100, 100), det(400, 100)]])
    events = trk.finish(1000.0)
    assert [e.track_id for e in events] == [1]  # track 2 was still tentative
