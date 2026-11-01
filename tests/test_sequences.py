"""Ground truth of the synthetic tracking sequences: the benchmark is only as good as these."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from facetrack.bench.sequences import load_actors, load_backgrounds, render_actor_sequence, scenarios

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "widerface" / "dev.json"
pytestmark = pytest.mark.skipif(not DEV.exists(), reason="run scripts/prepare_widerface.py")


@pytest.fixture(scope="module")
def assets():
    coco = json.loads(DEV.read_text())
    root = ROOT / coco["info"]["image_root"]
    rng = np.random.default_rng(0)
    return load_actors(coco, root, 6, rng), load_backgrounds(coco, root, 1, (640, 480), rng)[0]


def render(name, assets, fps=15.0):
    sc = next(s for s in scenarios() if s.name == name)
    actors, bg = assets
    return sc, list(render_actor_sequence(sc, actors, bg, fps, seed=1))


def test_boxes_inside_frame_and_timestamps_increase(assets):
    sc, frames = render("multi_enter", assets)
    W, H = sc.size
    ts = [t for _, t, _ in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    for img, _, faces in frames:
        assert img.shape == (H, W, 3)
        for f in faces:
            x, y, w, h = f["bbox"]
            assert x >= 0 and y >= 0 and x + w <= W + 1e-6 and y + h <= H + 1e-6 and w > 0 and h > 0


def test_reentry_is_new_gt_track_same_person(assets):
    _, frames = render("reentry", assets)
    by_track = {}
    for _, _, faces in frames:
        for f in faces:
            by_track.setdefault(f["track_gt_id"], set()).add(f["person_id"])
    persons = [next(iter(p)) for p in by_track.values()]
    assert len(by_track) == 3 and len(set(persons)) == 2  # two appearances share one person


def test_occluded_and_turned_faces_are_ignored(assets):
    _, frames = render("temporary_occlusion", assets)
    vis = [f for _, _, faces in frames for f in faces]
    assert any(f["ignore"] and f["visibility"] < 0.5 for f in vis)
    assert all(not f["ignore"] for f in vis if f["visibility"] >= 0.5 and f["bbox"][3] >= 10)
    _, frames = render("turn_away", assets)
    assert any(f["ignore"] for _, _, faces in frames for f in faces)


def test_deterministic(assets):
    _, a = render("crossing", assets)
    _, b = render("crossing", assets)
    assert all(np.array_equal(x[0], y[0]) and x[2] == y[2] for x, y in zip(a, b))
