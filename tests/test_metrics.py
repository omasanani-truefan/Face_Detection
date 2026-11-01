"""Tracking metrics: hand-checked toy cases plus a cross-check against py-motmetrics."""

from __future__ import annotations

import numpy as np
import pytest

from facetrack.metrics.tracking import evaluate_sequence


def box(x, y, s=50):
    return np.array([x, y, x + s, y + s], dtype=float)


def seq(n=20):
    gt = {f: [(1, box(10 + f, 10), False), (2, box(300 - f, 200), False)] for f in range(n)}
    return gt, list(range(n))


def test_perfect_tracking():
    gt, frames = seq()
    pred = {f: [(7, b), (9, b2)] for f, ((_, b, _), (_, b2, _)) in gt.items()}
    m = evaluate_sequence(gt, pred, frames)
    assert m["MOTA"] == pytest.approx(1) and m["IDF1"] == pytest.approx(1) and m["HOTA"] == pytest.approx(1)
    assert m["IDSW"] == 0 and m["Frag"] == 0


def test_id_switch_counted_once_and_lowers_idf1():
    gt, frames = seq()
    pred = {f: [(7 if f < 10 else 8, gt[f][0][1]), (9, gt[f][1][1])] for f in frames}
    m = evaluate_sequence(gt, pred, frames)
    assert m["IDSW"] == 1 and m["MOTA"] == pytest.approx(1 - 1 / 40)
    assert m["IDF1"] < 1 and m["HOTA"] < 1


def test_gap_counts_fragmentation_not_switch():
    gt, frames = seq()
    pred = {f: ([(7, gt[f][0][1])] if not 5 <= f < 8 else []) + [(9, gt[f][1][1])] for f in frames}
    m = evaluate_sequence(gt, pred, frames)
    assert m["Frag"] == 1 and m["IDSW"] == 0 and m["FN"] == 3


def test_predictions_on_ignore_regions_are_not_false_positives():
    gt = {0: [(1, box(10, 10), False), (2, box(200, 10), True)]}
    pred = {0: [(1, box(10, 10)), (2, box(200, 10))]}
    m = evaluate_sequence(gt, pred, [0])
    assert m["FP"] == 0 and m["FN"] == 0


def test_lifecycle_timings():
    gt = {f: [(1, box(10, 10), False)] for f in range(10)}
    pred = {f: [(5, box(10, 10))] for f in range(2, 10)}
    ts = {f: f * 100.0 for f in range(20)}
    m = evaluate_sequence(gt, {**pred, **{f: [] for f in range(10, 20)}}, list(range(20)), ts, {5: 1900.0})
    assert m["confirmation_delay_ms_mean"] == pytest.approx(200)
    assert m["termination_delay_ms_mean"] == pytest.approx(1000)
    assert m["missed_tracks"] == 0 and m["false_tracks"] == 0


def test_matches_motmetrics_on_random_sequences():
    mm = pytest.importorskip("motmetrics")
    rng = np.random.default_rng(3)
    for trial in range(5):
        frames = list(range(60))
        gt, pred = {}, {}
        for f in frames:
            g = []
            for gid in range(4):
                if rng.random() < 0.9:
                    g.append((gid, box(80 * gid + f * 0.5, 50 + 10 * gid), False))
            gt[f] = g
            p = []
            for gid, b, _ in g:
                if rng.random() < 0.85:
                    tid = gid if rng.random() > 0.05 else gid + 10  # occasional identity flip
                    p.append((tid, b + rng.normal(0, 4, 4)))
            if rng.random() < 0.2:
                p.append((99, box(rng.uniform(0, 400), rng.uniform(0, 300))))
            pred[f] = p
        ours = evaluate_sequence(gt, pred, frames)
        acc = mm.MOTAccumulator(auto_id=True)
        for f in frames:
            # motmetrics 1.4's own iou_matrix calls np.asfarray (gone in NumPy 2), so the
            # distance matrix is built here with an independent IoU; matching and metric
            # bookkeeping are still motmetrics'.
            d = np.full((len(gt[f]), len(pred[f])), np.nan)
            for i, (_, g, _) in enumerate(gt[f]):
                for j, (_, p) in enumerate(pred[f]):
                    iw = max(0.0, min(g[2], p[2]) - max(g[0], p[0]))
                    ih = max(0.0, min(g[3], p[3]) - max(g[1], p[1]))
                    inter = iw * ih
                    union = (g[2] - g[0]) * (g[3] - g[1]) + (p[2] - p[0]) * (p[3] - p[1]) - inter
                    if inter / union >= 0.5:
                        d[i, j] = 1 - inter / union
            acc.update([g for g, _, _ in gt[f]], [t for t, _ in pred[f]], d)
        s = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches"], name="x")
        assert ours["MOTA"] == pytest.approx(float(s["mota"].iloc[0]), abs=1e-6), trial
        assert ours["IDF1"] == pytest.approx(float(s["idf1"].iloc[0]), abs=1e-6), trial
        assert ours["IDSW"] == int(s["num_switches"].iloc[0]), trial
