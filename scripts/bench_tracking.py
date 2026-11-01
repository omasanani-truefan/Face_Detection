"""Tracking benchmark: IoU vs SORT vs ByteTrack on identical detector outputs.

    python scripts/bench_tracking.py                     # detect (cached) + track + report
    python scripts/bench_tracking.py --detectors yunet@640

Stage 1 decodes every sequence video through ``VideoFileSource`` (the real upload path)
and caches each detector's per-frame output with a low score floor, so all trackers see
byte-identical detections. Stage 2 tunes the tracker's detection threshold on the **dev**
sequences (macro HOTA) and reports **test** metrics, per sequence and per scenario.
Stage 3 (best detector only) measures detector-interval skipping and frame-rate effects.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.benchenv import environment, latency_stats  # noqa: E402
from facetrack.detectors import create_detector  # noqa: E402
from facetrack.metrics.tracking import evaluate_sequence  # noqa: E402
from facetrack.sources import VideoFileSource  # noqa: E402
from facetrack.tracking import EventType, TrackerConfig, create_tracker  # noqa: E402
from facetrack.types import Detection  # noqa: E402

SEQ_DIR = ROOT / "data" / "sequences"
DETECTORS = ["yunet@640", "scrfd_500m_kps@640", "mediapipe_full", "mediapipe_short", "yunet@native"]
TRACKERS = ["iou", "sort", "bytetrack", "sort+lastobs", "bytetrack+lastobs"]
# "+lastobs" = the published tracker plus observation-centric recovery (TrackerConfig.
# lost_match_last_observation), added after the baseline exposed a Kalman-drift failure.
VARIANTS = {"+lastobs": {"lost_match_last_observation": True}}
DET_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
SCORE_FLOOR = 0.2  # detector floor; ByteTrack's low bucket is [0.2, det_threshold)
BASE = dict(min_hits=3, max_lost_ms=1000.0, tentative_timeout_ms=300.0, iou_threshold=0.3, low_det_threshold=SCORE_FLOOR)


def parse_det(cfg: str) -> tuple[str, int]:
    name, _, size = cfg.partition("@")
    return name, (0 if size in ("", "native") else int(size))


# --------------------------------------------------------------------------- stage 1
def cached_detections(cfg: str, seq: dict, out_dir: Path) -> dict:
    path = out_dir / "detections" / cfg.replace("@", "_") / seq["split"] / f"{seq['name']}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    name, size = parse_det(cfg)
    det = create_detector(name, score_threshold=SCORE_FLOOR, input_long_side=size)
    frames, lat = [], []
    with VideoFileSource(SEQ_DIR / seq["video"]) as src:
        for fr in src:
            dets, t = det.detect(fr.image)
            lat.append(t.total_ms)
            frames.append({"frame_id": fr.frame_id, "timestamp_ms": fr.timestamp_ms,
                           "dets": [[*(round(v, 2) for v in d.bbox), round(d.confidence, 4)] for d in dets]})
    rec = {"detector": det.describe(), "sequence": seq["name"], "frames": frames, "latency": latency_stats(lat)}
    det.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(rec, f)
    return rec


# --------------------------------------------------------------------------- stage 2
def load_gt(seq: dict) -> dict:
    return json.loads((SEQ_DIR / seq["gt"]).read_text())


def run_tracker(tracker_name: str, cfg: dict, det_rec: dict, keep: list[int] | None = None, detect_every: int = 1):
    """Replay cached detections through a tracker. ``keep`` = frame indices delivered
    (frame-rate experiments); ``detect_every`` = detector interval (skipped frames call
    predict())."""
    base, plus, suffix = tracker_name.partition("+")
    trk = create_tracker(base, TrackerConfig(**{**cfg, **VARIANTS.get(plus + suffix, {})}))
    frames = det_rec["frames"]
    idx = keep if keep is not None else list(range(len(frames)))
    pred, ts, ends = {}, {}, {}
    lat = []
    for n, i in enumerate(idx):
        fr = frames[i]
        t0 = time.perf_counter()
        if n % detect_every == 0:
            dets = [Detection((d[0], d[1], d[2], d[3]), d[4]) for d in fr["dets"]]
            out = trk.update(dets, fr["timestamp_ms"], i)
        else:
            out = trk.predict(fr["timestamp_ms"], i)
        lat.append((time.perf_counter() - t0) * 1e3)
        pred[i] = [(t.track_id, t.bbox_xyxy.copy()) for t in out.visible()]
        ts[i] = fr["timestamp_ms"]
        for e in out.events:
            if e.type is EventType.ENDED:
                ends[e.track_id] = e.timestamp_ms
    for e in trk.finish(frames[idx[-1]]["timestamp_ms"]):
        ends.setdefault(e.track_id, e.timestamp_ms)
    return pred, ts, ends, lat


def gt_frames(gt: dict):
    out = {}
    for fr in gt["frames"]:
        out[fr["frame_id"]] = [(f["track_gt_id"], np.array([f["bbox"][0], f["bbox"][1], f["bbox"][0] + f["bbox"][2],
                                                             f["bbox"][1] + f["bbox"][3]]), f["ignore"]) for f in fr["faces"]]
    return out


def reentry_reuse(gt: dict, pred: dict) -> int:
    """Count re-entries where the tracker re-used the earlier appearance's Track ID."""
    person_tracks = defaultdict(set)
    for fr in gt["frames"]:
        for f in fr["faces"]:
            person_tracks[f["person_id"]].add(f["track_gt_id"])
    from facetrack.tracking.matching import iou_matrix

    dominant = {}
    votes = defaultdict(lambda: defaultdict(int))
    for fr in gt["frames"]:
        p = pred.get(fr["frame_id"], [])
        g = [f for f in fr["faces"] if not f["ignore"]]
        if not p or not g:
            continue
        gb = np.array([[f["bbox"][0], f["bbox"][1], f["bbox"][0] + f["bbox"][2], f["bbox"][1] + f["bbox"][3]] for f in g])
        iou = iou_matrix(gb, np.array([b for _, b in p]))
        for gi, f in enumerate(g):
            j = int(np.argmax(iou[gi]))
            if iou[gi, j] >= 0.5:
                votes[f["track_gt_id"]][p[j][0]] += 1
    for g, v in votes.items():
        dominant[g] = max(v.items(), key=lambda kv: kv[1])[0]
    reused = 0
    for tracks in person_tracks.values():
        ids = [dominant[t] for t in sorted(tracks) if t in dominant]
        reused += len(ids) - len(set(ids))
    return reused


def eval_one(tracker: str, cfg: dict, det_rec: dict, gt: dict, keep=None, detect_every: int = 1) -> dict:
    pred, ts, ends, lat = run_tracker(tracker, cfg, det_rec, keep, detect_every)
    frames = sorted(ts)
    m = evaluate_sequence(gt_frames(gt), pred, frames, ts, ends)
    m["reentry_id_reused"] = reentry_reuse(gt, pred)
    m["tracker_ms"] = latency_stats(lat)
    return m


MACRO = ["HOTA", "IDF1", "MOTA", "DetA", "AssA"]
SUMS = ["IDSW", "Frag", "FP", "FN", "missed_tracks", "false_tracks", "gt_tracks", "pred_tracks", "reentry_id_reused"]


def aggregate(per_seq: dict[str, dict]) -> dict:
    out = {k: float(np.mean([m[k] for m in per_seq.values()])) for k in MACRO}
    out.update({k: int(sum(m[k] for m in per_seq.values())) for k in SUMS})
    for k in ("confirmation_delay_ms_mean", "termination_delay_ms_mean"):
        vals = [m[k] for m in per_seq.values() if m.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    tl = [m["tracker_ms"]["mean_ms"] for m in per_seq.values()]
    out["tracker_ms_mean"] = float(np.mean(tl))
    out["tracker_ms_p95_max"] = float(max(m["tracker_ms"]["p95_ms"] for m in per_seq.values()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detectors", nargs="+", default=DETECTORS)
    ap.add_argument("--trackers", nargs="+", default=TRACKERS)
    ap.add_argument("--ablation-detector", default="yunet@640", help="detector used for the interval / frame-rate ablations")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "tracking")
    args = ap.parse_args()
    manifest = json.loads((SEQ_DIR / "manifest.json").read_text())["sequences"]
    seqs = {s: [q for q in manifest if q["split"] == s] for s in ("dev", "test")}
    gts = {q["name"] + "|" + q["split"]: load_gt(q) for q in manifest}

    dets: dict[str, dict[str, dict]] = {}
    det_latency = {}
    for cfg in args.detectors:
        print(f"[detect] {cfg}", flush=True)
        dets[cfg] = {q["split"] + "/" + q["name"]: cached_detections(cfg, q, args.out) for q in manifest}
        det_latency[cfg] = {k: v["latency"] for k, v in dets[cfg].items()}

    results = []
    for cfg in args.detectors:
        for trk in args.trackers:
            # Tune on dev.
            best = None
            for thr in DET_THRESHOLDS:
                c = dict(BASE, det_threshold=thr)
                per = {q["name"]: eval_one(trk, c, dets[cfg]["dev/" + q["name"]], gts[q["name"] + "|dev"]) for q in seqs["dev"]}
                h = float(np.mean([m["HOTA"] for m in per.values()]))
                if best is None or h > best[1]:
                    best = (thr, h)
            c = dict(BASE, det_threshold=best[0])
            per = {q["name"]: eval_one(trk, c, dets[cfg]["test/" + q["name"]], gts[q["name"] + "|test"]) for q in seqs["test"]}
            agg = aggregate(per)
            by_gen = {g: aggregate({q["name"]: per[q["name"]] for q in seqs["test"] if q["generator"] == g})
                      for g in sorted({q["generator"] for q in seqs["test"]})}
            results.append({"detector": cfg, "tracker": trk, "det_threshold_from_dev": best[0], "dev_HOTA": best[1],
                            "config": c, "test": agg, "test_by_generator": by_gen, "test_per_sequence": per})
            print(f"[track] {cfg:20s} {trk:9s} thr={best[0]:.1f} HOTA={agg['HOTA']:.3f} IDF1={agg['IDF1']:.3f} "
                  f"MOTA={agg['MOTA']:.3f} IDSW={agg['IDSW']} Frag={agg['Frag']} reuse={agg['reentry_id_reused']} "
                  f"trk={agg['tracker_ms_mean']:.3f}ms", flush=True)

    # Stage 3: detector interval and frame rate, every tracker, on one detector's outputs.
    cfg = args.ablation_detector if args.ablation_detector in dets else args.detectors[0]
    ablations = []
    rng = np.random.default_rng(0)
    rates = [("30fps", lambda n: list(range(n))), ("15fps", lambda n: list(range(0, n, 2))),
             ("10fps", lambda n: list(range(0, n, 3))),
             ("jitter_drop40%", lambda n: sorted(set([0] + [i for i in range(n) if rng.random() > 0.4])))]
    for row in [r for r in results if r["detector"] == cfg]:
        trk, c = row["tracker"], row["config"]
        for every in (1, 2, 3):
            per = {q["name"]: eval_one(trk, c, dets[cfg]["test/" + q["name"]], gts[q["name"] + "|test"], detect_every=every)
                   for q in seqs["test"]}
            ablations.append({"tracker": trk, "experiment": "detect_every", "value": every, "test": aggregate(per)})
            print(f"[ablate] {trk:9s} detect_every={every} HOTA={ablations[-1]['test']['HOTA']:.3f} IDF1={ablations[-1]['test']['IDF1']:.3f} IDSW={ablations[-1]['test']['IDSW']}", flush=True)
        for lost_ms in (500, 1000, 1500, 2000, 3000):
            cl = dict(c, max_lost_ms=float(lost_ms))
            per = {q["name"]: eval_one(trk, cl, dets[cfg]["test/" + q["name"]], gts[q["name"] + "|test"]) for q in seqs["test"]}
            ablations.append({"tracker": trk, "experiment": "max_lost_ms", "value": lost_ms, "test": aggregate(per)})
            print(f"[ablate] {trk:9s} max_lost_ms={lost_ms} HOTA={ablations[-1]['test']['HOTA']:.3f} IDF1={ablations[-1]['test']['IDF1']:.3f} IDSW={ablations[-1]['test']['IDSW']} reuse={ablations[-1]['test']['reentry_id_reused']}", flush=True)
        for label, pick in rates:
            per = {}
            for q in seqs["test"]:
                rec = dets[cfg]["test/" + q["name"]]
                per[q["name"]] = eval_one(trk, c, rec, gts[q["name"] + "|test"], keep=pick(len(rec["frames"])))
            ablations.append({"tracker": trk, "experiment": "frame_rate", "value": label, "test": aggregate(per)})
            print(f"[ablate] {trk:9s} frames={label} HOTA={ablations[-1]['test']['HOTA']:.3f} IDF1={ablations[-1]['test']['IDF1']:.3f} IDSW={ablations[-1]['test']['IDSW']}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": {"tuning": "det_threshold in %s chosen by macro HOTA on dev sequences" % DET_THRESHOLDS,
                            "fixed": BASE, "detector_score_floor": SCORE_FLOOR,
                            "reported": "test sequences; macro-average over sequences for HOTA/IDF1/MOTA/DetA/AssA, sums for counts",
                            "visible_tracks": "confirmed tracks matched by a detection at the latest detector update"},
               "ablation_detector": cfg,
               "environment": environment(), "detector_latency_on_sequences": det_latency,
               "results": results, "ablations": ablations}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
