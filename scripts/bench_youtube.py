"""Benchmark every detector and tracker on online videos, streamed (never saved to disk).

    python scripts/bench_youtube.py URL [URL ...] [--max-frames 900]

Frames are read once from the stream, kept in memory as JPEG (q95) and decoded outside the
timed region, so every model sees exactly the same pixels. There is no human ground truth
for these videos, so the report may only claim:

* measured: latency, throughput, detections per frame, track-lifetime statistics;
* proxy: agreement with a heavy reference detector (SCRFD-10G at native resolution, run on
  every 3rd frame). The reference is itself imperfect and belongs to the SCRFD family, so
  agreement numbers are a sanity check, not accuracy.

Also measures what a recognition step (the suggested ArcFace models) would cost on the
faces actually found in these videos. Results -> results/youtube/summary.json, frame grids
-> results/youtube/*.jpg (frames from third-party videos: keep local, do not redistribute).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.benchenv import environment, latency_stats  # noqa: E402
from facetrack.detectors import create_detector  # noqa: E402
from facetrack.detectors.base import DetectorConfig  # noqa: E402
from facetrack.detectors.ort_common import make_session  # noqa: E402
from facetrack.detectors.scrfd import SCRFD  # noqa: E402
from facetrack.download import resolve_stream  # noqa: E402
from facetrack.models import model_path  # noqa: E402
from facetrack.sources import VideoFileSource  # noqa: E402
from facetrack.tracking import TRACKERS, TrackerConfig, create_tracker  # noqa: E402
from facetrack.tracking.base import TrackState  # noqa: E402

DL = ROOT / "models" / "_download"
DET_10G = DL / "buffalo_l" / "det_10g.onnx"
ARCFACE = {"arcface_r50": DL / "buffalo_l" / "w600k_r50.onnx", "arcface_mbf": DL / "buffalo_sc" / "w600k_mbf.onnx"}
FLOOR = 0.3  # detector-side score floor (ByteTrack needs the low-score detections)

# (label, registry name or "scrfd_10g", overrides, operating threshold used for "detections/frame" and agreement)
CONFIGS = [
    ("YuNet @640", "yunet", {"input_long_side": 640}, 0.6),
    ("YuNet @native", "yunet", {"input_long_side": 0}, 0.6),
    ("YuNet int8 @640", "yunet_int8", {"input_long_side": 640}, 0.6),
    ("YuNet OpenCV @640", "yunet_cv", {"input_long_side": 640}, 0.6),
    ("SCRFD-500M @640", "scrfd_500m_kps", {"input_long_side": 640}, 0.5),
    ("SCRFD-500M @native", "scrfd_500m_kps", {"input_long_side": 0}, 0.5),
    ("BlazeFace short", "mediapipe_short", {}, 0.5),
    ("BlazeFace full", "mediapipe_full", {}, 0.5),
    ("SCRFD-10G @640 (ResNet-class)", "scrfd_10g", {"input_long_side": 640}, 0.5),
]
REFERENCE_LABEL = "SCRFD-10G @native (reference)"
REF_THRESHOLD, REF_STRIDE = 0.6, 3


def build(name: str, overrides: dict):
    if name == "scrfd_10g":
        return SCRFD(DetectorConfig(str(DET_10G), FLOOR, 0.4, input_long_side=overrides.get("input_long_side", 640)))
    return create_detector(name, score_threshold=FLOOR, **overrides)


def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def agreement(dets: list, ref: list) -> dict:
    """Greedy IoU>=0.5 matching of detections (by score) to reference faces."""
    used, tp, per_bucket = set(), 0, {"<24px": [0, 0], "24-64px": [0, 0], ">=64px": [0, 0]}
    matched_ref = set()
    for d in sorted(dets, key=lambda d: -d.confidence):
        best, bj = 0.5, None
        for j, r in enumerate(ref):
            if j in used:
                continue
            v = iou(d.bbox, r.bbox)
            if v >= best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
            matched_ref.add(bj)
            tp += 1
    for j, r in enumerate(ref):
        h = r.bbox[3]
        b = "<24px" if h < 24 else "24-64px" if h < 64 else ">=64px"
        per_bucket[b][1] += 1
        per_bucket[b][0] += j in matched_ref
    return {"tp": tp, "fp": len(dets) - tp, "fn": len(ref) - tp, "buckets": per_bucket}


def summarise_agreement(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out = {"precision_vs_ref": p, "recall_vs_ref": r, "f1_vs_ref": 2 * p * r / max(p + r, 1e-9), "ref_faces": tp + fn}
    for b in ("<24px", "24-64px", ">=64px"):
        hit, tot = sum(x["buckets"][b][0] for x in rows), sum(x["buckets"][b][1] for x in rows)
        out[f"recall_{b}"] = hit / tot if tot else None
        out[f"n_ref_{b}"] = tot
    return out


def run_tracker(name: str, dets_per_frame: list, stamps: list[float], tcfg: dict) -> dict:
    tr = create_tracker(name, TrackerConfig.from_dict(tcfg))
    life: dict[int, list[float]] = {}
    visible = []
    t0 = time.perf_counter()
    for i, dets in enumerate(dets_per_frame):
        out = tr.update(dets, stamps[i], i)
        vis = [t for t in out.tracks if t.state is TrackState.CONFIRMED]
        visible.append(len(vis))
        for t in vis:
            life.setdefault(t.track_id, [stamps[i], stamps[i]])[1] = stamps[i]
    tr.finish(stamps[-1], len(stamps) - 1)
    secs = time.perf_counter() - t0
    lens = [(b - a) / 1000 for a, b in life.values()]
    return {"ids_issued": tr.ids_issued, "confirmed_tracks": len(life), "mean_visible": float(np.mean(visible)),
            "median_life_s": statistics.median(lens) if lens else 0.0, "share_under_1s": sum(l < 1 for l in lens) / max(len(lens), 1),
            "ms_per_frame": secs * 1e3 / len(stamps)}


def timed_pass(det, jpgs: list[bytes], frames_idx: list[int]) -> tuple[dict, list[list]]:
    tot, inf, dets_out = [], [], []
    img0 = cv2.imdecode(np.frombuffer(jpgs[0], np.uint8), cv2.IMREAD_COLOR)
    for _ in range(5):
        det.detect(img0)
    for i in frames_idx:
        img = cv2.imdecode(np.frombuffer(jpgs[i], np.uint8), cv2.IMREAD_COLOR)  # decode is outside the timer
        dets, t = det.detect(img)
        tot.append(t.total_ms)
        inf.append(t.inference_ms)
        dets_out.append(dets)
    return {"end_to_end": latency_stats(tot), "inference": latency_stats(inf), "fused": bool(t.fused)}, dets_out


def read_stream(url: str, max_frames: int, max_height: int):
    stream, meta = resolve_stream(url, max_height)  # one lookup gives both the stream and the metadata
    with VideoFileSource(stream) as src:
        info = src.info
        jpgs, stamps = [], []
        t0 = time.perf_counter()
        for f in src:
            jpgs.append(cv2.imencode(".jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
            stamps.append(f.timestamp_ms)
            if len(jpgs) >= max_frames:
                break
    return meta, info, jpgs, stamps, time.perf_counter() - t0


def draw_boxes(img, dets, thr, color, label):
    out = img.copy()
    s = max(0.5, out.shape[1] / 1400)
    n = 0
    for d in dets:
        if d.confidence < thr:
            continue
        x, y, w, h = (int(round(v)) for v in d.bbox)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, max(2, out.shape[1] // 500))
        n += 1
    cv2.rectangle(out, (0, 0), (out.shape[1], int(46 * s)), (0, 0, 0), -1)
    cv2.putText(out, f"{label}: {n} faces", (8, int(32 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def contact_sheet(jpgs, frame_i, per_label: dict, thr_by: dict, path: Path, tile_w: int = 640) -> None:
    img = cv2.imdecode(np.frombuffer(jpgs[frame_i], np.uint8), cv2.IMREAD_COLOR)
    tiles = []
    for label, dets in per_label.items():
        t = draw_boxes(img, dets, thr_by[label], (60, 220, 60), label)
        tiles.append(cv2.resize(t, (tile_w, int(t.shape[0] * tile_w / t.shape[1])), interpolation=cv2.INTER_AREA))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def embedding_cost(jpgs, frame_idx: list[int], dets_by_frame: dict[int, list]) -> dict:
    sessions = {k: make_session(str(p), 0) for k, p in ARCFACE.items()}
    names = {k: s.get_inputs()[0].name for k, s in sessions.items()}
    per_frame = {k: [] for k in sessions}
    nfaces = []
    for i in frame_idx:
        img = cv2.imdecode(np.frombuffer(jpgs[i], np.uint8), cv2.IMREAD_COLOR)
        crops = []
        for d in dets_by_frame[i]:
            if d.confidence < 0.6:
                continue
            x, y, w, h = d.bbox
            m = 0.2
            x1, y1, x2, y2 = max(int(x - m * w), 0), max(int(y - m * h), 0), min(int(x + w * (1 + m)), img.shape[1]), min(int(y + h * (1 + m)), img.shape[0])
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            c = cv2.resize(img[y1:y2, x1:x2], (112, 112))[:, :, ::-1].astype(np.float32)
            crops.append(((c - 127.5) / 127.5).transpose(2, 0, 1)[None])
        nfaces.append(len(crops))
        if not crops:
            continue
        for k, s in sessions.items():
            t0 = time.perf_counter()
            for c in crops:  # the exported ArcFace models have a fixed batch of 1
                s.run(None, {names[k]: c})
            per_frame[k].append((time.perf_counter() - t0) * 1e3)
    nz = [n for n in nfaces if n]
    return {"frames_sampled": len(frame_idx), "frames_with_faces": len(nz), "faces_per_frame_mean": float(np.mean(nfaces)),
            "faces_per_frame_p95": float(np.percentile(nfaces, 95)), "faces_per_frame_max": int(max(nfaces)),
            "per_frame_ms": {k: latency_stats(v) for k, v in per_frame.items() if v}}


def require_models() -> None:
    missing = [p.name for p in (DET_10G, *ARCFACE.values()) if not p.exists()]
    if missing:
        raise SystemExit(f"missing models: {', '.join(missing)}. Run: python scripts/download_suggested_models.py (and python scripts/download_models.py)")


def main() -> None:
    require_models()
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "youtube")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    tcfg = {k: v for k, v in cfg["tracker"].items() if k != "name"}
    args.out.mkdir(parents=True, exist_ok=True)
    result: dict = {"environment": environment(), "protocol": {
        "max_frames": args.max_frames, "max_height": args.max_height, "detector_floor": FLOOR, "tracker_config": tcfg,
        "reference": REFERENCE_LABEL, "reference_threshold": REF_THRESHOLD, "reference_stride": REF_STRIDE,
        "note": "no human ground truth; agreement is a proxy against a heavy detector"}, "videos": {}}

    for url in args.urls:
        vid = url.split("v=")[-1]
        try:
            meta, info, jpgs, stamps, read_s = read_stream(url, args.max_frames, args.max_height)
        except RuntimeError as e:
            raise SystemExit(f"error: {e}")
        n = len(jpgs)
        print(f"\n== {meta['title']} | {info.width}x{info.height} @ {info.fps} fps | {n} frames ({stamps[-1] / 1000:.1f}s) read in {read_s:.0f}s", flush=True)
        v = {"url": url, "title": meta["title"], "channel": meta.get("channel"), "duration_s": meta.get("duration"),
             "stream": {"width": info.width, "height": info.height, "fps": info.fps, "codec": info.codec},
             "frames_used": n, "seconds_used": stamps[-1] / 1000, "detectors": {}}
        all_idx = list(range(n))
        dets_by = {}
        for label, name, ov, thr in CONFIGS:
            det = build(name, ov)
            lat, dets = timed_pass(det, jpgs, all_idx)
            det.close()
            dets_by[label] = dets
            ok = [[d for d in fd if d.confidence >= thr] for fd in dets]
            lat["faces_per_frame_mean"] = float(np.mean([len(f) for f in ok]))
            lat["frames_with_face"] = float(np.mean([len(f) > 0 for f in ok]))
            lat["max_fps"] = 1000.0 / lat["end_to_end"]["mean_ms"]
            v["detectors"][label] = lat
            e = lat["end_to_end"]
            print(f"  {label:32s} p50 {e['p50_ms']:7.1f} ms p95 {e['p95_ms']:7.1f} ms  faces/frame {lat['faces_per_frame_mean']:.2f}", flush=True)

        ref_idx = all_idx[::REF_STRIDE]
        ref_det = build("scrfd_10g", {"input_long_side": 0})
        rlat, ref_dets = timed_pass(ref_det, jpgs, ref_idx)
        ref_det.close()
        v["reference"] = {"label": REFERENCE_LABEL, "latency": rlat, "frames": len(ref_idx)}
        ref_ok = [[d for d in fd if d.confidence >= REF_THRESHOLD] for fd in ref_dets]
        v["reference"]["faces_per_frame_mean"] = float(np.mean([len(f) for f in ref_ok]))
        print(f"  {REFERENCE_LABEL}: p50 {rlat['end_to_end']['p50_ms']:.0f} ms, {v['reference']['faces_per_frame_mean']:.2f} faces/frame on {len(ref_idx)} frames", flush=True)
        for label, _, _, thr in CONFIGS:
            rows = [agreement([d for d in dets_by[label][i] if d.confidence >= thr], r) for i, r in zip(ref_idx, ref_ok)]
            v["detectors"][label]["agreement"] = summarise_agreement(rows)

        stamp_list = list(stamps)
        for label, *_ in CONFIGS:
            v["detectors"][label]["trackers"] = {t: run_tracker(t, dets_by[label], stamp_list, tcfg) for t in TRACKERS}
        # ArcFace cost on the faces this video really contains (YuNet@native detections)
        sample = all_idx[::10]
        v["recognition_cost"] = embedding_cost(jpgs, sample, {i: dets_by["YuNet @native"][i] for i in sample})
        print("  recognition:", json.dumps({k: round(x["p50_ms"], 1) for k, x in v["recognition_cost"]["per_frame_ms"].items()}), "ms/frame p50;",
              f"faces/frame mean {v['recognition_cost']['faces_per_frame_mean']:.1f} max {v['recognition_cost']['faces_per_frame_max']}", flush=True)

        # frame grid at the frame where the reference sees the most faces
        k = int(np.argmax([len(f) for f in ref_ok]))
        fi = ref_idx[k]
        grid = {l: dets_by[l][fi] for l, *_ in CONFIGS}
        grid[REFERENCE_LABEL] = ref_dets[k]
        thr_by = {l: t for l, _, _, t in CONFIGS}
        thr_by[REFERENCE_LABEL] = REF_THRESHOLD
        contact_sheet(jpgs, fi, grid, thr_by, args.out / f"{vid}_grid.jpg")
        v["grid_frame"] = fi
        result["videos"][vid] = v
        (args.out / "summary.json").write_text(json.dumps(result, indent=1))
        del jpgs
    print(f"\nwrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
