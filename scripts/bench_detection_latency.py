"""Detector latency / resource benchmark on this machine's CPU.

    python scripts/bench_detection_latency.py               # all configs, 3 repetitions
    python scripts/bench_detection_latency.py --sustained 180

Every (config, frame resolution, repetition) runs in a **fresh subprocess**, so startup
time and peak memory are not polluted by earlier configs. Inside a worker:
model load -> first (cold) inference -> 30 warm-up calls -> 300 timed calls cycling over
the same 60 real frames (taken from the test sequences, resized to the target
resolution). Pre-processing, inference and post-processing are timed separately where
the runtime allows (see ``StageTimings.fused``).

Run on an otherwise idle machine, on mains power. Results go to
``results/detection_latency/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESOLUTIONS = {"480p": (640, 480), "720p": (1280, 720), "1080p": (1920, 1080)}
# (detector, input long side, ORT threads); threads 0 = runtime default.
CONFIGS = [
    ("yunet", 640, 0), ("yunet", 640, 1), ("yunet", 320, 0), ("yunet", None, 0),
    ("yunet_int8", 640, 0), ("yunet_cv", 640, 0),
    ("scrfd_500m_kps", 640, 0), ("scrfd_500m_kps", 640, 1), ("scrfd_500m_kps", 320, 0), ("scrfd_500m_kps", None, 0),
    ("mediapipe_short", None, 0), ("mediapipe_full", None, 0),
]
WARMUP, TIMED, N_FRAMES = 30, 300, 60


def config_id(name, size, threads) -> str:
    s = f"{name}@{size or 'native'}"
    return s + (f"/t{threads}" if threads else "")


def load_frames(wh: tuple[int, int]):
    import cv2

    from facetrack.sources import VideoFileSource

    src_seqs = ["test/multi_enter.mp4", "test/crossing.mp4", "test/far_group.mp4", "test/pan_crowd.mp4"]
    frames = []
    for s in src_seqs:
        with VideoFileSource(ROOT / "data" / "sequences" / s) as v:
            for k, f in enumerate(v):
                if k % 12 == 0:
                    frames.append(cv2.resize(f.image, wh, interpolation=cv2.INTER_AREA if f.image.shape[1] > wh[0] else cv2.INTER_LINEAR))
                if len(frames) >= N_FRAMES * (src_seqs.index(s) + 1) // len(src_seqs):
                    break
    return frames


def worker(name: str, size: int | None, threads: int, res: str) -> dict:
    import numpy as np
    import psutil

    from facetrack.benchenv import latency_stats
    from facetrack.detectors import create_detector

    proc = psutil.Process()
    frames = load_frames(RESOLUTIONS[res])
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    det = create_detector(name, input_long_side=size, num_threads=threads, score_threshold=0.5)
    t_load = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    det.detect(frames[0])
    t_first = (time.perf_counter() - t0) * 1e3
    rss_loaded = proc.memory_info().rss
    for i in range(WARMUP):
        det.detect(frames[i % len(frames)])
    pre, inf, post, tot, wall = [], [], [], [], []
    cpu0 = proc.cpu_times()
    w0 = time.perf_counter()
    rss_peak = 0
    fused = False
    for i in range(TIMED):
        f = frames[i % len(frames)]
        t0 = time.perf_counter()
        _, t = det.detect(f)
        wall.append((time.perf_counter() - t0) * 1e3)
        pre.append(t.preprocess_ms)
        inf.append(t.inference_ms)
        post.append(t.postprocess_ms)
        tot.append(t.total_ms)
        fused = t.fused
        if i % 50 == 0:
            rss_peak = max(rss_peak, proc.memory_info().rss)
    elapsed = time.perf_counter() - w0
    cpu1 = proc.cpu_times()
    cpu_s = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
    info = det.describe()
    det.close()
    return {
        "detector_info": info, "resolution": res, "frame_wh": RESOLUTIONS[res],
        "startup_ms": {"load": t_load, "first_inference": t_first},
        "preprocess": latency_stats(pre), "inference": latency_stats(inf), "postprocess": latency_stats(post),
        "end_to_end": latency_stats(wall), "stages_fused": fused,
        "fps_end_to_end": 1000.0 / float(np.mean(wall)),
        "cpu_cores_busy": cpu_s / elapsed,  # 1.0 = one core fully busy
        "rss_mb": {"before_load": rss0 / 2**20, "after_load": rss_loaded / 2**20, "peak_during_run": max(rss_peak, rss_loaded) / 2**20,
                   "model_delta": (rss_loaded - rss0) / 2**20},
    }


def sustained(name: str, size: int | None, seconds: float, res: str = "480p") -> dict:
    from facetrack.detectors import create_detector

    frames = load_frames(RESOLUTIONS[res])
    det = create_detector(name, input_long_side=size, score_threshold=0.5)
    for i in range(WARMUP):
        det.detect(frames[i % len(frames)])
    windows, cur, w_start, t_start, i = [], [], time.perf_counter(), time.perf_counter(), 0
    while time.perf_counter() - t_start < seconds:
        t0 = time.perf_counter()
        det.detect(frames[i % len(frames)])
        cur.append((time.perf_counter() - t0) * 1e3)
        i += 1
        if time.perf_counter() - w_start >= 10:
            windows.append({"t_end_s": round(time.perf_counter() - t_start, 1), "mean_ms": statistics.mean(cur),
                            "p95_ms": sorted(cur)[int(0.95 * len(cur))], "n": len(cur)})
            cur, w_start = [], time.perf_counter()
    return {"config": config_id(name, size, 0), "resolution": res, "seconds": seconds, "windows": windows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", nargs=4, metavar=("NAME", "SIZE", "THREADS", "RES"), help=argparse.SUPPRESS)
    ap.add_argument("--sustained-worker", nargs=3, metavar=("NAME", "SIZE", "SECONDS"), help=argparse.SUPPRESS)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--resolutions", nargs="+", default=list(RESOLUTIONS))
    ap.add_argument("--only", nargs="*", help="config ids to run (default all)")
    ap.add_argument("--sustained", type=float, default=0, help="also run a sustained test of this many seconds")
    ap.add_argument("--sustained-config", default="yunet@640")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "detection_latency")
    args = ap.parse_args()

    if args.worker:
        n, s, t, r = args.worker
        print(json.dumps(worker(n, 0 if s == "native" else int(s), int(t), r)))
        return
    if args.sustained_worker:
        n, s, sec = args.sustained_worker
        print(json.dumps(sustained(n, 0 if s == "native" else int(s), float(sec))))
        return

    from facetrack.bench.onnx_cost import onnx_cost
    from facetrack.benchenv import environment
    from facetrack.detectors import create_detector

    py = sys.executable
    rows = []
    for name, size, threads in CONFIGS:
        cid = config_id(name, size, threads)
        if args.only and cid not in args.only:
            continue
        for res in args.resolutions:
            reps = []
            for rep in range(args.reps):
                cmd = [py, __file__, "--worker", name, str(size or "native"), str(threads), res]
                out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
                line = next((ln for ln in reversed(out.stdout.splitlines()) if ln.startswith("{")), None)
                if out.returncode or not line:
                    print(f"[{cid} {res}] worker failed:\n{out.stderr[-2000:]}", flush=True)
                    break
                reps.append(json.loads(line))
            if not reps:
                continue
            med = lambda key, stat: statistics.median(r[key][stat] for r in reps)  # noqa: E731
            row = {"config": cid, "detector": name, "input_long_side": size, "threads": threads, "resolution": res,
                   "repetitions": len(reps), "stages_fused": reps[0]["stages_fused"],
                   "median_of_reps": {
                       "preprocess_p50_ms": med("preprocess", "p50_ms"), "inference_p50_ms": med("inference", "p50_ms"),
                       "postprocess_p50_ms": med("postprocess", "p50_ms"), "e2e_mean_ms": med("end_to_end", "mean_ms"),
                       "e2e_p50_ms": med("end_to_end", "p50_ms"), "e2e_p95_ms": med("end_to_end", "p95_ms"),
                       "fps": statistics.median(r["fps_end_to_end"] for r in reps),
                       "cpu_cores_busy": statistics.median(r["cpu_cores_busy"] for r in reps),
                       "startup_load_ms": statistics.median(r["startup_ms"]["load"] for r in reps),
                       "startup_first_inference_ms": statistics.median(r["startup_ms"]["first_inference"] for r in reps),
                       "rss_model_delta_mb": statistics.median(r["rss_mb"]["model_delta"] for r in reps),
                       "rss_peak_mb": statistics.median(r["rss_mb"]["peak_during_run"] for r in reps)},
                   "e2e_p50_spread_ms": [min(r["end_to_end"]["p50_ms"] for r in reps), max(r["end_to_end"]["p50_ms"] for r in reps)],
                   "detector_info": reps[0]["detector_info"], "reps": reps}
            rows.append(row)
            m = row["median_of_reps"]
            print(f"[{cid:26s} {res:5s}] e2e p50={m['e2e_p50_ms']:7.2f} p95={m['e2e_p95_ms']:7.2f} ms "
                  f"(pre {m['preprocess_p50_ms']:.2f} / inf {m['inference_p50_ms']:.2f} / post {m['postprocess_p50_ms']:.2f}"
                  f"{' fused' if row['stages_fused'] else ''})  {m['fps']:6.1f} FPS  cpu={m['cpu_cores_busy']:.2f} cores  "
                  f"rss+{m['rss_model_delta_mb']:.0f}MB  load={m['startup_load_ms']:.0f}ms", flush=True)

    # Model cost from the ONNX graphs (MediaPipe ships TFLite; not counted here).
    cost = {}
    for name in ("yunet", "scrfd_500m_kps"):
        path = create_detector(name).config.model_path
        cost[name] = {f"{h}x{w}": onnx_cost(path, (h, w)) for h, w in [(320, 320), (480, 640), (640, 640)]}

    sus = None
    if args.sustained:
        n, _, s = args.sustained_config.partition("@")
        cmd = [py, __file__, "--sustained-worker", n, s or "native", str(args.sustained)]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        line = next((ln for ln in reversed(out.stdout.splitlines()) if ln.startswith("{")), None)
        sus = json.loads(line) if line else {"error": out.stderr[-2000:]}
        if line:
            print("[sustained] " + "  ".join(f"{w['t_end_s']:.0f}s:{w['mean_ms']:.2f}" for w in sus["windows"]), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    name = "summary.json" if not args.only else "summary_partial.json"
    (args.out / name).write_text(json.dumps({
        "protocol": {"warmup": WARMUP, "timed_iterations": TIMED, "frames": N_FRAMES, "repetitions": args.reps,
                     "isolation": "fresh subprocess per (config, resolution, repetition)",
                     "reported": "median across repetitions of each repetition's statistic"},
        "environment": environment(), "model_cost_onnx_graph": cost, "results": rows, "sustained": sus}, indent=1))
    print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
