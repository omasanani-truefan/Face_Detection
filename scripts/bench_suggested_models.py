"""Side experiment: the additionally suggested models (ResNet, ArcFace, U-Net), as far as they can be measured.

    python scripts/bench_suggested_models.py

* ResNet-class detector -> SCRFD-10G-KPS (InsightFace buffalo_l), accuracy on the same
  WIDER dev/test protocol and latency, next to SCRFD-500M. Not vanilla ResNet: a backbone
  is not a detector.
* ArcFace -> w600k_r50 (ResNet-50 + ArcFace loss) and w600k_mbf (MobileFaceNet + ArcFace
  loss): per-crop embedding latency only. They are recognition models; the brief keeps
  recognition out of Phases 1-2, so this measures what Phase 3 would add, nothing more.
* U-Net -> not run: no pretrained face model exists, and it outputs masks, not faces.

Weights are InsightFace's (non-commercial research use). Results ->
results/suggested_models/summary.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.benchenv import environment, latency_stats  # noqa: E402
from facetrack.detectors.base import DetectorConfig  # noqa: E402
from facetrack.detectors.ort_common import make_session  # noqa: E402
from facetrack.detectors.scrfd import SCRFD  # noqa: E402
from facetrack.metrics.detection import evaluate  # noqa: E402

DL = ROOT / "models" / "_download"
DET_10G = DL / "buffalo_l" / "det_10g.onnx"
ARC = {"arcface_r50 (ResNet-50)": DL / "buffalo_l" / "w600k_r50.onnx", "arcface_mbf (MobileFaceNet)": DL / "buffalo_sc" / "w600k_mbf.onnx"}


def detect_all(det: SCRFD, images: list[dict], root: Path) -> list[dict]:
    out = []
    for im in images:
        dets, _ = det.detect(cv2.imread(str(root / im["file_name"])))
        out += [{"image_id": im["id"], "bbox": [round(v, 2) for v in d.bbox], "score": round(d.confidence, 5)} for d in dets]
    return out


def accuracy(model: Path, nms: float) -> dict:
    dev = json.loads((ROOT / "data/widerface/dev.json").read_text())
    test = json.loads((ROOT / "data/widerface/test.json").read_text())
    root = ROOT / dev["info"]["image_root"]
    det = SCRFD(DetectorConfig(str(model), 0.05, nms, input_long_side=640))
    t0 = time.perf_counter()
    dets = detect_all(det, dev["images"] + test["images"], root)
    secs = time.perf_counter() - t0
    dev_ids = {i["id"] for i in dev["images"]}
    thr, _ = evaluate(dev, [d for d in dets if d["image_id"] in dev_ids]).best_f1_threshold()
    r = evaluate(test, [d for d in dets if d["image_id"] not in dev_ids])
    op = r.at_threshold(thr, 0.5)
    return {"threshold_from_dev": thr, "AP50": r.ap["all"]["AP50"], "AP": r.ap["all"]["AP"], "precision": op["all"]["precision"],
            "recall": op["all"]["recall"], "f1": op["all"]["f1"], "fp_per_image": op["all"]["fp_per_image"],
            "recall_by_height_px": {b: op[b]["recall"] for b in ("tiny", "small", "medium", "large")}, "wall_seconds": secs}


def detector_latency(model: Path, frames: list[np.ndarray]) -> dict:
    det = SCRFD(DetectorConfig(str(model), 0.5, 0.4, input_long_side=640))
    for i in range(30):
        det.detect(frames[i % len(frames)])
    tot, inf = [], []
    for i in range(300):
        _, t = det.detect(frames[i % len(frames)])
        tot.append(t.total_ms)
        inf.append(t.inference_ms)
    return {"end_to_end": latency_stats(tot), "inference": latency_stats(inf)}


def embedding_latency(model: Path, threads: int) -> dict:
    s = make_session(str(model), threads)
    name = s.get_inputs()[0].name
    x = np.random.default_rng(0).normal(0, 1, (1, 3, 112, 112)).astype(np.float32)
    for _ in range(20):
        s.run(None, {name: x})
    lat = []
    for _ in range(100):
        t0 = time.perf_counter()
        out = s.run(None, {name: x})
        lat.append((time.perf_counter() - t0) * 1e3)
    return {"embedding_dim": int(out[0].shape[-1]), "size_mb": model.stat().st_size / 2**20, **latency_stats(lat)}


def main() -> None:
    missing = [p.name for p in (DET_10G, *ARC.values()) if not p.exists()]
    if missing:
        raise SystemExit(f"missing models: {', '.join(missing)}. Run: python scripts/download_suggested_models.py")
    from bench_detection_latency import load_frames  # same frames as the main latency benchmark

    frames = load_frames((640, 480))
    res: dict = {"environment": environment(), "license": "InsightFace pretrained weights: non-commercial research only"}
    print("SCRFD-10G accuracy (WIDER test, thr from dev)...", flush=True)
    res["scrfd_10g_accuracy"] = accuracy(DET_10G, 0.4)
    a = res["scrfd_10g_accuracy"]
    print(f"  AP50={a['AP50']:.3f} AP={a['AP']:.3f} R={a['recall']:.3f} P={a['precision']:.3f} thr={a['threshold_from_dev']:.2f}", flush=True)
    res["detector_latency_640x480"] = {}
    for name, p in [("scrfd_500m_kps", ROOT / "models/scrfd_500m_kps.onnx"), ("scrfd_10g_kps", DET_10G)]:
        res["detector_latency_640x480"][name] = detector_latency(p, frames)
        m = res["detector_latency_640x480"][name]["end_to_end"]
        print(f"  {name}: e2e p50 {m['p50_ms']:.1f} ms p95 {m['p95_ms']:.1f} ms", flush=True)
    res["embedding_latency_112x112"] = {}
    for threads in (0, 1):
        for name, p in ARC.items():
            key = f"{name} threads={'default' if not threads else threads}"
            res["embedding_latency_112x112"][key] = embedding_latency(p, threads)
            m = res["embedding_latency_112x112"][key]
            print(f"  {key}: p50 {m['p50_ms']:.1f} ms p95 {m['p95_ms']:.1f} ms dim {m['embedding_dim']} size {m['size_mb']:.0f} MB", flush=True)
    out = ROOT / "results" / "suggested_models"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
