"""Detection accuracy on WIDER FACE val (dev/test split), every detector, same images.

    python scripts/bench_detection_accuracy.py                 # full run
    python scripts/bench_detection_accuracy.py --limit 50      # smoke test

For each config the script:
1. runs the detector over every val image with a low score floor, saving raw detections
   (results/detection_accuracy/raw/<config>.json.gz) so evaluation can be re-run offline;
2. picks the score threshold that maximises F1 @ IoU 0.5 on **dev**;
3. reports AP and operating-point metrics on **test** at that threshold, per size bucket.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.benchenv import environment  # noqa: E402
from facetrack.detectors import create_detector  # noqa: E402
from facetrack.metrics.detection import SIZE_BUCKETS, evaluate  # noqa: E402

# (detector, input long side). None = model's own fixed input (MediaPipe).
MAIN = [("yunet", 640), ("yunet_int8", 640), ("scrfd_500m_kps", 640), ("mediapipe_short", None), ("mediapipe_full", None)]
SWEEP = [("yunet", 320), ("yunet", 480), ("yunet", 1024), ("scrfd_500m_kps", 320), ("scrfd_500m_kps", 480), ("scrfd_500m_kps", 1024)]


def config_id(name: str, size: int | None) -> str:
    return f"{name}@{size}" if size else name


def collect(name: str, size: int | None, images: list[dict], img_root: Path, floor: float, raw_dir: Path) -> list[dict]:
    raw_path = raw_dir / f"{config_id(name, size).replace('@', '_')}.json.gz"
    if raw_path.exists():
        with gzip.open(raw_path, "rt") as f:
            return json.load(f)["detections"]
    det = create_detector(name, score_threshold=floor, input_long_side=size)
    out, t0 = [], time.perf_counter()
    for k, im in enumerate(images):
        img = cv2.imread(str(img_root / im["file_name"]))
        dets, _ = det.detect(img)
        out += [{"image_id": im["id"], "bbox": [round(v, 2) for v in d.bbox], "score": round(d.confidence, 5)} for d in dets]
        if (k + 1) % 500 == 0:
            print(f"    {config_id(name, size)}: {k + 1}/{len(images)}", flush=True)
    info = det.describe()
    det.close()
    raw_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_path, "wt") as f:
        json.dump({"detector": info, "score_floor": floor, "seconds": time.perf_counter() - t0, "detections": out}, f)
    return out


def subset(gt: dict, n: int | None) -> dict:
    if not n:
        return gt
    imgs = gt["images"][:n]
    ids = {i["id"] for i in imgs}
    return dict(gt, images=imgs, annotations=[a for a in gt["annotations"] if a["image_id"] in ids])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", choices=["main", "sweep", "all"], default="all")
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=None, help="images per split (smoke test)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "detection_accuracy")
    args = ap.parse_args()

    dev = subset(json.loads((ROOT / "data/widerface/dev.json").read_text()), args.limit)
    test = subset(json.loads((ROOT / "data/widerface/test.json").read_text()), args.limit)
    img_root = ROOT / dev["info"]["image_root"]
    images = dev["images"] + test["images"]
    raw_dir = args.out / ("raw" if not args.limit else f"raw_limit{args.limit}")
    configs = {"main": MAIN, "sweep": SWEEP, "all": MAIN + SWEEP}[args.configs]

    rows = []
    for name, size in configs:
        cid = config_id(name, size)
        print(f"[{cid}] collecting", flush=True)
        dets = collect(name, size, images, img_root, args.score_floor, raw_dir)
        dev_ids = {i["id"] for i in dev["images"]}
        r_dev = evaluate(dev, [d for d in dets if d["image_id"] in dev_ids])
        thr, dev_f1 = r_dev.best_f1_threshold()
        test_dets = [d for d in dets if d["image_id"] not in dev_ids]
        r_test = evaluate(test, test_dets)
        r_rel = evaluate(test, test_dets, relative=True)
        op = r_test.at_threshold(thr, 0.5)
        op30 = r_test.at_threshold(thr, 0.3)
        rows.append({"config": cid, "detector": name, "input_long_side": size, "threshold_from_dev": thr,
                     "dev_f1": dev_f1, "test_ap": r_test.ap, "test_at_threshold_iou50": op,
                     "test_at_threshold_iou30": op30, "test_gt_counts": r_test.gt_counts,
                     "test_relative_at_threshold_iou50": r_rel.at_threshold(thr, 0.5),
                     "test_relative_at_threshold_iou30": r_rel.at_threshold(thr, 0.3),
                     "test_relative_gt_counts": r_rel.gt_counts})
        a = r_test.ap["all"]
        print(f"[{cid}] thr={thr:.2f} test AP50={a['AP50']:.3f} AP={a['AP']:.3f} "
              f"R={op['all']['recall']:.3f} P={op['all']['precision']:.3f} "
              f"recall tiny/small/medium/large={op['tiny']['recall']:.2f}/{op['small']['recall']:.2f}/"
              f"{op['medium']['recall']:.2f}/{op['large']['recall']:.2f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": {"name": "WIDER FACE val", "dev_images": len(dev["images"]), "test_images": len(test["images"]),
                           "size_buckets_px_height": SIZE_BUCKETS, "limit": args.limit},
               "protocol": "threshold = argmax F1@IoU0.5 on dev; all reported metrics on test",
               "environment": environment(), "results": rows}
    name = "summary.json" if not args.limit else f"summary_limit{args.limit}.json"
    (args.out / name).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
