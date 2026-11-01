"""Sample detection visualisations for the report: the same WIDER *test* images through
each detector at its dev-tuned threshold. Green = ground truth, colour = detections.

    python scripts/make_visuals.py   # after bench_detection_accuracy.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.detectors import create_detector  # noqa: E402

DETECTORS = [("yunet", 640), ("scrfd_500m_kps", 640), ("mediapipe_full", 0), ("mediapipe_short", 0)]


def pick_images(test: dict) -> list[dict]:
    """A close-range group, a crowd of small faces, and a poorly lit scene."""
    by: dict[int, list] = {}
    for a in test["annotations"]:
        if not a["iscrowd"]:
            by.setdefault(a["image_id"], []).append(a)
    imgs = {i["id"]: i for i in test["images"]}

    def score(cond):
        c = [k for k, v in by.items() if cond(imgs[k], v)]
        return min(c) if c else None

    close = score(lambda im, v: 3 <= len(v) <= 6 and min(a["bbox"][3] for a in v) > 0.18 * im["height"])
    crowd = score(lambda im, v: 25 <= len(v) <= 60 and np.median([a["bbox"][3] for a in v]) < 25)
    dark = score(lambda im, v: 2 <= len(v) <= 8 and sum(a["attributes"]["illumination"] for a in v) >= len(v) and min(a["bbox"][3] for a in v) > 40)
    return [imgs[k] for k in (close, crowd, dark) if k is not None]


def main() -> None:
    test = json.loads((ROOT / "data/widerface/test.json").read_text())
    summary = json.loads((ROOT / "results/detection_accuracy/summary.json").read_text())
    thr = {r["config"]: r["threshold_from_dev"] for r in summary["results"]}
    root = ROOT / test["info"]["image_root"]
    chosen = pick_images(test)
    gt = {}
    for a in test["annotations"]:
        if not a["iscrowd"]:
            gt.setdefault(a["image_id"], []).append(a["bbox"])
    tiles = []
    for name, size in DETECTORS:
        cid = f"{name}@{size}" if size else name
        det = create_detector(name, score_threshold=thr[cid], input_long_side=size)
        row = []
        for im in chosen:
            img = cv2.imread(str(root / im["file_name"]))
            dets, _ = det.detect(img)
            for x, y, w, h in gt.get(im["id"], []):
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 0), 1)
            for d in dets:
                x, y, w, h = (int(v) for v in d.bbox)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 80, 255), 2)
            label = f"{cid}  thr {thr[cid]:.2f}  found {len(dets)} / GT {len(gt.get(im['id'], []))}"
            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            row.append(cv2.resize(img, (560, int(560 * img.shape[0] / img.shape[1]))))
        det.close()
        h = max(t.shape[0] for t in row)
        tiles.append(np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in row]))
    w = max(t.shape[1] for t in tiles)
    grid = np.vstack([cv2.copyMakeBorder(t, 0, 4, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles])
    out = ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "detections_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {out / 'detections_grid.jpg'} from {[im['file_name'] for im in chosen]}")


if __name__ == "__main__":
    main()
