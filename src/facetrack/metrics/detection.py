"""COCO-style face detection metrics, bucketed by face *height* in pixels.

Evaluation is pycocotools' reference implementation -- the matching rules (greedy by
score, ignore regions, crowd semantics) are not reimplemented here. Two adaptations:

* Size buckets use box height, the dimension that decides whether a detector can see a
  face, instead of COCO's area. pycocotools buckets on the ``area`` field, so we set
  ``area = height**2`` on GT and detections and give ``areaRng`` in squared pixels.
* Operating-point metrics (precision / recall / F1 / FP per image at a score threshold)
  are read from pycocotools' per-image match records. Matching is greedy in descending
  score order, so thresholding those records afterwards is exact.
"""

from __future__ import annotations

import contextlib
import copy
import io
from dataclasses import dataclass

import numpy as np

# Face height in original-image pixels: what a detector run at native resolution sees.
SIZE_BUCKETS: dict[str, tuple[float, float]] = {
    "all": (0, 1e5),
    "tiny": (0, 16),
    "small": (16, 32),
    "medium": (32, 96),
    "large": (96, 1e5),
}
# Face height as a fraction of frame height. Models with a fixed input (BlazeFace squeezes
# every frame to 128 or 192 px) see faces at this *relative* size, whatever the camera
# resolution -- so this, not pixels, predicts how far from the camera they still work.
RELATIVE_BUCKETS: dict[str, tuple[float, float]] = {
    "all": (0, 1.01),
    "rel<5%": (0, 0.05),
    "rel5-10%": (0.05, 0.10),
    "rel10-20%": (0.10, 0.20),
    "rel>=20%": (0.20, 1.01),
}
# 0.3 is kept as a localisation-tolerant "was the face found at all" diagnostic: box
# conventions differ (MediaPipe's boxes are squarer than WIDER's), which costs IoU
# without being a detection failure. AP[.5:.95] is computed over 0.50..0.95 only.
IOU_THRS = np.array([0.3] + [round(0.5 + 0.05 * i, 2) for i in range(10)])


@dataclass
class EvalResult:
    ap: dict[str, dict[str, float]]  # bucket -> {"AP50", "AP75", "AP", "AP30"}
    images: int
    gt_counts: dict[str, int]
    _eval: object

    def at_threshold(self, score_thr: float, iou: float = 0.5) -> dict[str, dict[str, float]]:
        t = int(np.argmin(np.abs(IOU_THRS - iou)))
        out: dict[str, dict[str, float]] = {}
        E = self._eval
        n_img = len(E.params.imgIds)
        for a, label in enumerate(E.params.areaRngLbl):
            tp = fp = n_gt = 0
            for e in E.evalImgs[a * n_img : (a + 1) * n_img]:
                if e is None:
                    continue
                n_gt += int(np.sum(~np.asarray(e["gtIgnore"], dtype=bool)))
                if not len(e["dtScores"]):
                    continue
                s = np.asarray(e["dtScores"]) >= score_thr
                matched = e["dtMatches"][t] > 0
                ign = e["dtIgnore"][t].astype(bool)
                tp += int(np.sum(s & matched & ~ign))
                fp += int(np.sum(s & ~matched & ~ign))
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / n_gt if n_gt else 0.0
            out[label] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                          "tp": tp, "fp": fp, "fn": n_gt - tp, "fp_per_image": fp / n_img}
        return out

    def best_f1_threshold(self, iou: float = 0.5, bucket: str = "all") -> tuple[float, float]:
        """Score threshold that maximises F1 -- call this on the *dev* split only."""
        best = (0.5, -1.0)
        for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
            f1 = self.at_threshold(float(thr), iou)[bucket]["f1"]
            if f1 > best[1]:
                best = (float(thr), f1)
        return best


def evaluate(gt: dict, detections: list[dict], relative: bool = False) -> EvalResult:
    """``gt`` is a COCO dict; ``detections`` are COCO results (image_id, bbox xywh, score).

    ``relative=False`` buckets by face height in pixels (SIZE_BUCKETS); ``True`` by face
    height / image height (RELATIVE_BUCKETS).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    buckets = RELATIVE_BUCKETS if relative else SIZE_BUCKETS
    img_h = {i["id"]: i["height"] for i in gt["images"]}

    def size(a: dict) -> float:
        return float(a["bbox"][3]) / img_h[a["image_id"]] if relative else float(a["bbox"][3])

    gt = copy.deepcopy(gt)
    for a in gt["annotations"]:
        a["area"] = size(a) ** 2
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        dets = [dict(d, category_id=1) for d in detections]
        coco_dt = coco_gt.loadRes(dets) if dets else _empty_results(coco_gt)
        for a in coco_dt.anns.values():
            a["area"] = size(a) ** 2
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.params.iouThrs = IOU_THRS
        E.params.maxDets = [1000]
        E.params.areaRng = [[lo**2, hi**2] for lo, hi in buckets.values()]
        E.params.areaRngLbl = list(buckets)
        E.evaluate()
        E.accumulate()

    prec = E.eval["precision"]  # [T, R, K, A, M]
    ap: dict[str, dict[str, float]] = {}
    for a, label in enumerate(buckets):
        per_t = []
        for t in range(len(IOU_THRS)):
            p = prec[t, :, 0, a, 0]
            per_t.append(float(np.mean(p[p > -1])) if np.any(p > -1) else float("nan"))
        ap[label] = {"AP30": per_t[0], "AP50": per_t[1], "AP75": per_t[6], "AP": float(np.nanmean(per_t[1:]))}
    gt_counts = {}
    valid = [a for a in gt["annotations"] if not a["iscrowd"]]
    for label, (lo, hi) in buckets.items():
        gt_counts[label] = sum(1 for a in valid if lo <= size(a) < hi)
    return EvalResult(ap, len(gt["images"]), gt_counts, E)


def _empty_results(coco_gt):
    from pycocotools.coco import COCO

    c = COCO()
    c.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"], "annotations": []}
    c.createIndex()
    return c
