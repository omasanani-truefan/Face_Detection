"""Convert WIDER FACE val annotations to COCO-style JSON, split into dev and test.

    python scripts/prepare_widerface.py

* ``dev``  -- used to tune score thresholds and tracker parameters.
* ``test`` -- held out; final numbers are reported on it only.

The split is deterministic and stratified by WIDER's 61 event folders: within each
event, images are ordered by a hash of their file name and the first half goes to dev.
Faces WIDER marks ``invalid`` become ``iscrowd=1`` (ignore regions): a detection on one
is neither a true nor a false positive.

WIDER FACE is distributed for non-commercial research use; we only evaluate on it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "_download" / "widerface"
OUT = ROOT / "data" / "widerface"


def parse_gt(path: Path) -> list[tuple[str, list[list[int]]]]:
    lines = path.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        name = lines[i].strip()
        n = int(lines[i + 1])
        # Images with zero faces still carry one all-zero placeholder row.
        rows = [list(map(int, lines[i + 2 + k].split())) for k in range(max(n, 1))]
        out.append((name, rows if n else []))
        i += 2 + max(n, 1)
    return out


def main() -> None:
    gt_file = RAW / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    img_root = RAW / "WIDER_val" / "images"
    if not gt_file.exists():
        sys.exit(f"missing {gt_file}; download WIDER_val.zip and wider_face_split.zip into {RAW}")
    entries = parse_gt(gt_file)

    by_event: dict[str, list[tuple[str, list[list[int]]]]] = {}
    for name, rows in entries:
        by_event.setdefault(name.split("/")[0], []).append((name, rows))
    split_of: dict[str, str] = {}
    for items in by_event.values():
        items.sort(key=lambda it: hashlib.sha1(it[0].encode()).hexdigest())
        half = len(items) // 2
        for k, (name, _) in enumerate(items):
            split_of[name] = "dev" if k < half else "test"

    cocos = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "face"}],
                 "info": {"source": "WIDER FACE val", "split": s, "image_root": str(img_root.relative_to(ROOT))}}
             for s in ("dev", "test")}
    ann_id = 1
    for img_id, (name, rows) in enumerate(entries, start=1):
        h, w = cv2.imread(str(img_root / name)).shape[:2]
        coco = cocos[split_of[name]]
        coco["images"].append({"id": img_id, "file_name": name, "width": w, "height": h, "event": name.split("/")[0]})
        for x, y, bw, bh, blur, expr, illum, invalid, occl, pose in rows:
            if bw <= 0 or bh <= 0:
                continue
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": 1, "bbox": [x, y, bw, bh], "area": bw * bh,
                "iscrowd": int(invalid), "attributes": {"blur": blur, "expression": expr, "illumination": illum,
                                                         "occlusion": occl, "pose": pose},
            })
            ann_id += 1

    OUT.mkdir(parents=True, exist_ok=True)
    for s, coco in cocos.items():
        (OUT / f"{s}.json").write_text(json.dumps(coco))
        valid = [a for a in coco["annotations"] if not a["iscrowd"]]
        print(f"{s:4s}: {len(coco['images'])} images, {len(valid)} faces (+{len(coco['annotations']) - len(valid)} ignore)")


if __name__ == "__main__":
    main()
