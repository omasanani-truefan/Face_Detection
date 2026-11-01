"""Build the synthetic-motion tracking sequences (see facetrack/bench/sequences.py).

    python scripts/build_sequences.py

Writes ``data/sequences/{dev,test}/<name>.mp4`` + ``<name>.json`` (per-frame GT) and a
``manifest.json`` with checksums. dev sequences are built only from WIDER dev images and
test sequences only from WIDER test images, so nothing used for tuning leaks into the
held-out evaluation. Deterministic for a given seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.bench.sequences import (  # noqa: E402
    Appearance,
    load_actors,
    load_backgrounds,
    render_actor_sequence,
    render_camera_sequence,
    scenarios,
    write_sequence,
)
from facetrack.benchenv import sha256_of  # noqa: E402

FPS = 30.0
LIGHTING_VARIANTS = {"multi_enter": ["low_light", "backlight", "uneven"], "crossing": ["low_light", "backlight", "uneven"],
                     "temporary_occlusion": ["low_light"], "pan_crowd": ["low_light", "backlight", "uneven"]}


def scaled(sc, factor: float, name: str):
    """Same choreography at a lower camera resolution (faces shrink with the frame)."""
    W, H = sc.size
    aps = [replace(a, keys=[(t, x * factor, y * factor, h * factor) for t, x, y, h in a.keys]) for a in sc.appearances]
    return replace(sc, name=name, size=(int(W * factor), int(H * factor)), appearances=aps, tags=sc.tags + ["low_resolution"])


def pick_crowd_images(coco: dict, n: int, rng) -> list[tuple[dict, list[dict]]]:
    """Group photos whose labelled faces are all clearly sized and none invalid. WIDER is
    not exhaustively labelled in dense crowds; there, correct detections of unlabelled
    faces would be scored as false positives."""
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    good = []
    for im in coco["images"]:
        anns = by_img.get(im["id"], [])
        if not 8 <= len(anns) <= 30 or im["width"] < 1000:
            continue
        if all(not a["iscrowd"] and a["bbox"][3] >= 14 for a in anns):
            good.append((im, anns))
    rng.shuffle(good)
    return good[:n]


def build_split(split: str, seed: int, out_root: Path) -> list[dict]:
    coco = json.loads((ROOT / "data/widerface" / f"{split}.json").read_text())
    img_root = ROOT / coco["info"]["image_root"]
    rng = np.random.default_rng(seed)
    scs = scenarios()
    scs.append(scaled(next(s for s in scs if s.name == "far_group"), 0.5, "far_group_lowres"))
    actors = load_actors(coco, img_root, sum(s.n_actors for s in scs), rng)
    backgrounds = load_backgrounds(coco, img_root, len(scs), (640, 480), rng)
    out_dir = out_root / split
    records = []
    a0 = 0
    for i, sc in enumerate(scs):
        cast = actors[a0:a0 + sc.n_actors]
        a0 += sc.n_actors
        meta = {"split": split, "generator": "actors", "scenario": sc.name, "tags": sc.tags, "width": sc.size[0],
                "height": sc.size[1], "synthetic": True, "actors": [a.source for a in cast], "seed": seed + i}
        variants = [None] + LIGHTING_VARIANTS.get(sc.name, [])
        for light in variants:
            name = sc.name if not light else f"{sc.name}__{light}"
            frames = render_actor_sequence(sc, cast, backgrounds[i], FPS, seed + i)
            gt = write_sequence(out_dir, name, frames, FPS, meta, light, seed + i)
            records.append(gt)
            print(f"  {split}/{name}: {gt['num_frames']} frames {sc.size[0]}x{sc.size[1]}", flush=True)

    for j, (im, anns) in enumerate(pick_crowd_images(coco, 2, rng)):
        img = cv2.imread(str(img_root / im["file_name"]))
        Wi, Hi = im["width"], im["height"]
        valid = [a for a in anns if not a["iscrowd"]]
        fcx = float(np.mean([a["bbox"][0] + a["bbox"][2] / 2 for a in valid]))
        fcy = float(np.mean([a["bbox"][1] + a["bbox"][3] / 2 for a in valid]))
        cams = {
            "pan_crowd": ([(0, 0.275 * Wi, fcy, 0.55 * Wi), (8, 0.725 * Wi, fcy, 0.55 * Wi)], ["camera_motion", "enter", "leave", "multi"]),
            "zoom_out_crowd": ([(0, fcx, fcy, 0.35 * Wi), (7, fcx, fcy, 1.0 * Wi)], ["camera_motion", "scale", "far", "multi"]),
        }
        name, (path, tags) = list(cams.items())[j]
        meta = {"split": split, "generator": "camera", "scenario": name, "tags": tags, "width": 640, "height": 360,
                "synthetic": True, "source_image": im["file_name"], "seed": seed + 100 + j}
        for light in [None] + LIGHTING_VARIANTS.get(name, []):
            nm = name if not light else f"{name}__{light}"
            frames = render_camera_sequence(img, anns, path, path[-1][0], FPS, (640, 360))
            gt = write_sequence(out_dir, nm, frames, FPS, meta, light, seed + 100 + j)
            records.append(gt)
            print(f"  {split}/{nm}: {gt['num_frames']} frames 640x360", flush=True)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sequences")
    args = ap.parse_args()
    manifest = []
    for k, split in enumerate(("dev", "test")):
        for gt in build_split(split, args.seed + 1000 * k, args.out):
            v = args.out / split / gt["video"]
            faces = [f for fr in gt["frames"] for f in fr["faces"] if not f["ignore"]]
            manifest.append({"split": split, "name": gt["name"], "video": f"{split}/{gt['video']}", "sha256": sha256_of(v),
                             "gt": f"{split}/{gt['name']}.json", "fps": gt["fps"], "width": gt["width"], "height": gt["height"],
                             "frames": gt["num_frames"], "lighting": gt["lighting"], "tags": gt["tags"],
                             "generator": gt["generator"], "gt_tracks": len({f["track_gt_id"] for f in faces}),
                             "gt_boxes": len(faces), "synthetic": True})
    (args.out / "manifest.json").write_text(json.dumps({"seed": args.seed, "sequences": manifest}, indent=1))
    print(f"wrote {len(manifest)} sequences -> {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
