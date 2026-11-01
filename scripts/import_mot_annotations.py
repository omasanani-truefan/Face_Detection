"""Import a manually corrected MOT-format annotation (e.g. CVAT "MOT 1.1" export) for a
recorded video into the benchmark's ground-truth format, and register it in
data/sequences/manifest.json so bench_tracking.py evaluates it next to the synthetic set.

    python scripts/import_mot_annotations.py --video rec/desk_3people.mp4 --gt rec/gt.txt \
        --split test --tags close multi enter --device "Pixel 7 front" --distance-m 1.5

Workflow (docs/DATA_PROTOCOL.md): pre-annotate with `facetrack-demo --export-mot`,
correct boxes and identities in CVAT, export MOT 1.1, import here. MOT rows are
``frame,id,x,y,w,h,flag,class,visibility`` (1-based frames); rows with flag 0 or
visibility < 0.5 become ignore regions. Timestamps come from the video itself.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.benchenv import sha256_of  # noqa: E402
from facetrack.sources import VideoFileSource  # noqa: E402

SEQ = ROOT / "data" / "sequences"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="MOT-format gt.txt")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--name", help="sequence name (default: video stem)")
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--device", default="unknown", help="camera / phone model and which camera")
    ap.add_argument("--distance-m", type=float, help="approximate subject distance")
    ap.add_argument("--lighting", default="normal")
    ap.add_argument("--consent-ref", required=True, help="reference to the signed consent record (never the form itself)")
    ap.add_argument("--seq-root", type=Path, default=SEQ, help=argparse.SUPPRESS)
    args = ap.parse_args()

    name = args.name or args.video.stem
    rows: dict[int, list] = {}
    for line in args.gt.read_text().splitlines():
        if not line.strip():
            continue
        p = [float(v) for v in line.split(",")]
        frame, tid, x, y, w, h = int(p[0]), int(p[1]), *p[2:6]
        flag = p[6] if len(p) > 6 else 1
        vis = p[8] if len(p) > 8 and p[8] >= 0 else 1.0  # -1 = unknown (pre-annotations)
        rows.setdefault(frame - 1, []).append({"track_gt_id": tid, "person_id": f"{name}-p{tid}", "bbox": [x, y, w, h],
                                               "visibility": vis, "ignore": bool(flag == 0 or vis < 0.5 or h < 10)})
    frames = []
    with VideoFileSource(args.video) as src:
        info = src.info
        for f in src:
            frames.append({"frame_id": f.frame_id, "timestamp_ms": round(f.timestamp_ms, 3), "faces": rows.get(f.frame_id, [])})
    out = args.seq_root / args.split
    out.mkdir(parents=True, exist_ok=True)
    video_dst = out / f"{name}{args.video.suffix.lower()}"
    shutil.copyfile(args.video, video_dst)
    gt = {"split": args.split, "generator": "recorded", "scenario": name, "tags": args.tags, "width": info.width,
          "height": info.height, "synthetic": False, "device": args.device, "distance_m": args.distance_m,
          "consent_ref": args.consent_ref, "name": name, "video": video_dst.name, "fps": info.fps,
          "lighting": args.lighting, "num_frames": len(frames), "frames": frames}
    (out / f"{name}.json").write_text(json.dumps(gt))
    manifest_path = args.seq_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sequences": []}
    manifest["sequences"] = [s for s in manifest["sequences"] if not (s["split"] == args.split and s["name"] == name)]
    faces = [f for fr in frames for f in fr["faces"] if not f["ignore"]]
    manifest["sequences"].append({"split": args.split, "name": name, "video": f"{args.split}/{video_dst.name}",
                                  "sha256": sha256_of(video_dst), "gt": f"{args.split}/{name}.json", "fps": info.fps,
                                  "width": info.width, "height": info.height, "frames": len(frames), "lighting": args.lighting,
                                  "tags": args.tags, "generator": "recorded", "gt_tracks": len({f["track_gt_id"] for f in faces}),
                                  "gt_boxes": len(faces), "synthetic": False, "device": args.device})
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"imported {name}: {len(frames)} frames, {len(faces)} GT boxes -> {args.split}; delete "
          f"results/tracking/detections/*/{args.split}/{name}.json.gz to force re-detection after edits")


if __name__ == "__main__":
    main()
