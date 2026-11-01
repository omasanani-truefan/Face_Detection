"""Export the frames the browser benchmark uses (same source frames as the Python
latency benchmark), as JPEGs under web/public/bench-frames/<res>/.

    python scripts/export_bench_frames.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_detection_latency import N_FRAMES, RESOLUTIONS, load_frames  # noqa: E402


def main() -> None:
    out = ROOT / "web" / "public" / "bench-frames"
    index = {}
    for res, wh in RESOLUTIONS.items():
        d = out / res
        d.mkdir(parents=True, exist_ok=True)
        frames = load_frames(wh)[:N_FRAMES]
        names = []
        for i, f in enumerate(frames):
            name = f"{i:03d}.jpg"
            cv2.imwrite(str(d / name), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            names.append(f"bench-frames/{res}/{name}")
        index[res] = names
    (out / "index.json").write_text(json.dumps(index))
    print(f"wrote {sum(len(v) for v in index.values())} frames to {out}")


if __name__ == "__main__":
    main()
