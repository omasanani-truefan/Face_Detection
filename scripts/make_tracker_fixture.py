"""Write web/tests/fixtures/tracker_parity.json: detection streams plus the Python
trackers' outputs, which the TypeScript port must reproduce exactly.

    python scripts/make_tracker_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facetrack.tracking import TrackerConfig, create_tracker  # noqa: E402
from facetrack.types import Detection  # noqa: E402


def stream(seed: int, n_frames: int = 160) -> list[dict]:
    """Faces walking with births, exits, dropouts, score dips and irregular frame gaps."""
    rng = np.random.default_rng(seed)
    faces = []
    for k in range(int(rng.integers(2, 6))):
        faces.append({
            "x": rng.uniform(0, 500), "y": rng.uniform(0, 300), "vx": rng.uniform(-6, 6), "vy": rng.uniform(-2, 2),
            "s": rng.uniform(30, 120), "start": int(rng.integers(0, 60)), "end": int(rng.integers(80, n_frames)),
        })
    t, frames = 0.0, []
    for f in range(n_frames):
        t += float(rng.choice([33.3, 33.3, 33.3, 16.7, 66.7, 120.0]))
        dets = []
        for fc in faces:
            if not fc["start"] <= f < fc["end"] or rng.random() < 0.08:
                continue
            x = fc["x"] + fc["vx"] * f + rng.normal(0, 1.5)
            y = fc["y"] + fc["vy"] * f + rng.normal(0, 1.5)
            s = fc["s"] * (1 + rng.normal(0, 0.02))
            conf = float(np.clip(rng.normal(0.8, 0.12), 0.05, 1.0))
            if rng.random() < 0.1:
                conf = float(rng.uniform(0.2, 0.45))  # blur / head turn dip
            dets.append([round(x, 3), round(y, 3), round(s, 3), round(s * 1.2, 3), round(conf, 4)])
        if rng.random() < 0.05:
            dets.append([round(rng.uniform(0, 600), 3), round(rng.uniform(0, 400), 3), 40.0, 48.0, round(float(rng.uniform(0.5, 0.9)), 4)])
        frames.append({"t": round(t, 3), "dets": dets})
    return frames


def main() -> None:
    cases = []
    for seed in range(6):
        frames = stream(seed)
        cfg = dict(det_threshold=0.5, low_det_threshold=0.2, min_hits=3, max_lost_ms=600, tentative_timeout_ms=200)
        expected = {}
        for name in ("iou", "sort", "bytetrack", "sort+lastobs", "bytetrack+lastobs"):
            base, _, variant = name.partition("+")
            trk = create_tracker(base, TrackerConfig(**cfg, lost_match_last_observation=variant == "lastobs"))
            out_frames = []
            for i, fr in enumerate(frames):
                dets = [Detection((d[0], d[1], d[2], d[3]), d[4]) for d in fr["dets"]]
                out = trk.update(dets, fr["t"], i)
                out_frames.append({
                    "visible": [[t.track_id, *[round(float(v), 4) for v in t.bbox_xyxy]] for t in out.visible()],
                    "events": [[e.type.value, e.track_id] for e in out.events if e.type.value != "track_updated"],
                })
            expected[name] = out_frames
        cases.append({"seed": seed, "config": cfg, "frames": frames, "expected": expected})
    dst = ROOT / "web" / "tests" / "fixtures" / "tracker_parity.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(cases))
    print(f"wrote {dst} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
