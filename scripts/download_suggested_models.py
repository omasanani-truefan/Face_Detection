"""Fetch the optional heavy models used by the "suggested models" experiments.

    python scripts/download_suggested_models.py

* SCRFD-10G (ResNet-class detector)  <- InsightFace ``buffalo_l`` (289 MB archive)
* ArcFace w600k_r50 (ResNet-50)      <- InsightFace ``buffalo_l``
* ArcFace w600k_mbf (MobileFaceNet)  <- InsightFace ``buffalo_sc`` (15 MB archive)

Needed only by ``bench_youtube.py`` and ``bench_suggested_models.py``; the main detectors
come from ``download_models.py``. The archives are checksum-verified, then only the files
the benchmarks read are extracted to ``models/_download/buffalo_l|buffalo_sc/``.

LICENCE: InsightFace pretrained weights are for non-commercial research use only. Do not
ship them in a product.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from download_models import DOWNLOAD_DIR, fetch  # noqa: E402
from facetrack.detectors.base import sha256_file  # noqa: E402

BASE = "https://github.com/deepinsight/insightface/releases/download/model-zoo"
# (archive, sha256 of the archive, files to extract)
PACKS = [
    ("buffalo_l", "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", ["det_10g.onnx", "w600k_r50.onnx"]),
    ("buffalo_sc", "57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72", ["w600k_mbf.onnx"]),
]


def main() -> None:
    for pack, sha, members in PACKS:
        out = DOWNLOAD_DIR / pack
        if all((out / m).exists() for m in members):
            print(f"ok   {pack:12s} {', '.join(members)}")
            continue
        archive = fetch(f"{BASE}/{pack}.zip")
        got = sha256_file(archive)
        if got != sha:
            archive.unlink()
            raise SystemExit(f"checksum mismatch for {pack}.zip: expected {sha}, got {got} (file removed; re-run to download again)")
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for m in members:
                with z.open(m) as r, open(out / m, "wb") as f:
                    f.write(r.read())
        print(f"new  {pack:12s} {', '.join(members)}")


if __name__ == "__main__":
    main()
