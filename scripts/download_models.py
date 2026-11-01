"""Fetch every pretrained model in the manifest, verify its checksum, and derive the
dynamic-shape YuNet files.

    python scripts/download_models.py

Downloads land in ``models/_download`` first and only a verified file is copied into
``models/``. Re-running is cheap: files already present with the right hash are skipped.
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facetrack.detectors.base import sha256_file  # noqa: E402
from facetrack.models import MODELS, MODELS_DIR, ModelSpec  # noqa: E402

DOWNLOAD_DIR = MODELS_DIR / "_download"


def fetch(url: str) -> Path:
    dest = DOWNLOAD_DIR / url.rsplit("/", 1)[-1]
    if not dest.exists():
        print(f"  downloading {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(dest)
    return dest


def make_dynamic_yunet(src: Path, dst: Path) -> None:
    """Relax the 1x3x640x640 input (and matching output) shapes to symbolic dims.

    Every Reshape in the YuNet graph uses ``[1, -1, C]``, so the network is already
    spatially dynamic; only the declared shapes pin it. No weights or nodes change.
    """
    import onnx

    m = onnx.load(str(src))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    for i, name in ((2, "height"), (3, "width")):
        dims[i].ClearField("dim_value")
        dims[i].dim_param = name
    for o in m.graph.output:
        d = o.type.tensor_type.shape.dim[1]
        d.ClearField("dim_value")
        d.dim_param = f"anchors_{o.name}"
    del m.graph.value_info[:]
    onnx.save(m, str(dst))


def ensure(key: str, spec: ModelSpec) -> None:
    dst = MODELS_DIR / spec.file
    if spec.derived_from:
        src = MODELS_DIR / MODELS[spec.derived_from].file
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            make_dynamic_yunet(src, dst)
        print(f"ok   {key:32s} (derived from {spec.derived_from})")
        return
    if dst.exists() and sha256_file(dst) == spec.sha256:
        print(f"ok   {key:32s} {spec.sha256[:12]}")
        return
    downloaded = fetch(spec.url)
    if spec.archive_member:
        with zipfile.ZipFile(downloaded) as z:
            member = next(n for n in z.namelist() if n.endswith(spec.archive_member))
            with z.open(member) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
    else:
        shutil.copyfile(downloaded, dst)
    got = sha256_file(dst)
    if got != spec.sha256:
        dst.unlink()
        raise SystemExit(f"checksum mismatch for {key}: expected {spec.sha256}, got {got}")
    print(f"new  {key:32s} {got[:12]}")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    # Base files first so derived ones can find their source.
    for key, spec in sorted(MODELS.items(), key=lambda kv: kv[1].derived_from is not None):
        ensure(key, spec)


if __name__ == "__main__":
    main()
