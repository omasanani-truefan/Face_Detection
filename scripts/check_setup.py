"""Check that this machine can run FaceTrack, and say how to fix whatever is missing.

    python scripts/check_setup.py                    # offline checks only
    python scripts/check_setup.py --youtube          # also try one YouTube lookup (nothing is downloaded)
    python scripts/check_setup.py --youtube URL

Exit code 1 if something required is missing. Optional pieces only warn.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

results: list[tuple[str, str, str]] = []  # (level, what, how to fix)


def add(level: str, what: str, fix: str = "") -> None:
    results.append((level, what, fix))
    print(f"{level:4s} {what}" + (f"\n       -> {fix}" if fix and level != "ok" else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--youtube", nargs="?", const="https://www.youtube.com/watch?v=5nGBIincOIo", metavar="URL")
    args = ap.parse_args()

    v = sys.version_info
    add("ok" if v >= (3, 10) else "FAIL", f"Python {v.major}.{v.minor}.{v.micro}", "need Python 3.10 or newer")
    add("ok" if sys.prefix != sys.base_prefix else "warn", "virtual environment active",
        "python -m venv .venv, activate it, then: python -m pip install --upgrade pip")

    for mod, extra, required in [("numpy", "", True), ("cv2", "", True), ("onnxruntime", "", True), ("yaml", "", True), ("scipy", "", True),
                                 ("mediapipe", "mediapipe", False), ("pycocotools", "bench", False), ("onnx", "bench", False),
                                 ("yt_dlp", "youtube", False), ("matplotlib", "report", False), ("pytest", "dev", False)]:
        try:
            importlib.import_module(mod)
            add("ok", f"python package {mod}")
        except ImportError:
            fix = f'pip install -e ".[{extra}]"' if extra else 'pip install -e "."'
            add("FAIL" if required else "warn", f"python package {mod} missing", fix + ("" if required else f"  (needed for the '{extra}' features)"))

    try:
        from facetrack.models import MODELS, MODELS_DIR

        missing = [s.file for s in MODELS.values() if not (MODELS_DIR / s.file).exists()]
        add("ok" if not missing else "FAIL", "core models present" if not missing else f"core models missing: {', '.join(missing)}",
            "python scripts/download_models.py")
    except ImportError:
        add("FAIL", "facetrack package not importable", 'pip install -e ".[mediapipe,bench,youtube,report,dev]"')
    dl = ROOT / "models" / "_download"
    heavy = [dl / "buffalo_l" / "det_10g.onnx", dl / "buffalo_l" / "w600k_r50.onnx", dl / "buffalo_sc" / "w600k_mbf.onnx"]
    add("ok" if all(p.exists() for p in heavy) else "warn", "suggested heavy models present (SCRFD-10G, ArcFace)",
        "python scripts/download_suggested_models.py   (305 MB; only the benchmark comparison needs them)")

    node = shutil.which("node")
    add("ok" if node else "warn", "Node.js found (browser app)" if node else "Node.js not found", "install Node 20+ from nodejs.org (browser app only)")
    add("ok" if (ROOT / "web" / "node_modules").exists() else "warn", "web dependencies installed", "cd web && npm install")
    chrome = [p for p in (os.environ.get("CHROME_PATH"), "C:/Program Files/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome",
                          "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome") if p and Path(p).exists()]
    add("ok" if chrome else "warn", "Chrome found (PDF report, browser tests)" if chrome else "Chrome not found",
        "install Chrome or set CHROME_PATH; without it the report is written as HTML only")
    cookies = os.environ.get("FACETRACK_YT_COOKIES_BROWSER") or os.environ.get("FACETRACK_YT_COOKIES_FILE")
    add("ok" if cookies else "info", "YouTube cookie option set" if cookies else "YouTube cookies not configured (only needed if YouTube asks to sign in)")

    if args.youtube:
        try:
            from facetrack.download import resolve_stream

            _, info = resolve_stream(args.youtube)
            add("ok", f"YouTube lookup works: {info.get('title')!r}")
        except Exception as e:  # noqa: BLE001  report any failure to the user
            add("FAIL", f"YouTube lookup failed: {e}",
                "set FACETRACK_YT_COOKIES_BROWSER=chrome (or firefox/edge) and retry, or try later; local video files work regardless")

    bad = [r for r in results if r[0] == "FAIL"]
    print(f"\n{'READY' if not bad else 'NOT READY'}: {len(bad)} required problem(s), {sum(r[0] == 'warn' for r in results)} optional warning(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
