"""Transcode a few test sequences to browser-playable H.264 MP4 (and one Y4M for
Chrome's fake camera), so the web app can be tried immediately and smoke-tested.

    python scripts/make_web_samples.py

OpenCV's bundled writer only produces MPEG-4 Part 2 ('mp4v'), which desktop Chrome
cannot decode; real phone/camera recordings are H.264/HEVC and play natively. Uses the
static ffmpeg shipped in the ``imageio-ffmpeg`` wheel (dev-only dependency).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
SEQ = ROOT / "data" / "sequences" / "test"
OUT = ROOT / "data" / "samples_h264"
SAMPLES = ["reentry", "crossing", "multi_enter__low_light", "temporary_occlusion", "far_group", "pan_crowd"]


def main() -> None:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in SAMPLES:
        dst = OUT / f"{name}.mp4"
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(SEQ / f"{name}.mp4"), "-c:v", "libx264", "-preset", "medium",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
        print(f"wrote {dst}")
    # Chrome's --use-file-for-fake-video-capture accepts Y4M: a camera that shows real faces.
    y4m = OUT / "fake_camera_multi_enter.y4m"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(SEQ / "multi_enter.mp4"), "-pix_fmt", "yuv420p", str(y4m)], check=True)
    print(f"wrote {y4m}")


if __name__ == "__main__":
    main()
