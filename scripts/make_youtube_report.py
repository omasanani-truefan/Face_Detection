"""Figure for docs/BENCHMARK_REPORT from results/youtube/summary.json.

    python scripts/make_youtube_report.py   # -> docs/images/youtube_benchmark.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "results" / "youtube" / "summary.json").read_text())
vids = list(res["videos"].values())
labels = list(vids[0]["detectors"])
short = {"SCRFD-10G @640 (ResNet-class)": "SCRFD-10G @640 (ResNet-class)"}

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE})

fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08}, sharey=True)
y = np.arange(len(labels))[::-1]

lat = [np.mean([v["detectors"][l]["end_to_end"]["p50_ms"] for v in vids]) for l in labels]
a.barh(y, lat, height=0.62, color=BLUE)
a.axvline(33.3, color=MUTED, lw=1, ls=(0, (4, 3)))
a.text(34.5, y[0] + 0.45, "30 fps budget", color=MUTED, fontsize=7.5, va="center")
for yi, v in zip(y, lat):
    a.text(v + 1.2, yi, f"{v:.1f}", va="center", color=INK, fontsize=8)
a.set_yticks(y, labels)
a.set_xlabel("Median ms per frame (lower is better)")
a.set_xlim(0, 58)
a.set_title("Speed (1280x720 frames, CPU)", loc="left", color=INK, fontsize=9.5, fontweight="bold")

h = 0.34
names = ["Walking street", "Fandom selfie ad"]
for k, (v, c) in enumerate(zip(vids, (BLUE, ORANGE))):
    f1 = [v["detectors"][l]["agreement"]["f1_vs_ref"] for l in labels]
    b.barh(y + (0.17 if k == 0 else -0.17), f1, height=h - 0.02, color=c, label=names[k])
    for yi, x in zip(y, f1):
        b.text(x + 0.012, yi + (0.17 if k == 0 else -0.17), f"{x:.2f}", va="center", color=INK, fontsize=7)
b.set_xlim(0, 1.08)
b.set_xlabel("F1 vs reference (higher is better)")
b.set_title("Agreement with reference (proxy)", loc="left", color=INK, fontsize=9.5, fontweight="bold")
b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=7.5, handlelength=1.0)
for ax in (a, b):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
out = ROOT / "docs" / "images" / "youtube_benchmark.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
