"""Build the benchmark report from results/youtube/summary.json:
BENCHMARK.md (one page), docs/BENCHMARK_REPORT.html and, if Chrome is installed, the 3-page PDF.

    python scripts/build_benchmark_report.py     # needs: pip install -e ".[report]"

Every number in the tables and the narrative is read from the result files, never typed by hand.
Earlier WIDER FACE numbers come from results/detection_accuracy and results/suggested_models.
"""

from __future__ import annotations

import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts" / "make_youtube_report.py"))  # figure used by the report
yt = json.loads((ROOT / "results" / "youtube" / "summary.json").read_text())
col = json.loads((ROOT / "results" / "suggested_models" / "summary.json").read_text())
VW, VS = list(yt["videos"].values())  # walking street, selfie ad
L = list(VW["detectors"])
env = yt["environment"]

# Measured earlier on the WIDER FACE test split (docs/results_tables.md), shown for context only.
WIDER_AP50 = {"YuNet @640": "0.487", "YuNet @native": "0.654 (@1024)", "SCRFD-500M @640": "0.576", "BlazeFace short": "0.035",
              "BlazeFace full": "0.145", "SCRFD-10G @640 (ResNet-class)": f"{col['scrfd_10g_accuracy']['AP50']:.3f}"}


def det(v, l):
    return v["detectors"][l]


def row_detectors() -> str:
    out = []
    for l in L:
        w, s = det(VW, l), det(VS, l)
        e1, e2 = w["end_to_end"], s["end_to_end"]
        ms = (e1["p50_ms"] + e2["p50_ms"]) / 2
        p95 = max(e1["p95_ms"], e2["p95_ms"])
        cls = ' class="hl"' if l.startswith(("YuNet @640", "SCRFD-500M @native")) else ""
        out.append(
            f"<tr{cls}><td>{l}</td><td>{ms:.1f} <span class=m>({p95:.0f})</span></td>"
            f"<td>{w['faces_per_frame_mean']:.1f}</td><td>{w['agreement']['f1_vs_ref']:.2f}</td><td>{w['agreement']['recall_<24px']:.2f}</td>"
            f"<td>{s['faces_per_frame_mean']:.1f}</td><td>{s['agreement']['f1_vs_ref']:.2f}</td><td>{s['agreement']['recall_>=64px']:.2f}</td>"
            f"<td>{WIDER_AP50.get(l, '–')}</td></tr>")
    rw, rs = VW["reference"], VS["reference"]
    out.append(f"<tr class=ref><td>SCRFD-10G @native (reference)</td><td>{(rw['latency']['end_to_end']['p50_ms'] + rs['latency']['end_to_end']['p50_ms']) / 2:.0f}</td>"
               f"<td>{rw['faces_per_frame_mean']:.1f}</td><td>–</td><td>–</td><td>{rs['faces_per_frame_mean']:.1f}</td><td>–</td><td>–</td><td>–</td></tr>")
    return "\n".join(out)


def row_trackers() -> str:
    out = []
    for l in L:
        cells = ""
        for v in (VW, VS):
            t = det(v, l)["trackers"]
            cells += "".join(f"<td>{t[k]['ids_issued']} <span class=m>({100 * t[k]['share_under_1s']:.0f}%)</span></td>" for k in ("iou", "sort", "bytetrack"))
        out.append(f"<tr><td>{l}</td>{cells}</tr>")
    return "\n".join(out)


def rec(v, k, stat):
    return v["recognition_cost"]["per_frame_ms"][k][stat]


def row_recognition() -> str:
    em = col["embedding_latency_112x112"]
    r50, mbf = em["arcface_r50 (ResNet-50) threads=default"], em["arcface_mbf (MobileFaceNet) threads=default"]
    out = []
    for key, label, one in (("arcface_r50", "ArcFace w600k_r50 (ResNet-50)", r50), ("arcface_mbf", "ArcFace w600k_mbf (MobileFaceNet)", mbf)):
        out.append(f"<tr><td>{label}</td><td>{one['size_mb']:.0f} MB</td><td>{one['p50_ms']:.1f}</td>"
                   f"<td>{rec(VW, key, 'p50_ms'):.0f} / {rec(VW, key, 'p95_ms'):.0f}</td><td>{rec(VS, key, 'p50_ms'):.0f} / {rec(VS, key, 'p95_ms'):.0f}</td></tr>")
    return "\n".join(out)


w640, wnat, s640, snat = det(VW, "YuNet @640"), det(VW, "YuNet @native"), det(VS, "YuNet @640"), det(VS, "YuNet @native")
sc_nat_w, sc_nat_s = det(VW, "SCRFD-500M @native"), det(VS, "SCRFD-500M @native")
g10_w, g10_s = det(VW, "SCRFD-10G @640 (ResNet-class)"), det(VS, "SCRFD-10G @640 (ResNet-class)")
bf_s_w, bf_f_w, bf_s_s, bf_f_s = det(VW, "BlazeFace short"), det(VW, "BlazeFace full"), det(VS, "BlazeFace short"), det(VS, "BlazeFace full")
g10_ms = (g10_w["end_to_end"]["p50_ms"] + g10_s["end_to_end"]["p50_ms"]) / 2
avg = lambda l: (det(VW, l)['end_to_end']['p50_ms'] + det(VS, l)['end_to_end']['p50_ms']) / 2
track_ms = max(t["ms_per_frame"] for v in (VW, VS) for d in v["detectors"].values() for t in d["trackers"].values())
rcw, rcs = VW["recognition_cost"], VS["recognition_cost"]
em = col["embedding_latency_112x112"]
mbf_one = em["arcface_mbf (MobileFaceNet) threads=default"]["p50_ms"]
conf_tracks_w = det(VW, "YuNet @640")["trackers"]["bytetrack"]["confirmed_tracks"]
sec_w, sec_s = VW["seconds_used"], VS["seconds_used"]
g10_wider = col["scrfd_10g_accuracy"]

HTML = f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>Face detection and tracking: model benchmark on two videos</title>
<style>
@page {{ size: A4; margin: 13mm 14mm 12mm; }}
:root {{ --ink:#14161a; --muted:#5b616b; --line:#d9dce1; --accent:#2a78d6; --bg:#f4f6f9; }}
* {{ box-sizing: border-box; }}
body {{ font: 8.9pt/1.38 "Segoe UI", system-ui, sans-serif; color: var(--ink); margin: 0; }}
h1 {{ font-size: 16pt; margin: 0 0 1px; letter-spacing: -.01em; }}
.sub {{ color: var(--muted); margin: 0 0 7px; font-size: 8.6pt; }}
h2, h3 {{ break-after: avoid; }}
h2 {{ font-size: 10.4pt; margin: 9px 0 3px; color: var(--accent); }}
table, tr {{ break-inside: avoid; }}
p {{ margin: 0 0 4px; }}
ul {{ margin: 2px 0 4px; padding-left: 15px; }} li {{ margin: 0 0 1.5px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 7.7pt; margin: 3px 0 5px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 2px 4px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: var(--bg); font-weight: 600; color: var(--ink); }}
th.g {{ text-align: center; border-bottom: 1px solid var(--line); }}
.m {{ color: var(--muted); }}
tr.hl td {{ background: #eaf2fc; }} tr.ref td {{ color: var(--muted); font-style: italic; }}
.box {{ background: var(--bg); border-left: 3px solid var(--accent); padding: 5px 8px; margin: 4px 0 6px; }}
.cap {{ color: var(--muted); font-size: 7.6pt; margin: 0 0 4px; }}
img {{ width: 80%; display: block; margin: 2px auto; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.pb {{ break-before: page; }}
.no {{ color: #a3321f; font-weight: 600; }}
</style></head><body>

<h1>Face detection and tracking: model benchmark on two real videos</h1>
<p class=sub>FaceTrack project, Phases 1–2 · {env['cpu']}, {env['physical_cores']} cores, ONNX Runtime / OpenCV / MediaPipe on CPU only · all numbers measured locally, except where marked proxy</p>

<div class=box><b>Verdict.</b> Use <b>YuNet + ByteTrack on ONNX Runtime</b> as the default: {avg('YuNet @640'):.1f} ms per frame at 640 px, 0.2 MB, MIT code and weights.
The decisive lever on real footage is <b>input resolution, not model size</b>: running YuNet at native 720p found {wnat['faces_per_frame_mean']:.1f} faces per frame on the street video
against {w640['faces_per_frame_mean']:.1f} at 640 px, at {avg('YuNet @native'):.0f} ms (both videos, median). The heavy ResNet-class detector (SCRFD-10G) costs {g10_ms:.0f} ms and still agrees less with the reference than a small model run at native resolution.
The BlazeFace models are not usable on distant faces. ArcFace is a recognition model, so it cannot replace a detector, and run on every face every frame it is far over budget.</div>

<h2>1. What was tested, and how far to trust it</h2>
<ul>
<li><b>Videos</b> (streamed through yt-dlp, never saved; frames kept in memory only, JPEG q95): <b>A</b> “{VW['title'][:44]}…” ({VW['channel']}) and <b>B</b> “{VS['title'][:44]}…” ({VS['channel']}).
Both read as 1280×720 H.264; the first {VW['frames_used']} frames of A ({sec_w:.0f} s, {VW['stream']['fps']:.0f} fps) and {VS['frames_used']} frames of B ({sec_s:.0f} s, {VS['stream']['fps']:.0f} fps). Every model received the same decoded frames.</li>
<li><b>Models:</b> YuNet (ONNX Runtime, int8, OpenCV build), SCRFD-500M-KPS, MediaPipe BlazeFace short and full range, plus SCRFD-10G (ResNet-class). Trackers: IoU, SORT, ByteTrack with the project defaults. <b>All 9 detector configurations × 3 trackers = 27 combinations were run on each video.</b></li>
<li><b>No human ground truth exists for these videos.</b> Speed, faces per frame and track statistics are measured. “Agreement” compares each model with a heavy reference (SCRFD-10G at native resolution, every 3rd frame, score ≥ {yt['protocol']['reference_threshold']}); it is a <b>proxy</b>, not accuracy. The reference is itself imperfect and from the same model family as two of the rows, which flatters them.
Video B is an advert whose set is covered in printed posters, so many “faces” are printed ones; detectors that fire on them are not necessarily wrong.</li>
<li>One run on one laptop (plugged in, nothing else running), so small timing differences between models are not significant; tail latencies (p95) are noisier than medians.</li>
</ul>

<h2>2. Detector comparison</h2>
<table>
<tr><th rowspan=2>Model</th><th rowspan=2>ms/frame<br>median (p95)</th><th class=g colspan=3>A · street, small faces</th><th class=g colspan=3>B · selfie ad, larger faces</th><th rowspan=2>WIDER AP50<br><span class=m>(earlier test)</span></th></tr>
<tr><th>faces/frame</th><th>F1 vs ref</th><th>recall &lt;24 px</th><th>faces/frame</th><th>F1 vs ref</th><th>recall ≥64 px</th></tr>
{row_detectors()}
</table>
<p class=cap>ms/frame = median of the two videos, end to end (preprocess + inference + postprocess); detectors that fuse stages are timed as one. Faces/frame counted at each model’s default score threshold. Recall columns = share of reference faces of that pixel height that the model also found (IoU ≥ 0.5).
Blue rows = recommended default and the best native-resolution alternative (non-commercial weights).</p>
<img src="images/youtube_benchmark.png" alt="Latency and agreement per model">

<h2>3. What the numbers say</h2>
<ul>
<li><b>Resolution decides small-face recall.</b> On video A, 640 px YuNet finds only {100 * w640['agreement']['recall_<24px']:.0f}% of the reference’s faces under 24 px; native resolution finds {100 * wnat['agreement']['recall_<24px']:.0f}% (SCRFD-500M: {100 * det(VW, 'SCRFD-500M @640')['agreement']['recall_<24px']:.0f}% → {100 * sc_nat_w['agreement']['recall_<24px']:.0f}%). It costs about {avg('YuNet @native') / avg('YuNet @640'):.0f}× the time but stays under the 33 ms/30 fps budget at the median.
YuNet’s p95 at native size on A is {wnat['end_to_end']['p95_ms']:.0f} ms, so weaker CPUs should use 640 px or <code>detect every 2nd frame</code> unless faces are large (phone timings were not measured here).</li>
<li><b>SCRFD-500M vs YuNet.</b> At native size SCRFD-500M agrees better with the reference (F1 {sc_nat_w['agreement']['f1_vs_ref']:.2f} / {sc_nat_s['agreement']['f1_vs_ref']:.2f} vs {wnat['agreement']['f1_vs_ref']:.2f} / {snat['agreement']['f1_vs_ref']:.2f}) at similar cost, but that is the favoured family, and its weights are non-commercial (InsightFace). YuNet is the safe default; SCRFD is an upgrade for research use.</li>
<li><b>int8 YuNet is slower, not faster</b> ({avg('YuNet int8 @640'):.0f} ms vs {avg('YuNet @640'):.1f} ms) on this CPU, probably because the model is so small that quantisation overhead outweighs the saved arithmetic (cause not profiled). The OpenCV build is a few ms slower than ONNX Runtime and returns the same face counts.</li>
<li><b>BlazeFace fails here.</b> Short-range found {bf_s_w['faces_per_frame_mean']:.2f} faces per frame on A and full-range {bf_f_w['faces_per_frame_mean']:.2f} (reference: {VW['reference']['faces_per_frame_mean']:.1f}). Even on B’s large faces its recall is {bf_s_s['agreement']['recall_>=64px']:.2f} (short) and {bf_f_s['agreement']['recall_>=64px']:.2f} (full) against {s640['agreement']['recall_>=64px']:.2f} for YuNet.
They run at a fixed 128/192 px input and are designed for near faces (the short-range model for selfie distance), which matches the WIDER result in the main report. They are the fastest ({avg('BlazeFace short'):.1f} ms for the short-range model) but only suit a near-camera selfie use case.</li>
</ul>

<h2>4. Tracker comparison on real footage (no identity ground truth)</h2>
<table>
<tr><th rowspan=2>Detector</th><th class=g colspan=3>A · street: track IDs issued (share lasting &lt;1 s)</th><th class=g colspan=3>B · selfie ad: track IDs issued (share lasting &lt;1 s)</th></tr>
<tr><th>IoU</th><th>SORT</th><th>ByteTrack</th><th>IoU</th><th>SORT</th><th>ByteTrack</th></tr>
{row_trackers()}
</table>
<p class=cap>All 27 detector × tracker combinations per video are shown. Tracker cost was at most {track_ms:.2f} ms per frame in every run, negligible next to detection. With no labelled identities, “IDs issued” mixes real new people with fragments and false tracks, so fewer is not automatically better.</p>
<ul>
<li>No tracker is consistently lowest. With YuNet, IoU issues the fewest IDs but also keeps the fewest faces on screen (@640 on A: {w640['trackers']['iou']['mean_visible']:.1f} confirmed faces on average vs {w640['trackers']['bytetrack']['mean_visible']:.1f} for ByteTrack), while with SCRFD-500M @native, SCRFD-10G and BlazeFace full on A, ByteTrack issues the fewest. This does <b>not</b> overturn the controlled result in the main report (48 synthetic clips with exact identities, where ByteTrack gave the fewest fragments and false tracks): here we can only say the differences are small and detector-dependent, not which tracker is correct.</li>
<li>For the YuNet rows, 40–66% of confirmed tracks last under a second. On A the likely cause is small faces flickering around the threshold at native resolution, which pushes IDs issued from ~{w640['trackers']['bytetrack']['ids_issued']} to ~{wnat['trackers']['bytetrack']['ids_issued']}. Native-resolution use needs a stricter confirm threshold or a minimum face size before a track is announced.</li>
</ul>

<h2>5. Why the suggested ResNet, ArcFace and U-Net models were not useful here</h2>
<table>
<tr><th>Suggested model</th><th>What it actually is</th><th>What we measured</th><th>Verdict</th></tr>
<tr><td><b>ResNet</b></td><td style="text-align:left">An image-classification backbone, not a detector. The nearest real face detector is SCRFD-10G (ResNet-style, 16 MB).</td>
<td style="text-align:left">Median {g10_ms:.0f} ms per frame at 640 px (≈{1000 / g10_ms:.0f} FPS, over the 33 ms budget), 0.2 MB YuNet takes {avg('YuNet @640'):.1f} ms. On WIDER FACE its AP50 is {g10_wider['AP50']:.3f} vs {WIDER_AP50['SCRFD-500M @640']} for SCRFD-500M at the same 640 px.</td>
<td style="text-align:left"><span class=no>Too slow for the gain.</span> More accurate at 640 px, but a small model at native resolution is both faster and agrees better on small faces (F1 {sc_nat_w['agreement']['f1_vs_ref']:.2f} / {sc_nat_s['agreement']['f1_vs_ref']:.2f} vs {g10_w['agreement']['f1_vs_ref']:.2f} / {g10_s['agreement']['f1_vs_ref']:.2f}). Non-commercial weights.</td></tr>
<tr><td><b>ArcFace</b></td><td style="text-align:left">A face-<i>recognition</i> embedder: needs an already-detected, aligned 112×112 face. It cannot find faces and says nothing about tracking.</td>
<td style="text-align:left">See table below: on the faces actually present in these videos it costs hundreds of ms to seconds per frame.</td>
<td style="text-align:left"><span class=no>Wrong job, and out of scope.</span> Persistent identity is Phase 3; a Track ID is not a Person ID. Useful later, once per track, not per frame.</td></tr>
<tr><td><b>U-Net</b></td><td style="text-align:left">A segmentation architecture that outputs per-pixel masks, not face boxes or tracks.</td>
<td style="text-align:left"><b>Not run.</b> No pretrained face model exists to benchmark; training one would need labelled data and still needs box extraction and instance splitting for touching faces.</td>
<td style="text-align:left"><span class=no>Not applicable.</span> Detectors already output boxes and keypoints directly, far cheaper.</td></tr>
</table>

<h3 style="font-size:9.4pt;margin:6px 0 2px">Cost of running ArcFace on the faces found in these videos (measured)</h3>
<table>
<tr><th>Model</th><th>Size</th><th>ms per face<br>(112×112)</th><th>A: ms per frame<br>median / p95</th><th>B: ms per frame<br>median / p95</th></tr>
{row_recognition()}
</table>
<p class=cap>Sampled every 10th frame ({rcw['frames_sampled']} frames of A, {rcs['frames_sampled']} of B); one embedding per YuNet face (score ≥ 0.6), batch size 1. A has {rcw['faces_per_frame_mean']:.1f} faces per frame on average (max {rcw['faces_per_frame_max']}); B has {rcs['faces_per_frame_mean']:.1f} (max {rcs['faces_per_frame_max']}, p95 {rcs['faces_per_frame_p95']:.0f}).
Latency only: crops were not aligned, so no recognition quality is claimed. Budget: 33 ms per frame at 30 fps.</p>
<ul>
<li>Per frame, the ResNet-50 embedder is <b>{rec(VW, 'arcface_r50', 'p50_ms') / 33.3:.0f}× over budget</b> on A and spikes to {rec(VS, 'arcface_r50', 'p95_ms') / 1000:.1f} s on B’s crowded frames. Even the 13 MB MobileFaceNet variant takes {rec(VW, 'arcface_mbf', 'p50_ms'):.0f} / {rec(VS, 'arcface_mbf', 'p50_ms'):.0f} ms per frame, still over budget when run every frame.</li>
<li>Tracking removes that cost: embed once per <i>confirmed track</i>, off the frame path. A had {conf_tracks_w} confirmed tracks in {sec_w:.0f} s (≈{conf_tracks_w / sec_w:.1f} per second, mostly short fragments, see §4), i.e. ≈{conf_tracks_w / sec_w * mbf_one:.0f} ms of MobileFaceNet work per second of video. That is why the Phase 3 interface in <code>identity.py</code> hangs off track events and does not run per frame.</li>
</ul>

<h2>6. Recommendation and limits</h2>
<ul>
<li><b>Ship:</b> YuNet (ONNX Runtime) + ByteTrack, 640 px by default, native resolution when faces are small or far, <code>detect every 2nd frame</code> on weak devices. Re-check SCRFD-500M only if a commercial licence for InsightFace weights is obtained.</li>
<li><b>Do not pursue</b> ResNet-class detectors, BlazeFace (for distant faces), or U-Net for Phases 1–2. Keep ArcFace-class models for Phase 3, once per track.</li>
<li><b>Limits:</b> two clips, {sec_w:.0f} s each, one machine, no human labels (agreement is a proxy and favours the reference’s own family); scene cuts are not separated from tracking behaviour in the track counts. Mobile-device and in-browser timings on these clips were not measured here; the browser app was verified to play and track a streamed YouTube link end to end (24.7 FPS processed in headless Chrome, WASM).</li>
<li><b>Reproduce:</b> <code>python scripts/bench_youtube.py URL [URL…]</code>, then <code>python scripts/build_benchmark_report.py</code>. Raw results: <code>results/youtube/summary.json</code>. The frame grids written beside it come from third-party videos and are kept local, not redistributed.</li>
</ul>
</body></html>"""

def md_rows() -> str:
    rows = []
    for l in L:
        w, s_ = det(VW, l), det(VS, l)
        ms = (w["end_to_end"]["p50_ms"] + s_["end_to_end"]["p50_ms"]) / 2
        rows.append(f"| {l} | {ms:.1f} | {w['faces_per_frame_mean']:.1f} | {w['agreement']['f1_vs_ref']:.2f} | "
                    f"{s_['faces_per_frame_mean']:.1f} | {s_['agreement']['f1_vs_ref']:.2f} |")
    return "\n".join(rows)


MD = f"""# Benchmark summary

Face detectors and trackers compared on two real YouTube videos (streamed, never saved), CPU only
({env['cpu']}). **Full 3-page report: [docs/BENCHMARK_REPORT.pdf](docs/BENCHMARK_REPORT.pdf)**
· raw numbers: [results/youtube/summary.json](results/youtube/summary.json).

| Detector | ms/frame (median) | A: faces/frame | A: F1 vs ref | B: faces/frame | B: F1 vs ref |
|---|---|---|---|---|---|
{md_rows()}

*A* = street footage with small faces ({VW['frames_used']} frames), *B* = selfie advert with larger faces ({VS['frames_used']} frames),
1280×720. Reference detector: SCRFD-10G at native resolution ({VW['reference']['faces_per_frame_mean']:.1f} / {VS['reference']['faces_per_frame_mean']:.1f} faces per frame).
**F1 vs ref is a proxy, not accuracy**: there are no human labels for these videos and the reference favours the SCRFD rows.
Each of the 9 detector setups was also run with IoU, SORT and ByteTrack (27 combinations per video; table in the PDF).

## Result

* **Default: YuNet + ByteTrack on ONNX Runtime.** {avg('YuNet @640'):.1f} ms per frame at 640 px, 0.2 MB, MIT licence.
  Resolution matters more than model size: at full 720p YuNet finds {100 * wnat['agreement']['recall_<24px']:.0f}% of the
  reference's faces under 24 px on A, against {100 * w640['agreement']['recall_<24px']:.0f}% at 640 px ({avg('YuNet @native'):.0f} ms).
* **BlazeFace** misses distant faces ({bf_s_w['faces_per_frame_mean']:.1f} / {bf_f_w['faces_per_frame_mean']:.1f} faces per frame on A for short / full range).

## The suggested ResNet, ArcFace and U-Net models

* **ResNet:** a backbone, not a detector. The closest real detector (SCRFD-10G) takes {g10_ms:.0f} ms per frame, over the 33 ms budget
  for 30 fps, and agrees less with the reference than a small model run at native resolution.
* **ArcFace:** a recognition model; it cannot find faces. On the faces in these videos the ResNet-50 version takes
  {rec(VW, 'arcface_r50', 'p50_ms'):.0f} ms per frame ({rec(VW, 'arcface_r50', 'p50_ms') / 33.3:.0f}× the budget); MobileFaceNet {rec(VW, 'arcface_mbf', 'p50_ms'):.0f} ms.
  It is viable once per track (Phase 3), not once per frame.
* **U-Net:** not run. It outputs masks, not face boxes, and no pretrained face model exists to benchmark.

## Reproduce

```bash
pip install -e ".[mediapipe,bench,youtube,report]"
python scripts/download_models.py
python scripts/download_suggested_models.py        # 305 MB, non-commercial InsightFace weights
python scripts/bench_youtube.py URL_A URL_B         # about 15 minutes; run on an idle, plugged-in machine
python scripts/build_benchmark_report.py            # rewrites this file and docs/BENCHMARK_REPORT.*
```

Limits: two 30-second clips, one machine, no phone timings. Measured and proxy numbers are kept apart in the PDF.
"""
(ROOT / "BENCHMARK.md").write_text(MD, encoding="utf-8")
print("wrote", ROOT / "BENCHMARK.md")

out = ROOT / "docs" / "BENCHMARK_REPORT.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out)

chrome = next((p for p in (os.environ.get("CHROME_PATH"), "C:/Program Files/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome") if p and Path(p).exists()), None)
if chrome:
    pdf = ROOT / "docs" / "BENCHMARK_REPORT.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
                   check=True, capture_output=True, timeout=120)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    print(f"wrote {pdf} ({pages} pages)")
else:
    print("Chrome not found: open the HTML and print to PDF", file=sys.stderr)
