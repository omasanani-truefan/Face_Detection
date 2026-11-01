"""Render every benchmark summary JSON into markdown tables (docs/results_tables.md).

    python scripts/make_report_tables.py

The report quotes these tables; regenerating them after a re-run keeps the numbers in
the docs identical to the measurements, with no hand transcription.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def load(p: str):
    f = RES / p
    return json.loads(f.read_text()) if f.exists() else None


def table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


f2 = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731
f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
f1 = lambda x: "–" if x is None else f"{x:.1f}"  # noqa: E731
f0 = lambda x: "–" if x is None else f"{x:.0f}"  # noqa: E731


def env_line(env: dict) -> str:
    p = env.get("packages", {})
    return (f"{env['cpu']} ({env['physical_cores']}C/{env['logical_cores']}T), {env['ram_gb']} GB RAM, {env['os']}, "
            f"Python {env['python']}, onnxruntime {p.get('onnxruntime')}, opencv {p.get('opencv-python')}, "
            f"mediapipe {p.get('mediapipe')}; {env['accelerator']}; on mains power: {env.get('power_plugged')}")


def detection_accuracy(s: dict) -> str:
    out = ["## Detection accuracy — WIDER FACE val, held-out test split", "",
           f"Protocol: {s['protocol']}. Dev {s['dataset']['dev_images']} images / test {s['dataset']['test_images']} images. "
           f"Size buckets by face height in pixels: {', '.join(f'{k} {v[0]:g}–{v[1]:g}' for k, v in s['dataset']['size_buckets_px_height'].items() if k != 'all')}.",
           f"Hardware: {env_line(s['environment'])}", ""]
    rows = []
    for r in s["results"]:
        a, o, o30 = r["test_ap"]["all"], r["test_at_threshold_iou50"], r["test_at_threshold_iou30"]["all"]
        rows.append([r["config"], f2(r["threshold_from_dev"]), f3(a["AP50"]), f3(a["AP75"]), f3(a["AP"]), f3(a["AP30"]),
                     f3(o["all"]["precision"]), f3(o["all"]["recall"]), f3(o["all"]["f1"]), f2(o["all"]["fp_per_image"]), f3(o30["recall"])])
    out.append(table(["config", "thr (dev)", "AP50", "AP75", "AP[.5:.95]", "AP@0.3", "P", "R", "F1", "FP/img", "R@IoU0.3"], rows))
    out += ["", "### Recall by face height (pixels), at the dev threshold, IoU 0.5 (IoU 0.3 in brackets)", ""]
    rows = []
    for r in s["results"]:
        o, o30 = r["test_at_threshold_iou50"], r["test_at_threshold_iou30"]
        rows.append([r["config"]] + [f"{o[b]['recall']:.3f} ({o30[b]['recall']:.3f})" for b in ("tiny", "small", "medium", "large")])
    c = s["results"][0]["test_gt_counts"]
    out.append(table(["config"] + [f"{b} (n={c[b]})" for b in ("tiny", "small", "medium", "large")], rows))
    out += ["", "### Recall by face height as a fraction of frame height (predicts far-range behaviour), IoU 0.5 (0.3)", ""]
    rows = []
    rb = ["rel<5%", "rel5-10%", "rel10-20%", "rel>=20%"]
    for r in s["results"]:
        o, o30 = r["test_relative_at_threshold_iou50"], r["test_relative_at_threshold_iou30"]
        rows.append([r["config"]] + [f"{o[b]['recall']:.3f} ({o30[b]['recall']:.3f})" for b in rb])
    c = s["results"][0]["test_relative_gt_counts"]
    out.append(table(["config"] + [f"{b} (n={c[b]})" for b in rb], rows))
    return "\n".join(out)


def detection_latency(s: dict) -> str:
    out = ["## Detector latency and resources — Python, CPU only", "",
           f"Protocol: {json.dumps(s['protocol'])}", f"Hardware: {env_line(s['environment'])}", "",
           "Per-stage values are p50 ms. *fused* = the runtime does pre/post-processing inside one call, so only the total is "
           "observable. CPU = average cores busy during the timed loop (1.00 = one core fully used).", ""]
    rows = []
    for r in s["results"]:
        m = r["median_of_reps"]
        stages = "fused" if r["stages_fused"] else f"{m['preprocess_p50_ms']:.2f} / {m['inference_p50_ms']:.2f} / {m['postprocess_p50_ms']:.2f}"
        rows.append([r["config"], r["resolution"], stages, f2(m["e2e_p50_ms"]), f2(m["e2e_p95_ms"]), f2(m["e2e_mean_ms"]),
                     f0(m["fps"]), f2(m["cpu_cores_busy"]), f1(m["rss_model_delta_mb"]), f0(m["rss_peak_mb"]),
                     f0(m["startup_load_ms"]), f1(m["startup_first_inference_ms"]),
                     f"{r['e2e_p50_spread_ms'][0]:.2f}–{r['e2e_p50_spread_ms'][1]:.2f}"])
    out.append(table(["config", "frame", "pre / inf / post", "e2e p50", "e2e p95", "mean", "FPS", "CPU cores", "RSS +MB (load)",
                      "RSS peak MB", "load ms", "1st infer ms", "p50 range over reps"], rows))
    out += ["", "### Model cost counted from the ONNX graph (Conv/Gemm MACs; FLOPs = 2 x MACs)", ""]
    rows = []
    for name, by in s["model_cost_onnx_graph"].items():
        rows.append([name, f"{list(by.values())[0]['params']:,}"] + [f"{v['gflops']:.3f}" for v in by.values()])
    out.append(table(["model", "params (ONNX initializers)"] + [f"GFLOPs @{k}" for k in next(iter(s["model_cost_onnx_graph"].values()))], rows))
    if s.get("sustained") and s["sustained"].get("windows"):
        w = s["sustained"]["windows"]
        out += ["", f"### Sustained run: {s['sustained']['config']} @ {s['sustained']['resolution']}, {s['sustained']['seconds']:.0f} s", "",
                table(["window end (s)"] + [f0(x["t_end_s"]) for x in w], [["mean ms"] + [f2(x["mean_ms"]) for x in w],
                                                                          ["p95 ms"] + [f2(x["p95_ms"]) for x in w]])]
    return "\n".join(out)


def browser_latency(s: dict) -> str:
    env = s["environment"]
    out = ["## Detector latency in the browser (Web Worker)", "",
           f"Browser: {s['browser']} ({'headless' if s['headless'] else 'headed'}); cores reported: {env['hardwareConcurrency']}; "
           f"cross-origin isolated: {env['crossOriginIsolated']}; WebGPU: {env['webgpu']}.",
           f"Protocol: {s['protocol']['warmup']} warm-up + {s['protocol']['timed']} timed calls; e2e includes the ImageBitmap "
           "transfer to the worker and the result message back.", ""]
    rows = []
    for r in s["results"]:
        if "error" in r:
            rows.append([r["config"], "unavailable", "", "", "", r["error"][:90]])
            continue
        stages = "fused" if r["stages_fused"] else f"{r['preprocess']['p50_ms']:.2f} / {r['inference']['p50_ms']:.2f} / {r['postprocess']['p50_ms']:.2f}"
        rows.append([r["config"], f2(r["end_to_end_incl_transfer"]["p50_ms"]), f2(r["end_to_end_incl_transfer"]["p95_ms"]), stages,
                     f0(r["startup_ms"]), r["execution_used"]])
    out.append(table(["config", "e2e p50 ms", "e2e p95 ms", "pre / inf / post p50", "startup ms", "execution"], rows))
    return "\n".join(out)


def tracking(s: dict) -> str:
    p = s["protocol"]
    out = ["## Tracking — synthetic-motion sequences over real WIDER faces, test split", "",
           f"Protocol: {p['tuning']}; fixed {json.dumps(p['fixed'])}; detector score floor {p['detector_score_floor']}; "
           f"{p['reported']}. Visible tracks = {p['visible_tracks']}.", f"Hardware: {env_line(s['environment'])}", ""]
    rows = []
    for r in s["results"]:
        t = r["test"]
        rows.append([r["detector"], r["tracker"], f1(r["det_threshold_from_dev"]), f3(t["HOTA"]), f3(t["IDF1"]), f3(t["MOTA"]),
                     f3(t["DetA"]), f3(t["AssA"]), t["IDSW"], t["Frag"], t["FP"], t["FN"], t["missed_tracks"], t["false_tracks"],
                     f0(t["confirmation_delay_ms_mean"]), f0(t["termination_delay_ms_mean"]), t["reentry_id_reused"],
                     f"{t['tracker_ms_mean']:.3f}", f"{t['tracker_ms_p95_max']:.2f}"])
    out.append(table(["detector", "tracker", "thr", "HOTA", "IDF1", "MOTA", "DetA", "AssA", "IDSW", "Frag", "FP", "FN", "missed trk",
                      "false trk", "confirm ms", "end delay ms", "re-entry ID reused", "tracker ms mean", "tracker ms p95 (worst seq)"], rows))
    for gen in s["results"][0]["test_by_generator"]:
        out += ["", f"### By generator: {gen}", ""]
        rows = []
        for r in s["results"]:
            t = r["test_by_generator"][gen]
            rows.append([r["detector"], r["tracker"], f3(t["HOTA"]), f3(t["IDF1"]), f3(t["MOTA"]), t["IDSW"], t["Frag"]])
        out.append(table(["detector", "tracker", "HOTA", "IDF1", "MOTA", "IDSW", "Frag"], rows))
    det = s.get("ablation_detector")
    out += ["", f"### Per scenario ({det}, base clips without lighting variants): IDF1 / IDSW / Frag", ""]
    rows_by_seq: dict[str, list] = {}
    trackers = []
    for r in s["results"]:
        if r["detector"] != det:
            continue
        trackers.append(r["tracker"])
        for name, m in r["test_per_sequence"].items():
            rows_by_seq.setdefault(name, []).append(f"{m['IDF1']:.3f} / {m['IDSW']} / {m['Frag']:.0f}")
    out.append(table(["sequence"] + trackers, [[k] + v for k, v in rows_by_seq.items()]))
    out += ["", f"### Detector interval and frame-rate ablations ({det}, test split)", ""]
    rows = []
    for a in s["ablations"]:
        t = a["test"]
        rows.append([a["tracker"], a["experiment"], a["value"], f3(t["HOTA"]), f3(t["IDF1"]), f3(t["MOTA"]), t["IDSW"], t["Frag"],
                     f0(t["termination_delay_ms_mean"])])
    out.append(table(["tracker", "experiment", "value", "HOTA", "IDF1", "MOTA", "IDSW", "Frag", "end delay ms"], rows))
    return "\n".join(out)


def main() -> None:
    parts = ["# Benchmark result tables", "",
             "Generated by `scripts/make_report_tables.py` from `results/*/summary.json`. All numbers are **measured locally**; "
             "published figures live in `docs/research/published_reference.md` and are never mixed into these tables.", ""]
    for path, fn in [("detection_accuracy/summary.json", detection_accuracy), ("detection_latency/summary.json", detection_latency),
                     ("browser_latency/summary.json", browser_latency), ("tracking/summary.json", tracking)]:
        s = load(path)
        parts += [fn(s) if s else f"## {path}\n\n_not run yet_", ""]
    dst = ROOT / "docs" / "results_tables.md"
    dst.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
