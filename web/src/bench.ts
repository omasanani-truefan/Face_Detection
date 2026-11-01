// In-browser detector benchmark. Same frames as scripts/bench_detection_latency.py
// (exported by scripts/export_bench_frames.py). Open /bench.html on any device -- a
// phone included -- and press Run; ?auto=1 runs immediately (used by browser-bench.mjs).
import { DetectorClient } from "./detector/client.ts";
import type { DetectorName, ExecutionPreference } from "./detector/protocol.ts";

const WARMUP = 10;
const TIMED = 100;
const statusEl = document.getElementById("status")!;
const outEl = document.getElementById("out")!;

interface Config {
  detector: DetectorName;
  inputLongSide: number | null;
  execution: ExecutionPreference;
}

const CONFIGS: Config[] = [
  { detector: "yunet", inputLongSide: 640, execution: "wasm" },
  { detector: "yunet", inputLongSide: 640, execution: "webgpu" },
  { detector: "yunet", inputLongSide: 320, execution: "wasm" },
  { detector: "scrfd_500m_kps", inputLongSide: 640, execution: "wasm" },
  { detector: "scrfd_500m_kps", inputLongSide: 640, execution: "webgpu" },
  { detector: "mediapipe_short", inputLongSide: null, execution: "cpu" },
  { detector: "mediapipe_short", inputLongSide: null, execution: "gpu" },
  { detector: "mediapipe_full", inputLongSide: null, execution: "cpu" },
  { detector: "mediapipe_full", inputLongSide: null, execution: "gpu" },
];

const stats = (a: number[]) => {
  const s = [...a].sort((x, y) => x - y);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { mean_ms: s.reduce((x, y) => x + y, 0) / s.length, p50_ms: q(0.5), p95_ms: q(0.95), n: s.length };
};

async function loadFrames(res: string): Promise<ImageBitmap[]> {
  const index = await (await fetch("/bench-frames/index.json")).json();
  return Promise.all((index[res] as string[]).map(async (p) => createImageBitmap(await (await fetch("/" + p)).blob())));
}

async function gpuInfo(): Promise<string> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ info?: Record<string, string> } | null> } }).gpu;
  if (!gpu) return "navigator.gpu absent";
  const a = await gpu.requestAdapter().catch(() => null);
  if (!a) return "no WebGPU adapter";
  const i = a.info ?? {};
  return `adapter: ${[i.vendor, i.architecture, i.description].filter(Boolean).join(" ") || "available"}`;
}

async function run() {
  const results: unknown[] = [];
  const resolutions = new URLSearchParams(location.search).get("res")?.split(",") ?? ["480p", "720p"];
  const env = { userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated: self.crossOriginIsolated, webgpu: await gpuInfo() };
  outEl.textContent = JSON.stringify(env, null, 1) + "\n";
  for (const res of resolutions) {
    const frames = await loadFrames(res);
    for (const cfg of CONFIGS) {
      const label = `${cfg.detector}@${cfg.inputLongSide ?? "native"} ${cfg.execution} ${res}`;
      statusEl.textContent = `running ${label}`;
      const client = new DetectorClient();
      try {
        const t0 = performance.now();
        const info = await client.init({ ...cfg, scoreThreshold: 0.5, nmsThreshold: 0.3 });
        const startup = performance.now() - t0;
        const requested = cfg.execution === "wasm" || cfg.execution === "cpu" ? ["wasm", "CPU"] : ["webgpu", "GPU"];
        if (!requested.some((r) => info.execution.includes(r))) throw new Error(`requested ${cfg.execution}, got ${info.execution}: ${info.decision}`);
        const pre: number[] = [], inf: number[] = [], post: number[] = [], e2e: number[] = [];
        let fused = false;
        for (let i = 0; i < WARMUP + TIMED; i++) {
          const bmp = await createImageBitmap(frames[i % frames.length]); // the worker takes ownership
          const s = performance.now();
          const r = await client.detect(i, i * 33.3, bmp);
          const el = performance.now() - s;
          if (i < WARMUP) continue;
          pre.push(r.timings.preprocessMs);
          inf.push(r.timings.inferenceMs);
          post.push(r.timings.postprocessMs);
          e2e.push(el);
          fused = r.timings.fused;
        }
        const row = { config: label, ...cfg, resolution: res, execution_used: info.execution, runtime: info.runtime, startup_ms: startup,
          load_ms: info.loadMs, stages_fused: fused, preprocess: stats(pre), inference: stats(inf), postprocess: stats(post),
          end_to_end_incl_transfer: stats(e2e) };
        results.push(row);
        outEl.textContent += `${label.padEnd(40)} e2e p50 ${row.end_to_end_incl_transfer.p50_ms.toFixed(2)} ms  p95 ${row.end_to_end_incl_transfer.p95_ms.toFixed(2)} ms  (${info.execution})\n`;
      } catch (e) {
        results.push({ config: label, ...cfg, resolution: res, error: (e as Error).message });
        outEl.textContent += `${label.padEnd(40)} unavailable: ${(e as Error).message.slice(0, 160)}\n`;
      } finally {
        client.terminate();
      }
    }
    frames.forEach((f) => f.close());
  }
  statusEl.textContent = "done";
  (window as unknown as { __bench: unknown }).__bench = { environment: env, protocol: { warmup: WARMUP, timed: TIMED }, results };
}

document.getElementById("run")!.onclick = () => void run();
if (new URLSearchParams(location.search).has("auto")) void run();
