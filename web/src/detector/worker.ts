/// <reference lib="webworker" />
// Detection runs here, never on the UI thread. One request at a time: the main thread
// only sends a new frame once the previous result is back, so a slow device drops
// frames instead of queueing them (no growing lag behind the camera).
//
// Each runtime is imported on first use, so a session that only runs YuNet never
// downloads MediaPipe's JS (and vice versa).
import type { FaceDetector } from "@mediapipe/tasks-vision";
import type { Detection, StageTimings } from "../types.ts";
import type { OrtFaceModel } from "./ort-models.ts";
import type { BackendInfo, InitRequest, WorkerRequest, WorkerResponse } from "./protocol.ts";

declare const self: DedicatedWorkerGlobalScope;

const asset = (p: string) => new URL(p, self.location.origin + "/").href;

async function loadOrt() {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths = asset("ort/");
  // Threads need cross-origin isolation (COOP/COEP, set in vite.config.ts); otherwise 1.
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  return { ort, models: await import("./ort-models.ts") };
}

interface Backend {
  detect(bitmap: ImageBitmap, ts: number): Promise<{ detections: Detection[]; timings: StageTimings }>;
  dispose(): Promise<void> | void;
}

const MODEL_FILES: Record<string, string> = {
  yunet: "models/face_detection_yunet_2023mar_dyn.onnx",
  scrfd_500m_kps: "models/scrfd_500m_kps.onnx",
  mediapipe_short: "models/blaze_face_short_range.tflite",
  mediapipe_full: "models/blaze_face_full_range.tflite",
};

let backend: Backend | null = null;

function probeBitmap(): Promise<ImageBitmap> {
  const c = new OffscreenCanvas(640, 480);
  const g = c.getContext("2d")!;
  g.fillStyle = "#777";
  g.fillRect(0, 0, 640, 480);
  return createImageBitmap(c);
}

async function timeIt(fn: () => Promise<unknown>, n: number): Promise<number> {
  await fn(); // first call compiles shaders / allocates
  const t: number[] = [];
  for (let i = 0; i < n; i++) {
    const s = performance.now();
    await fn();
    t.push(performance.now() - s);
  }
  t.sort((a, b) => a - b);
  return t[Math.floor(t.length / 2)];
}

async function webgpuAvailable(): Promise<boolean> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return false;
  try {
    return (await gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function initOrt(req: InitRequest): Promise<{ backend: Backend; info: BackendInfo }> {
  const { ort, models } = await loadOrt();
  const { YuNet, Scrfd } = models;
  const make = (): OrtFaceModel =>
    req.detector === "yunet"
      ? new YuNet(req.inputLongSide, req.scoreThreshold, req.nmsThreshold)
      : new Scrfd(req.inputLongSide, req.scoreThreshold, req.nmsThreshold);
  const url = asset(MODEL_FILES[req.detector]);
  const t0 = performance.now();
  const notes: string[] = [];
  const candidates: ("webgpu" | "wasm")[] = [];
  if (req.execution === "wasm" || req.execution === "cpu") candidates.push("wasm");
  else if (await webgpuAvailable()) candidates.push("webgpu", "wasm");
  else {
    candidates.push("wasm");
    notes.push("WebGPU unavailable (no navigator.gpu or no adapter)");
  }
  const loaded: { ep: "webgpu" | "wasm"; model: OrtFaceModel; ms: number }[] = [];
  const probe = await probeBitmap();
  for (const ep of candidates) {
    try {
      const m = make();
      await m.load(url, ep);
      const ms = req.execution === "auto" && candidates.length > 1 ? await timeIt(() => m.detect(probe), 5) : NaN;
      loaded.push({ ep, model: m, ms });
      if (req.execution !== "auto") break;
    } catch (e) {
      notes.push(`${ep} failed: ${(e as Error).message.slice(0, 120)}`);
    }
  }
  probe.close();
  if (!loaded.length) throw new Error(`no execution provider worked: ${notes.join("; ")}`);
  loaded.sort((a, b) => (Number.isNaN(a.ms) ? 0 : a.ms) - (Number.isNaN(b.ms) ? 0 : b.ms));
  const chosen = loaded[0];
  for (const l of loaded.slice(1)) await l.model.release();
  if (loaded.length > 1) notes.push(`probe p50: ${loaded.map((l) => `${l.ep} ${l.ms.toFixed(1)} ms`).join(", ")}`);
  const threads = chosen.ep === "wasm" ? ` (${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads === 1 ? "" : "s"})` : "";
  return {
    backend: { detect: (b) => chosen.model.detect(b), dispose: () => chosen.model.release() },
    info: {
      detector: req.detector,
      runtime: `onnxruntime-web ${ort.env.versions?.web ?? ""}`.trim(),
      execution: chosen.ep + threads,
      loadMs: performance.now() - t0,
      decision: notes.join("; ") || `${chosen.ep} requested`,
      keypoints: make().keypoints,
    },
  };
}

async function initMediaPipe(req: InitRequest): Promise<{ backend: Backend; info: BackendInfo }> {
  const t0 = performance.now();
  const { FaceDetector, FilesetResolver } = await import("@mediapipe/tasks-vision");
  const fileset = await FilesetResolver.forVisionTasks(asset("mediapipe"), true);
  const notes: string[] = [];
  const delegates: ("GPU" | "CPU")[] = req.execution === "cpu" || req.execution === "wasm" ? ["CPU"] : ["GPU", "CPU"];
  const loaded: { d: "GPU" | "CPU"; det: FaceDetector; ms: number }[] = [];
  const probe = await probeBitmap();
  let probeTs = 0;
  for (const d of delegates) {
    try {
      const det = await FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: asset(MODEL_FILES[req.detector]), delegate: d },
        runningMode: "VIDEO",
        minDetectionConfidence: req.scoreThreshold,
        minSuppressionThreshold: req.nmsThreshold,
      });
      const ms =
        req.execution === "auto" && delegates.length > 1
          ? await timeIt(async () => det.detectForVideo(probe, ++probeTs), 5)
          : NaN;
      loaded.push({ d, det, ms });
      if (req.execution !== "auto") break;
    } catch (e) {
      notes.push(`${d} delegate failed: ${(e as Error).message.slice(0, 120)}`);
    }
  }
  probe.close();
  if (!loaded.length) throw new Error(notes.join("; "));
  loaded.sort((a, b) => (Number.isNaN(a.ms) ? 0 : a.ms) - (Number.isNaN(b.ms) ? 0 : b.ms));
  const chosen = loaded[0];
  for (const l of loaded.slice(1)) l.det.close();
  if (loaded.length > 1) notes.push(`probe p50: ${loaded.map((l) => `${l.d} ${l.ms.toFixed(1)} ms`).join(", ")}`);
  // VIDEO mode needs strictly increasing timestamps; a restarted video goes back to 0.
  let lastTs = probeTs;
  return {
    backend: {
      async detect(bitmap, ts) {
        const t0 = performance.now();
        lastTs = Math.max(lastTs + 1, Math.round(ts));
        const r = chosen.det.detectForVideo(bitmap, lastTs);
        const t1 = performance.now();
        const W = bitmap.width;
        const H = bitmap.height;
        const detections: Detection[] = r.detections.map((d) => {
          const bb = d.boundingBox!;
          const x1 = Math.max(0, bb.originX);
          const y1 = Math.max(0, bb.originY);
          const x2 = Math.min(W, bb.originX + bb.width);
          const y2 = Math.min(H, bb.originY + bb.height);
          return {
            bbox: [x1, y1, x2 - x1, y2 - y1],
            confidence: d.categories[0]?.score ?? 0,
            keypoints: d.keypoints.map((k) => [k.x * W, k.y * H] as [number, number]),
          };
        });
        return { detections, timings: { preprocessMs: 0, inferenceMs: t1 - t0, postprocessMs: performance.now() - t1, fused: true } };
      },
      dispose: () => chosen.det.close(),
    },
    info: {
      detector: req.detector,
      runtime: "@mediapipe/tasks-vision",
      execution: chosen.d === "GPU" ? "MediaPipe GPU delegate (WebGL)" : "MediaPipe CPU (wasm)",
      loadMs: performance.now() - t0,
      decision: notes.join("; ") || `${chosen.d} requested`,
      keypoints: ["right_eye", "left_eye", "nose_tip", "mouth_center", "right_ear_tragion", "left_ear_tragion"],
    },
  };
}

const post = (m: WorkerResponse) => self.postMessage(m);

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  if (msg.type === "init") {
    try {
      await backend?.dispose();
      backend = null;
      const r = msg.detector.startsWith("mediapipe") ? await initMediaPipe(msg) : await initOrt(msg);
      backend = r.backend;
      post({ type: "ready", info: r.info });
    } catch (e) {
      post({ type: "error", message: `could not initialise ${msg.detector}: ${(e as Error).message}` });
    }
  } else if (msg.type === "detect") {
    const { bitmap, frameId, timestampMs } = msg;
    try {
      if (!backend) throw new Error("detector not initialised");
      const r = await backend.detect(bitmap, timestampMs);
      post({ type: "result", frameId, timestampMs, detections: r.detections, timings: r.timings, frameW: bitmap.width, frameH: bitmap.height });
    } catch (e) {
      post({ type: "error", message: (e as Error).message, frameId });
    } finally {
      bitmap.close();
    }
  } else if (msg.type === "dispose") {
    await backend?.dispose();
    backend = null;
  }
};

