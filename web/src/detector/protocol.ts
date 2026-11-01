// Messages between the main thread and the detection worker.
import type { Detection, StageTimings } from "../types.ts";

export type DetectorName = "yunet" | "scrfd_500m_kps" | "mediapipe_short" | "mediapipe_full";
/** auto = measure the available accelerators on this device and keep the faster one. */
export type ExecutionPreference = "auto" | "webgpu" | "wasm" | "gpu" | "cpu";

export interface InitRequest {
  type: "init";
  detector: DetectorName;
  inputLongSide: number | null; // null = native frame size (ORT models only)
  scoreThreshold: number;
  nmsThreshold: number;
  execution: ExecutionPreference;
}

export interface DetectRequest {
  type: "detect";
  frameId: number;
  timestampMs: number;
  bitmap: ImageBitmap;
}

export type WorkerRequest = InitRequest | DetectRequest | { type: "dispose" };

export interface BackendInfo {
  detector: DetectorName;
  runtime: string; // e.g. "onnxruntime-web 1.30.0 / wasm (4 threads)"
  execution: string; // the accelerator actually used
  loadMs: number;
  /** Why this execution path was chosen (probe timings, fallbacks taken). */
  decision: string;
  keypoints: string[];
}

export type WorkerResponse =
  | { type: "ready"; info: BackendInfo }
  | { type: "result"; frameId: number; timestampMs: number; detections: Detection[]; timings: StageTimings; frameW: number; frameH: number }
  | { type: "error"; message: string; frameId?: number };
