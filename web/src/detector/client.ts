// Main-thread handle on the detection worker: promise per request, at most one in flight.
import type { Detection, StageTimings } from "../types.ts";
import type { BackendInfo, InitRequest, WorkerResponse } from "./protocol.ts";

export interface DetectResult {
  frameId: number;
  timestampMs: number;
  detections: Detection[];
  timings: StageTimings;
  frameW: number;
  frameH: number;
}

export class DetectorClient {
  private worker: Worker;
  private pending: { resolve: (v: unknown) => void; reject: (e: Error) => void } | null = null;
  info: BackendInfo | null = null;

  constructor() {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const m = ev.data;
      const p = this.pending;
      this.pending = null;
      if (!p) return;
      if (m.type === "error") p.reject(new Error(m.message));
      else if (m.type === "ready") p.resolve(m.info);
      else p.resolve(m);
    };
    this.worker.onerror = (ev) => {
      this.pending?.reject(new Error(ev.message || "detector worker crashed"));
      this.pending = null;
    };
  }

  get busy(): boolean {
    return this.pending !== null;
  }

  private request<T>(msg: unknown, transfer: Transferable[] = []): Promise<T> {
    if (this.pending) return Promise.reject(new Error("detector busy"));
    return new Promise<T>((resolve, reject) => {
      this.pending = { resolve: resolve as (v: unknown) => void, reject };
      this.worker.postMessage(msg, transfer);
    });
  }

  async init(req: Omit<InitRequest, "type">): Promise<BackendInfo> {
    this.info = await this.request<BackendInfo>({ type: "init", ...req });
    return this.info;
  }

  detect(frameId: number, timestampMs: number, bitmap: ImageBitmap): Promise<DetectResult> {
    return this.request<DetectResult>({ type: "detect", frameId, timestampMs, bitmap }, [bitmap]);
  }

  terminate(): void {
    this.worker.terminate();
  }
}
