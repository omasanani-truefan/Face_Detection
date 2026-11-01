// YuNet and SCRFD on ONNX Runtime Web. Decoders mirror src/facetrack/detectors/yunet.py
// and scrfd.py (which are verified against OpenCV's FaceDetectorYN).
import * as ort from "onnxruntime-web";
import type { Detection, StageTimings } from "../types.ts";

const STRIDES = [8, 16, 32];

export interface Letterbox {
  scale: number;
  inW: number;
  inH: number;
  newW: number;
  newH: number;
}

export function letterboxSize(w: number, h: number, longSide: number | null): Letterbox {
  const ls = longSide ?? Math.max(w, h);
  const scale = ls / Math.max(w, h);
  const newW = Math.max(1, Math.round(w * scale));
  const newH = Math.max(1, Math.round(h * scale));
  return { scale, newW, newH, inW: Math.ceil(newW / 32) * 32, inH: Math.ceil(newH / 32) * 32 };
}

/** Greedy NMS on xyxy boxes, highest score first. */
export function nms(boxes: Float32Array[], scores: number[], iouThr: number): number[] {
  const order = scores.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0]).map(([, i]) => i);
  const keep: number[] = [];
  const dead = new Uint8Array(scores.length);
  for (const i of order) {
    if (dead[i]) continue;
    keep.push(i);
    const a = boxes[i];
    const aa = (a[2] - a[0]) * (a[3] - a[1]);
    for (const j of order) {
      if (dead[j] || j === i) continue;
      const b = boxes[j];
      const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
      const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
      const inter = iw * ih;
      if (inter / (aa + (b[2] - b[0]) * (b[3] - b[1]) - inter) > iouThr) dead[j] = 1;
    }
  }
  return keep;
}

export abstract class OrtFaceModel {
  protected session!: ort.InferenceSession;
  protected canvas: OffscreenCanvas | null = null;
  protected ctx: OffscreenCanvasRenderingContext2D | null = null;
  readonly keypoints = ["right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner"];

  constructor(
    protected inputLongSide: number | null,
    protected scoreThreshold: number,
    protected nmsThreshold: number,
  ) {}

  async load(url: string, ep: "webgpu" | "wasm"): Promise<void> {
    this.session = await ort.InferenceSession.create(url, { executionProviders: [ep], graphOptimizationLevel: "all" });
  }

  async release(): Promise<void> {
    await this.session?.release();
  }

  /** Fill the letterboxed tensor in the model's channel order / normalisation. */
  protected abstract fill(rgba: Uint8ClampedArray, out: Float32Array, plane: number): void;
  protected abstract decode(outputs: ort.InferenceSession.OnnxValueMapType, lb: Letterbox): {
    boxes: Float32Array[];
    scores: number[];
    kps: Float32Array[];
  };

  async detect(bitmap: ImageBitmap): Promise<{ detections: Detection[]; timings: StageTimings }> {
    const t0 = performance.now();
    const W = bitmap.width;
    const H = bitmap.height;
    const lb = letterboxSize(W, H, this.inputLongSide);
    if (!this.canvas || this.canvas.width !== lb.inW || this.canvas.height !== lb.inH) {
      this.canvas = new OffscreenCanvas(lb.inW, lb.inH);
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    }
    const ctx = this.ctx!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, lb.inW, lb.inH);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, 0, 0, lb.newW, lb.newH);
    const rgba = ctx.getImageData(0, 0, lb.inW, lb.inH).data;
    const plane = lb.inW * lb.inH;
    const data = new Float32Array(3 * plane);
    this.fill(rgba, data, plane);
    const input = new ort.Tensor("float32", data, [1, 3, lb.inH, lb.inW]);
    const t1 = performance.now();
    const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
    const t2 = performance.now();
    const { boxes, scores, kps } = this.decode(outputs, lb);
    const keep = nms(boxes, scores, this.nmsThreshold);
    const detections: Detection[] = [];
    for (const i of keep) {
      const b = boxes[i];
      const x1 = Math.max(0, b[0] / lb.scale);
      const y1 = Math.max(0, b[1] / lb.scale);
      const x2 = Math.min(W, b[2] / lb.scale);
      const y2 = Math.min(H, b[3] / lb.scale);
      if (x2 - x1 < 1 || y2 - y1 < 1) continue;
      const k = kps[i];
      detections.push({
        bbox: [x1, y1, x2 - x1, y2 - y1],
        confidence: scores[i],
        keypoints: Array.from({ length: 5 }, (_, j) => [k[2 * j] / lb.scale, k[2 * j + 1] / lb.scale] as [number, number]),
      });
    }
    for (const v of Object.values(outputs)) (v as ort.Tensor).dispose?.();
    const t3 = performance.now();
    return { detections, timings: { preprocessMs: t1 - t0, inferenceMs: t2 - t1, postprocessMs: t3 - t2, fused: false } };
  }
}

/** YuNet: raw BGR 0..255 input; anchor-free, one prior per cell. */
export class YuNet extends OrtFaceModel {
  protected fill(rgba: Uint8ClampedArray, out: Float32Array, plane: number): void {
    for (let i = 0, p = 0; p < plane; i += 4, p++) {
      out[p] = rgba[i + 2];
      out[plane + p] = rgba[i + 1];
      out[2 * plane + p] = rgba[i];
    }
  }

  protected decode(o: ort.InferenceSession.OnnxValueMapType, lb: Letterbox) {
    const boxes: Float32Array[] = [];
    const scores: number[] = [];
    const kps: Float32Array[] = [];
    for (const s of STRIDES) {
      const cols = lb.inW / s;
      const cls = o[`cls_${s}`].data as Float32Array;
      const obj = o[`obj_${s}`].data as Float32Array;
      const bb = o[`bbox_${s}`].data as Float32Array;
      const kp = o[`kps_${s}`].data as Float32Array;
      for (let i = 0; i < cls.length; i++) {
        const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
        if (score < this.scoreThreshold) continue;
        const c = i % cols;
        const r = Math.floor(i / cols);
        const cx = (c + bb[4 * i]) * s;
        const cy = (r + bb[4 * i + 1]) * s;
        const w = Math.exp(bb[4 * i + 2]) * s;
        const h = Math.exp(bb[4 * i + 3]) * s;
        boxes.push(Float32Array.of(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2));
        const k = new Float32Array(10);
        for (let j = 0; j < 5; j++) {
          k[2 * j] = (kp[10 * i + 2 * j] + c) * s;
          k[2 * j + 1] = (kp[10 * i + 2 * j + 1] + r) * s;
        }
        kps.push(k);
        scores.push(score);
      }
    }
    return { boxes, scores, kps };
  }
}

/** SCRFD-500M-KPS: RGB (x-127.5)/128; 2 anchors per cell, distance-encoded boxes. */
export class Scrfd extends OrtFaceModel {
  protected fill(rgba: Uint8ClampedArray, out: Float32Array, plane: number): void {
    for (let i = 0, p = 0; p < plane; i += 4, p++) {
      out[p] = (rgba[i] - 127.5) / 128;
      out[plane + p] = (rgba[i + 1] - 127.5) / 128;
      out[2 * plane + p] = (rgba[i + 2] - 127.5) / 128;
    }
  }

  protected decode(o: ort.InferenceSession.OnnxValueMapType, lb: Letterbox) {
    // Output names are auto-generated; group by (rows, last dim) like the Python decoder.
    const byRows = new Map<number, Map<number, Float32Array>>();
    for (const v of Object.values(o)) {
      const t = v as ort.Tensor;
      const last = t.dims[t.dims.length - 1];
      const rows = t.data.length / last;
      if (!byRows.has(rows)) byRows.set(rows, new Map());
      byRows.get(rows)!.set(last, t.data as Float32Array);
    }
    const boxes: Float32Array[] = [];
    const scores: number[] = [];
    const kps: Float32Array[] = [];
    for (const s of STRIDES) {
      const cols = lb.inW / s;
      const rowsN = (lb.inH / s) * cols * 2;
      const g = byRows.get(rowsN)!;
      const sc = g.get(1)!;
      const dist = g.get(4)!;
      const kd = g.get(10)!;
      for (let i = 0; i < rowsN; i++) {
        if (sc[i] < this.scoreThreshold) continue;
        const cell = Math.floor(i / 2);
        const cx = (cell % cols) * s;
        const cy = Math.floor(cell / cols) * s;
        boxes.push(Float32Array.of(cx - dist[4 * i] * s, cy - dist[4 * i + 1] * s, cx + dist[4 * i + 2] * s, cy + dist[4 * i + 3] * s));
        const k = new Float32Array(10);
        for (let j = 0; j < 5; j++) {
          k[2 * j] = cx + kd[10 * i + 2 * j] * s;
          k[2 * j + 1] = cy + kd[10 * i + 2 * j + 1] * s;
        }
        kps.push(k);
        scores.push(sc[i]);
      }
    }
    return { boxes, scores, kps };
  }
}
