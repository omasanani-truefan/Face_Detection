// Frame sources for the browser. Both drive the same callback, (video element, frame id,
// timestamp ms), so the detector and tracker never know which one they are fed by.
//
// Timestamps: uploaded video -> the frame's presentation time (requestVideoFrameCallback
// mediaTime), so variable-frame-rate files keep their real timing; camera -> the
// capture clock. Neither is derived from a frame counter.

export type FrameCallback = (video: HTMLVideoElement, frameId: number, timestampMs: number) => void;

export interface SourceInfo {
  kind: "video" | "camera";
  label: string;
  width: number;
  height: number;
  durationS: number | null;
  /** Estimated from frame timestamps; browsers do not expose a container FPS. */
  measuredFps: number | null;
}

type VideoWithRvfc = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: { mediaTime: number; presentedFrames: number }) => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

abstract class FrameSource {
  protected frameId = 0;
  private handle: number | null = null;
  private lastTs: number | null = null;
  private intervals: number[] = [];
  protected onFrame: FrameCallback | null = null;

  constructor(readonly video: VideoWithRvfc) {}

  abstract info(): SourceInfo;
  protected abstract timestamp(meta: { mediaTime: number } | null): number;

  start(cb: FrameCallback): void {
    this.onFrame = cb;
    this.loop();
  }

  protected loop(): void {
    const v = this.video;
    const tick = (meta: { mediaTime: number } | null) => {
      if (!this.onFrame) return;
      const ts = this.timestamp(meta);
      if (this.lastTs === null || ts > this.lastTs) {
        if (this.lastTs !== null) {
          this.intervals.push(ts - this.lastTs);
          if (this.intervals.length > 60) this.intervals.shift();
        }
        this.lastTs = ts;
        this.onFrame(v, this.frameId++, ts);
      }
      this.schedule(tick);
    };
    this.schedule(tick);
  }

  private schedule(tick: (meta: { mediaTime: number } | null) => void): void {
    if (this.video.requestVideoFrameCallback) this.handle = this.video.requestVideoFrameCallback((_n, meta) => tick(meta));
    else this.handle = requestAnimationFrame(() => tick(null)); // fallback: poll currentTime each paint
  }

  protected stopLoop(): void {
    if (this.handle !== null) {
      if (this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this.handle);
      cancelAnimationFrame(this.handle);
    }
    this.handle = null;
    this.onFrame = null;
  }

  /** Forget timestamp history (the video was rewound). */
  protected resetClock(): void {
    this.lastTs = null;
    this.intervals = [];
  }

  protected fps(): number | null {
    if (this.intervals.length < 5) return null;
    const s = [...this.intervals].sort((a, b) => a - b);
    return 1000 / s[Math.floor(s.length / 2)];
  }

  abstract stop(): void;
}

/** A video the page can play: a user's file, or a same-origin URL (e.g. the local YouTube bridge). */
export type VideoInput = File | { url: string; label: string };

export class VideoFileSource extends FrameSource {
  private url: string;
  private label: string;
  private owned: boolean;
  constructor(video: HTMLVideoElement, input: VideoInput) {
    super(video);
    this.owned = input instanceof File;
    this.url = input instanceof File ? URL.createObjectURL(input) : input.url;
    this.label = input instanceof File ? input.name : input.label;
  }

  async open(): Promise<void> {
    const v = this.video;
    v.srcObject = null;
    v.src = this.url;
    v.muted = true;
    v.playsInline = true;
    const unsupported = () =>
      new Error(`cannot decode "${this.label}" in this browser (unsupported codec or container; H.264/VP9/AV1 in MP4 or WebM play everywhere)`);
    await new Promise<void>((resolve, reject) => {
      v.onloadedmetadata = () => (v.videoWidth > 0 ? resolve() : reject(unsupported()));
      v.onerror = () => reject(unsupported());
    });
    // A container can parse while its codec cannot (e.g. MPEG-4 Part 2 in .mp4 on Chrome).
    v.onerror = () => this.onError?.(unsupported());
  }

  onError: ((e: Error) => void) | null = null;

  protected timestamp(meta: { mediaTime: number } | null): number {
    return (meta ? meta.mediaTime : this.video.currentTime) * 1000;
  }

  info(): SourceInfo {
    const v = this.video;
    return { kind: "video", label: this.label, width: v.videoWidth, height: v.videoHeight,
      durationS: Number.isFinite(v.duration) ? v.duration : null, measuredFps: this.fps() };
  }

  play(): Promise<void> {
    return this.video.play();
  }
  pause(): void {
    this.video.pause();
  }
  restart(): void {
    this.video.currentTime = 0;
    this.resetClock();
    this.frameId = 0;
  }

  stop(): void {
    this.stopLoop();
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    if (this.owned) URL.revokeObjectURL(this.url);
  }
}

export class CameraError extends Error {}

export class CameraSource extends FrameSource {
  private stream: MediaStream | null = null;
  private t0 = 0;
  onEnded: (() => void) | null = null;

  static async list(): Promise<MediaDeviceInfo[]> {
    const devs = await navigator.mediaDevices.enumerateDevices();
    return devs.filter((d) => d.kind === "videoinput");
  }

  async open(deviceId: string | null, facingMode: "user" | "environment" = "user"): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) throw new CameraError("camera API unavailable (page must be served over https or localhost)");
    const video: MediaTrackConstraints = { width: { ideal: 1280 }, height: { ideal: 720 } };
    if (deviceId) video.deviceId = { exact: deviceId };
    else video.facingMode = facingMode;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      const n = (e as DOMException).name;
      const msg =
        n === "NotAllowedError" ? "camera permission denied"
        : n === "NotFoundError" || n === "OverconstrainedError" ? "no matching camera found"
        : n === "NotReadableError" ? "camera is in use by another application"
        : `camera error: ${(e as Error).message}`;
      throw new CameraError(msg);
    }
    // An unplugged camera or a revoked permission ends the track: surface it, don't freeze.
    for (const t of this.stream.getVideoTracks()) t.addEventListener("ended", () => this.onEnded?.());
    const v = this.video;
    v.removeAttribute("src");
    v.srcObject = this.stream;
    v.muted = true;
    v.playsInline = true;
    await v.play();
    this.t0 = performance.now();
  }

  protected timestamp(): number {
    return performance.now() - this.t0;
  }

  info(): SourceInfo {
    const s = this.stream?.getVideoTracks()[0]?.getSettings();
    return { kind: "camera", label: this.stream?.getVideoTracks()[0]?.label || "camera", width: this.video.videoWidth || s?.width || 0,
      height: this.video.videoHeight || s?.height || 0, durationS: null, measuredFps: this.fps() ?? s?.frameRate ?? null };
  }

  /** Release the device: the camera light must go off when the user leaves camera mode. */
  stop(): void {
    this.stopLoop();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }
}
