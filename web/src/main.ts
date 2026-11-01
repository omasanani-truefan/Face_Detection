import { DetectorClient } from "./detector/client.ts";
import type { DetectorName, ExecutionPreference } from "./detector/protocol.ts";
import { drawOverlay, trackColor } from "./overlay.ts";
import { CameraError, CameraSource, type VideoInput, VideoFileSource } from "./sources.ts";
import { type BaseTracker, createTracker, type TrackerName, type TrackerOutput, type TrackEvent, snapshot, visible } from "./tracking/tracker.ts";
import type { Detection } from "./types.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>("video");
const overlay = $<HTMLCanvasElement>("overlay");
const statusEl = $<HTMLParagraphElement>("status");
const statsEl = $<HTMLDListElement>("stats");
const eventsEl = $<HTMLOListElement>("events");

// Tracker settings from the benchmark (docs/REPORT.md): the detector runs with a low
// floor so ByteTrack's second stage can use low-score boxes; only >= detThreshold starts tracks.
const DET_FLOOR = 0.3;
const TRACKER_CFG = { detThreshold: 0.6, lowDetThreshold: DET_FLOOR, minHits: 3, maxLostMs: 1000, tentativeTimeoutMs: 300 };

const client = new DetectorClient();
let tracker: BaseTracker = createTracker("bytetrack", TRACKER_CFG);
let source: VideoFileSource | CameraSource | null = null;
let ready = false;
let initSeq = 0;
let n = 0;
let lastTs = -Infinity;
let lastOut: TrackerOutput | null = null;
let lastRaw: Detection[] | null = null;
let frameW = 0;
let frameH = 0;

// Rolling measurements.
const win = <T>(arr: T[], v: T, max = 90) => {
  arr.push(v);
  if (arr.length > max) arr.shift();
};
const pre: number[] = [];
const inf: number[] = [];
const post: number[] = [];
const trk: number[] = [];
const e2e: number[] = [];
const doneAt: number[] = [];
let fused = false;
let dropped = 0;
let offered = 0;
const p50 = (a: number[]) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)] : NaN);
const p95 = (a: number[]) => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length * 0.95)] : NaN);

// Export buffer: one JSON line per processed frame.
let records: string[] = [];

function setStatus(msg: string, error = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", error);
}

function settings() {
  const size = $<HTMLSelectElement>("input-size").value;
  return {
    detector: $<HTMLSelectElement>("detector").value as DetectorName,
    inputLongSide: size === "native" ? null : Number(size),
    tracker: $<HTMLSelectElement>("tracker").value as TrackerName,
    execution: $<HTMLSelectElement>("execution").value as ExecutionPreference,
    detectEvery: Number($<HTMLSelectElement>("detect-every").value),
    showRaw: $<HTMLInputElement>("show-raw").checked,
  };
}

async function initDetector() {
  const s = settings();
  const seq = ++initSeq;
  ready = false;
  setStatus(`Loading ${s.detector}…`);
  try {
    const info = await client.init({ detector: s.detector, inputLongSide: s.inputLongSide, scoreThreshold: DET_FLOOR,
      nmsThreshold: s.detector === "scrfd_500m_kps" ? 0.4 : 0.3, execution: s.execution });
    if (seq !== initSeq) return;
    ready = true;
    setStatus(`${info.detector} ready on ${info.execution} in ${info.loadMs.toFixed(0)} ms`);
  } catch (e) {
    if (seq === initSeq) setStatus((e as Error).message, true);
  }
  resetTracking();
}

function resetTracking() {
  tracker = createTracker(settings().tracker, TRACKER_CFG);
  n = 0;
  lastTs = -Infinity;
  lastOut = null;
  for (const a of [pre, inf, post, trk, e2e, doneAt]) a.length = 0;
  dropped = offered = 0;
  records = [];
  eventsEl.replaceChildren();
  $<HTMLButtonElement>("export").disabled = true;
  redraw();
}

function logEvents(events: TrackEvent[]) {
  for (const e of events) {
    if (e.event === "track_updated") continue;
    const li = document.createElement("li");
    li.innerHTML = `<span class="t">${(e.timestampMs / 1000).toFixed(2)}s</span><span class="dot"></span><span></span>`;
    (li.children[1] as HTMLElement).style.background = trackColor(e.trackId);
    li.children[2].textContent = `${e.event.replace("track_", "")} · ID ${e.trackId}`;
    eventsEl.prepend(li);
    while (eventsEl.children.length > 200) eventsEl.lastChild!.remove();
  }
}

function redraw() {
  const s = settings();
  if (!frameW) {
    const g = overlay.getContext("2d")!;
    g.clearRect(0, 0, overlay.width, overlay.height);
    return;
  }
  drawOverlay(overlay, frameW, frameH, lastOut?.tracks ?? [], s.showRaw ? lastRaw : null, false);
}

async function onFrame(v: HTMLVideoElement, frameId: number, ts: number) {
  if (!ready) return;
  if (ts < lastTs) resetTracking(); // user seeked backwards
  offered++;
  const s = settings();
  // One detection in flight at most: if busy, drop this frame rather than queue it.
  if (client.busy) {
    dropped++;
    return;
  }
  frameW = v.videoWidth;
  frameH = v.videoHeight;
  const runDetector = n++ % s.detectEvery === 0;
  lastTs = ts;
  if (!runDetector) {
    lastOut = tracker.predict(ts, frameId);
    redraw();
    return;
  }
  const t0 = performance.now();
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(v);
  } catch {
    return; // frame not decodable yet (e.g. right after a seek)
  }
  try {
    const r = await client.detect(frameId, ts, bitmap);
    const t1 = performance.now();
    const out = tracker.update(r.detections, r.timestampMs, r.frameId);
    const t2 = performance.now();
    win(pre, r.timings.preprocessMs);
    win(inf, r.timings.inferenceMs);
    win(post, r.timings.postprocessMs);
    win(trk, t2 - t1);
    win(e2e, t2 - t0);
    win(doneAt, t2);
    fused = r.timings.fused;
    lastOut = out;
    lastRaw = r.detections;
    logEvents(out.events);
    records.push(JSON.stringify({
      type: "frame", frame_id: r.frameId, timestamp_ms: +r.timestampMs.toFixed(3), camera_id: source?.info().kind,
      detections: r.detections.map((d) => ({ bbox: d.bbox.map((x) => +x.toFixed(2)), confidence: +d.confidence.toFixed(4) })),
      tracks: out.tracks.filter((t) => t.state !== "tentative").map(snapshot),
      events: out.events.map((e) => ({ event: e.event, track_id: e.trackId, timestamp_ms: e.timestampMs })),
      timings_ms: { preprocess: r.timings.preprocessMs, inference: r.timings.inferenceMs, postprocess: r.timings.postprocessMs,
        detector_fused: r.timings.fused, tracking: t2 - t1, end_to_end: t2 - t0 },
    }));
    $<HTMLButtonElement>("export").disabled = false;
    redraw();
  } catch (e) {
    setStatus((e as Error).message, true);
  }
}

function finishStream() {
  const ts = lastOut?.timestampMs ?? 0;
  const ended = tracker.finish(ts);
  logEvents(ended);
  for (const e of ended) records.push(JSON.stringify({ type: "event", event: e.event, track_id: e.trackId, timestamp_ms: ts }));
  lastOut = null;
  redraw();
}

function renderStats() {
  const info = client.info;
  const now = performance.now();
  const recent = doneAt.filter((t) => now - t < 2000);
  const fps = recent.length > 1 ? (recent.length - 1) / ((recent[recent.length - 1] - recent[0]) / 1000) : NaN;
  const src = source?.info();
  const f = (x: number, d = 1) => (Number.isFinite(x) ? x.toFixed(d) : "–");
  const rows: [string, string][] = [
    ["Source", src ? `${src.kind} ${src.width}×${src.height}` : "–"],
    ["Source FPS", src?.measuredFps ? `${f(src.measuredFps)} (measured)` : "–"],
    ["Processed FPS", f(fps)],
    ["Dropped", offered ? `${dropped} / ${offered} (${f((100 * dropped) / offered, 0)}%)` : "–"],
    ["Detector", info ? `${info.detector}` : "–"],
    ["Accelerator", info?.execution ?? "–"],
    ["Pre / inf / post p50", fused ? `fused: ${f(p50(inf))} ms` : `${f(p50(pre))} / ${f(p50(inf))} / ${f(p50(post))} ms`],
    ["Frame→tracks p50 / p95", `${f(p50(e2e))} / ${f(p95(e2e))} ms`],
    ["Tracker p50", `${f(p50(trk), 2)} ms`],
    ["Visible tracks", lastOut ? String(visible(lastOut).length) : "0"],
    ["Track IDs issued", String(tracker.idsIssued)],
    ["WASM threads", self.crossOriginIsolated ? "multi (isolated)" : "1 (not isolated)"],
  ];
  statsEl.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    return [dt, dd];
  }));
  if (info?.decision) statsEl.title = info.decision;
}
setInterval(renderStats, 300);

// ---- mode switching -------------------------------------------------------------------
function stopSource() {
  if (!source) return;
  finishStream();
  source.stop();
  source = null;
  frameW = frameH = 0;
  $<HTMLDivElement>("placeholder").hidden = false;
  for (const id of ["play", "pause", "restart", "stop"]) $<HTMLButtonElement>(id).disabled = true;
  $<HTMLButtonElement>("cam-stop").disabled = true;
  $<HTMLButtonElement>("cam-start").disabled = false;
  redraw();
}

function selectTab(mode: "video" | "camera") {
  stopSource();
  $("tab-video").setAttribute("aria-selected", String(mode === "video"));
  $("tab-camera").setAttribute("aria-selected", String(mode === "camera"));
  $("panel-video").hidden = mode !== "video";
  $("panel-camera").hidden = mode !== "camera";
  if (mode === "camera") void refreshCameras();
}
$("tab-video").onclick = () => selectTab("video");
$("tab-camera").onclick = () => selectTab("camera");

// ---- uploaded video -------------------------------------------------------------------
async function openVideo(input: VideoInput) {
  stopSource();
  resetTracking();
  const src = new VideoFileSource(video, input);
  src.onError = (e) => {
    setStatus(e.message, true);
    stopSource();
  };
  try {
    await src.open();
  } catch (e) {
    setStatus((e as Error).message, true);
    return;
  }
  source = src;
  $<HTMLDivElement>("placeholder").hidden = true;
  for (const id of ["play", "pause", "restart", "stop"]) $<HTMLButtonElement>(id).disabled = false;
  const i = src.info();
  setStatus(`${i.label}: ${i.width}×${i.height}, ${i.durationS?.toFixed(1) ?? "?"} s`);
  video.onended = () => {
    finishStream();
    setStatus("Video finished. Restart to replay, or export the results.");
  };
  src.start(onFrame);
  await src.play().catch(() => setStatus("Press Play to start", false));
}

$<HTMLInputElement>("file").onchange = (ev) => {
  const file = (ev.target as HTMLInputElement).files?.[0];
  if (file) void openVideo(file);
};

// YouTube link: the local dev/preview server looks up a direct stream (yt-dlp) and relays it
// same-origin, so the <video> element is readable by the detector. Nothing is saved.
async function loadYoutube() {
  const link = $<HTMLInputElement>("yt-url").value.trim();
  if (!link) return;
  const btn = $<HTMLButtonElement>("yt-load");
  btn.disabled = true;
  stopSource();
  setStatus("Looking up the video…");
  try {
    const r = await fetch(`/api/yt/resolve?url=${encodeURIComponent(link)}`);
    const body = await r.json().catch(() => ({}));
    if (r.status === 404 && !body.error) throw new Error("YouTube links need the local dev server: run npm run dev (or npm run preview) from the web folder");
    if (!r.ok) throw new Error(body.error ?? `lookup failed (${r.status})`);
    await openVideo({ url: body.path, label: body.title ?? "YouTube video" });
  } catch (e) {
    setStatus((e as Error).message, true);
  } finally {
    btn.disabled = false;
  }
}
$("yt-load").onclick = () => void loadYoutube();
$("yt-url").addEventListener("keydown", (e) => {
  if ((e as KeyboardEvent).key === "Enter") void loadYoutube();
});
$("play").onclick = () => void (source instanceof VideoFileSource && source.play());
$("pause").onclick = () => source instanceof VideoFileSource && source.pause();
$("restart").onclick = () => {
  if (!(source instanceof VideoFileSource)) return;
  finishStream();
  source.restart();
  resetTracking();
  void source.play();
};
$("stop").onclick = () => stopSource();

// ---- live camera ----------------------------------------------------------------------
async function refreshCameras() {
  const sel = $<HTMLSelectElement>("camera-select");
  try {
    const cams = await CameraSource.list();
    const current = sel.value;
    sel.replaceChildren(new Option("Default camera", ""), ...cams.map((c, i) => new Option(c.label || `Camera ${i + 1}`, c.deviceId)));
    sel.value = current;
  } catch {
    /* enumerateDevices unavailable: keep the default entry */
  }
}

$("cam-start").onclick = async () => {
  stopSource();
  resetTracking();
  const cam = new CameraSource(video);
  cam.onEnded = () => {
    setStatus("Camera stream ended (unplugged, permission revoked, or taken by another app). Press Start camera to retry.", true);
    stopSource();
  };
  try {
    await cam.open($<HTMLSelectElement>("camera-select").value || null);
  } catch (e) {
    setStatus(e instanceof CameraError ? e.message : `camera error: ${(e as Error).message}`, true);
    return;
  }
  source = cam;
  $<HTMLDivElement>("placeholder").hidden = true;
  $<HTMLButtonElement>("cam-stop").disabled = false;
  $<HTMLButtonElement>("cam-start").disabled = true;
  const i = cam.info();
  setStatus(`${i.label}: ${i.width}×${i.height}`);
  void refreshCameras(); // labels are only visible after permission is granted
  cam.start(onFrame);
};
$("cam-stop").onclick = () => stopSource();

// ---- settings & export ----------------------------------------------------------------
for (const id of ["detector", "input-size", "execution"]) $(id).onchange = () => void initDetector();
$("tracker").onchange = () => resetTracking();
$("detect-every").onchange = () => resetTracking();
$("show-raw").onchange = () => redraw();

$("export").onclick = () => {
  const s = settings();
  const header = JSON.stringify({ type: "header", source: source?.info() ?? null, detector: client.info, tracker: { name: s.tracker, ...TRACKER_CFG },
    detect_every: s.detectEvery, user_agent: navigator.userAgent, exported_at: new Date().toISOString() });
  const blob = new Blob([[header, ...records].join("\n") + "\n"], { type: "application/x-ndjson" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `facetrack-${Date.now()}.jsonl`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

window.addEventListener("pagehide", () => {
  source?.stop();
  client.terminate();
});

void initDetector();
