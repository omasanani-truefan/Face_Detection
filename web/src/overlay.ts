// Canvas overlay drawn over the <video>; canvas pixels = source-frame pixels.
import type { Track } from "./tracking/tracker.ts";
import type { Detection } from "./types.ts";

export function trackColor(id: number): string {
  const h = Math.round(((id * 0.618033988749895) % 1) * 360);
  return `hsl(${h} 90% 55%)`;
}

export function drawOverlay(
  canvas: HTMLCanvasElement, w: number, h: number, tracks: Track[], raw: Detection[] | null, showTentative: boolean,
): void {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const g = canvas.getContext("2d")!;
  g.clearRect(0, 0, w, h);
  const lw = Math.max(2, Math.round(Math.min(w, h) / 300));
  const font = Math.max(12, Math.round(Math.min(w, h) / 36));
  g.font = `600 ${font}px system-ui, sans-serif`;
  g.textBaseline = "bottom";
  if (raw) {
    g.strokeStyle = "rgba(255,255,255,0.55)";
    g.lineWidth = 1;
    g.setLineDash([]);
    for (const d of raw) g.strokeRect(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]);
  }
  for (const t of tracks) {
    if (t.state === "tentative" && !showTentative) continue;
    const [x1, y1, x2, y2] = t.box;
    const solid = t.state === "confirmed" && t.matched;
    const color = t.state === "tentative" ? "rgba(200,200,200,0.8)" : trackColor(t.trackId);
    g.strokeStyle = color;
    g.lineWidth = solid ? lw + 1 : lw;
    g.setLineDash(solid ? [] : [lw * 3, lw * 3]);
    g.strokeRect(x1, y1, x2 - x1, y2 - y1);
    g.setLineDash([]);
    if (solid && t.keypoints) {
      g.fillStyle = color;
      for (const [kx, ky] of t.keypoints) {
        g.beginPath();
        g.arc(kx, ky, lw, 0, Math.PI * 2);
        g.fill();
      }
    }
    const label = `ID ${t.trackId} · ${t.confidence.toFixed(2)} · ${t.state}`;
    const tw = g.measureText(label).width + 8;
    const ty = Math.max(font + 4, y1);
    g.fillStyle = color;
    g.fillRect(x1, ty - font - 4, tw, font + 4);
    g.fillStyle = "#000";
    g.fillText(label, x1 + 4, ty - 2);
  }
}
