// Local-only YouTube bridge for the dev / preview server.
//
// A browser cannot read pixels from YouTube's player (cross-origin), so detection could
// never run on an embedded video. This plugin makes the page's own origin serve the video:
//
//   GET /api/yt/resolve?url=<youtube link>   yt-dlp looks up a direct H.264 stream address
//   GET /api/yt/stream/<id>                  the bytes are relayed (Range supported), same-origin
//
// Nothing is written to disk: the video is relayed to the <video> element as it plays.
// It only exists while `vite` / `vite preview` runs on your machine; it is not part of the
// built site. Only use videos you are allowed to use; check the licence and the site's terms.
//
// Needs yt-dlp: pip install -e ".[youtube]" in the repo root (the .venv is found
// automatically), or set FACETRACK_PYTHON to a Python that has yt_dlp installed.
// If YouTube demands a sign-in, set FACETRACK_YT_COOKIES_BROWSER=chrome (or FACETRACK_YT_COOKIES_FILE).
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const YT_HOSTS = new Set(["youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"]);
const STREAM_HOST = /\.googlevideo\.com$/;
// One progressive-style H.264 video stream: every browser decodes it, audio is not needed.
const FORMAT = "bv*[vcodec^=avc1][ext=mp4][protocol=https][height<=720]/b[ext=mp4][protocol=https][height<=720]";
const MAX_ENTRIES = 8;
const RESOLVE_TIMEOUT_MS = 60_000;

/** @type {Map<string, {url: string, headers: Record<string, string>}>} */
const streams = new Map();

function pythonCandidates() {
  const root = resolve(here, "..", "..");
  return [
    process.env.FACETRACK_PYTHON,
    join(root, ".venv", "Scripts", "python.exe"),
    join(root, ".venv", "bin", "python"),
    process.platform === "win32" ? "python" : "python3",
  ].filter((p) => p && (!p.includes("/") && !p.includes("\\") ? true : existsSync(p)));
}

function ytdlpJson(link) {
  const py = pythonCandidates()[0];
  return new Promise((res, rej) => {
    const args = ["-I", "-m", "yt_dlp", "-J", "--no-playlist", "--no-warnings", "-f", FORMAT];
    // YouTube sometimes demands a sign-in before answering automated lookups; let the user lend their own browser login.
    if (process.env.FACETRACK_YT_COOKIES_BROWSER) args.push("--cookies-from-browser", process.env.FACETRACK_YT_COOKIES_BROWSER);
    if (process.env.FACETRACK_YT_COOKIES_FILE) args.push("--cookies", process.env.FACETRACK_YT_COOKIES_FILE);
    const p = spawn(py, [...args, link], { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    let err = "";
    const timer = setTimeout(() => {
      p.kill();
      rej(new Error("timed out while looking up the video"));
    }, RESOLVE_TIMEOUT_MS);
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    p.on("error", (e) => {
      clearTimeout(timer);
      rej(new Error(`cannot start Python (${py}): ${e.message}. Set FACETRACK_PYTHON.`));
    });
    p.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        try {
          res(JSON.parse(out));
        } catch {
          rej(new Error("yt-dlp returned unreadable output"));
        }
      } else if (/confirm you.{1,3}re not a bot/.test(err)) {
        rej(new Error("YouTube asked for a sign-in before it would give out this video. Start the server with FACETRACK_YT_COOKIES_BROWSER=chrome (or firefox, edge) so yt-dlp can use your browser's login, or try again later."));
      } else if (/No module named yt_dlp/.test(err)) {
        rej(new Error('yt-dlp is not installed: run pip install -e ".[youtube]" in the repository root'));
      } else {
        const line = err.split("\n").filter((l) => /ERROR/.test(l)).pop() ?? err.trim().split("\n").pop() ?? "";
        rej(new Error(line.replace(/^ERROR:\s*(\[[^\]]+\]\s*)?/, "") || "yt-dlp failed"));
      }
    });
  });
}

function json(res, status, body) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(body));
}

/** Refuse requests that did not come from a page on this same host (CSRF / DNS rebinding). */
function sameHost(req) {
  const host = req.headers.host ?? "";
  if (!/^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/.test(host)) return false;
  const origin = req.headers.origin;
  return !origin || new URL(origin).host === host;
}

async function handleResolve(req, res, query) {
  let link;
  try {
    link = new URL(query.get("url") ?? "");
  } catch {
    return json(res, 400, { error: "that is not a valid link" });
  }
  if (!["http:", "https:"].includes(link.protocol) || !YT_HOSTS.has(link.hostname)) {
    return json(res, 400, { error: "only YouTube links (youtube.com, youtu.be) are supported" });
  }
  try {
    const info = await ytdlpJson(link.href);
    if (info.is_live) return json(res, 422, { error: "live streams are not supported; use a normal video or the Live camera tab" });
    const fmt = info.requested_formats?.find((f) => f.vcodec && f.vcodec !== "none") ?? info;
    if (!fmt.url || !STREAM_HOST.test(new URL(fmt.url).hostname)) return json(res, 422, { error: "no directly playable H.264 stream for this video" });
    const id = randomBytes(9).toString("base64url");
    streams.set(id, { url: fmt.url, headers: fmt.http_headers ?? info.http_headers ?? {} });
    while (streams.size > MAX_ENTRIES) streams.delete(streams.keys().next().value);
    json(res, 200, {
      id, title: info.title, channel: info.channel ?? info.uploader ?? null, durationS: info.duration ?? null,
      width: fmt.width ?? null, height: fmt.height ?? null, fps: fmt.fps ?? null, path: `/api/yt/stream/${id}`,
    });
  } catch (e) {
    json(res, 502, { error: e.message });
  }
}

async function handleStream(req, res, id) {
  const entry = streams.get(id);
  if (!entry) return json(res, 404, { error: "unknown or expired stream; load the link again" });
  const ctl = new AbortController();
  res.on("close", () => ctl.abort());
  const headers = { ...entry.headers };
  if (req.headers.range) headers.Range = req.headers.range;
  let up;
  try {
    up = await fetch(entry.url, { headers, signal: ctl.signal });
  } catch (e) {
    if (!res.headersSent) json(res, 502, { error: `could not reach the video server: ${e.message}` });
    return;
  }
  if (up.status === 403 || up.status === 410) streams.delete(id); // the signed address expired
  res.statusCode = up.status;
  for (const h of ["content-type", "content-length", "content-range", "accept-ranges"]) {
    const v = up.headers.get(h);
    if (v) res.setHeader(h, v);
  }
  res.setHeader("Cache-Control", "no-store");
  if (!up.body) return res.end();
  Readable.fromWeb(up.body).on("error", () => res.destroy()).pipe(res);
}

export function youtubeBridge() {
  const middleware = (req, res, next) => {
    const u = new URL(req.url ?? "/", "http://localhost");
    if (!u.pathname.startsWith("/api/yt/")) return next();
    if (!sameHost(req)) return json(res, 403, { error: "forbidden" });
    if (u.pathname === "/api/yt/resolve") return void handleResolve(req, res, u.searchParams);
    const m = u.pathname.match(/^\/api\/yt\/stream\/([\w-]+)$/);
    if (m) return void handleStream(req, res, m[1]);
    json(res, 404, { error: "not found" });
  };
  return {
    name: "facetrack-youtube-bridge",
    configureServer: (server) => void server.middlewares.use(middleware),
    configurePreviewServer: (server) => void server.middlewares.use(middleware),
  };
}
