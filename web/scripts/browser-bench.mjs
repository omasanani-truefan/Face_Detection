// Drives the installed Chrome/Edge through bench.html and the app smoke test.
//   npm run build && npm run bench:browser [-- --headful]
// Writes ../results/browser_latency/summary.json.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const PORT = 4179;
const CANDIDATES = [
  process.env.CHROME_PATH,
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);
const executablePath = CANDIDATES.find((p) => existsSync(p));
if (!executablePath) throw new Error("no Chrome/Edge found; set CHROME_PATH");

const server = spawn(process.execPath, [join(web, "node_modules/vite/bin/vite.js"), "preview", "--port", String(PORT), "--strictPort"], { cwd: web, stdio: "pipe" });
await new Promise((resolve, reject) => {
  server.stdout.on("data", (d) => String(d).includes(String(PORT)) && resolve());
  server.on("exit", (c) => reject(new Error(`vite preview exited ${c}`)));
  setTimeout(() => reject(new Error("vite preview did not start")), 30000);
});

const headful = process.argv.includes("--headful");
const browser = await puppeteer.launch({
  executablePath,
  headless: !headful,
  args: ["--enable-unsafe-webgpu", "--enable-features=WebGPU", "--ignore-gpu-blocklist", "--no-first-run"],
});
try {
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(`http://localhost:${PORT}/bench.html?auto=1&res=480p,720p`, { waitUntil: "load" });
  await page.waitForFunction("window.__bench", { timeout: 15 * 60 * 1000, polling: 1000 });
  const bench = await page.evaluate("window.__bench");
  const out = join(web, "..", "results", "browser_latency");
  mkdirSync(out, { recursive: true });
  const version = await browser.version();
  writeFileSync(join(out, "summary.json"), JSON.stringify({ browser: version, headless: !headful, ...bench, console: logs.slice(-50) }, null, 1));
  for (const r of bench.results)
    console.log(r.error ? `${r.config.padEnd(42)} unavailable: ${r.error.slice(0, 120)}`
      : `${r.config.padEnd(42)} e2e p50 ${r.end_to_end_incl_transfer.p50_ms.toFixed(2).padStart(7)} ms  p95 ${r.end_to_end_incl_transfer.p95_ms.toFixed(2).padStart(7)} ms  inf p50 ${r.inference.p50_ms.toFixed(2)} ms  [${r.execution_used}]`);
  console.log(`browser: ${version}; webgpu: ${bench.environment.webgpu}; isolated: ${bench.environment.crossOriginIsolated}`);
} finally {
  await browser.close();
  server.kill();
}
