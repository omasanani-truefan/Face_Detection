// Optional network test of the YouTube link feature in the real app (headless Chrome).
//   npm run build && node scripts/youtube-smoke.mjs [youtube-url]
// Needs internet and yt-dlp. Streams the video through the local bridge, waits for tracks,
// and checks that bad links produce a clear error. Nothing is saved.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const url = process.argv[2] ?? "https://www.youtube.com/watch?v=5nGBIincOIo";
const PORT = 4181;
const chrome = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean).find((p) => existsSync(p));
if (!chrome) throw new Error("no Chrome found; set CHROME_PATH");

const server = spawn(process.execPath, [join(web, "node_modules/vite/bin/vite.js"), "preview", "--port", String(PORT), "--strictPort"], { cwd: web, stdio: "pipe" });
await new Promise((res, rej) => {
  server.stdout.on("data", (d) => String(d).includes(String(PORT)) && res());
  setTimeout(() => rej(new Error("vite preview did not start")), 30000);
});
const browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
const failures = [];
const check = (cond, msg) => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${msg}`);
  if (!cond) failures.push(msg);
};
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.setViewport({ width: 1440, height: 900 });
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "load" });
  await page.waitForFunction(() => /ready on/.test(document.getElementById("status").textContent), { timeout: 60000 });

  // A link that is not YouTube must be refused with a readable message.
  await page.type("#yt-url", "https://example.com/video");
  await page.click("#yt-load");
  await page.waitForFunction(() => /only YouTube links/.test(document.getElementById("status").textContent), { timeout: 10000 });
  check(true, "non-YouTube link refused with a clear message");

  // A real link: resolve, stream, detect, track.
  await page.$eval("#yt-url", (e) => (e.value = ""));
  await page.type("#yt-url", url);
  await page.click("#yt-load");
  await page.waitForFunction(() => document.getElementById("video").videoWidth > 0, { timeout: 90000 });
  const label = await page.$eval("#status", (e) => e.textContent);
  console.log("status:", label);
  await page.waitForFunction(() => document.querySelectorAll("#events li").length >= 2, { timeout: 90000 });
  await new Promise((r) => setTimeout(r, 4000));
  mkdirSync(resolve(web, "..", "results", "demo"), { recursive: true });
  await page.screenshot({ path: resolve(web, "..", "results", "demo", "web_app_youtube.png") });
  const events = await page.$$eval("#events li", (lis) => lis.map((l) => l.children[2].textContent).reverse());
  const stats = await page.$$eval("#stats dt", (dts) => Object.fromEntries(dts.map((d) => [d.textContent, d.nextElementSibling.textContent])));
  console.log("events:", events.slice(0, 8).join(" | "));
  console.log("stats:", JSON.stringify(stats));
  check(events.some((e) => e.startsWith("started")), "YouTube video: tracks started");
  check(Number(stats["Processed FPS"]) > 0, `YouTube video: processing (${stats["Processed FPS"]} FPS)`);
  const t = await page.$eval("#video", (v) => v.currentTime);
  check(t > 1, `YouTube video: playing (currentTime ${t.toFixed(1)} s)`);
  check(await page.$eval("#stop", (b) => !b.disabled), "YouTube video: controls enabled like an uploaded file");
  check(errors.length === 0, `no page errors (${errors.slice(0, 3).join(" / ")})`);
} finally {
  await browser.close();
  server.kill();
}
if (failures.length) {
  console.error(`${failures.length} check(s) failed`);
  process.exit(1);
}
