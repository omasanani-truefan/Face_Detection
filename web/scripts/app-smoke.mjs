// End-to-end check of the real app in headless Chrome:
//   1. upload an H.264 video -> detection + tracking run, IDs appear, re-entry gets a new ID
//   2. upload an undecodable file -> clear error, no crash
//   3. live camera (Chrome fake device fed with a Y4M of real faces) -> tracks, then Stop
//      releases the device
// Needs: npm run build; python scripts/make_web_samples.py
import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const samples = resolve(web, "..", "data", "samples_h264");
const PORT = 4180;
const chrome = [process.env.CHROME_PATH, "C:/Program Files/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean).find((p) => existsSync(p));
if (!chrome) throw new Error("no Chrome found; set CHROME_PATH");

const server = spawn(process.execPath, [join(web, "node_modules/vite/bin/vite.js"), "preview", "--port", String(PORT), "--strictPort"], { cwd: web, stdio: "pipe" });
await new Promise((res, rej) => {
  server.stdout.on("data", (d) => String(d).includes(String(PORT)) && res());
  setTimeout(() => rej(new Error("vite preview did not start")), 30000);
});

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
    `--use-file-for-fake-video-capture=${join(samples, "fake_camera_multi_enter.y4m")}`, "--autoplay-policy=no-user-gesture-required"],
});
const failures = [];
const check = (cond, msg) => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${msg}`);
  if (!cond) failures.push(msg);
};
try {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("response", (r) => r.status() >= 400 && errors.push(`${r.status()} ${r.url()}`));
  await page.goto(`http://localhost:${PORT}/`, { waitUntil: "load" });
  await page.waitForFunction(() => /ready on/.test(document.getElementById("status").textContent), { timeout: 60000 });
  console.log("detector:", await page.$eval("#status", (e) => e.textContent));

  // 1. Uploaded video (default mode).
  const input = await page.$("#file");
  await page.setViewport({ width: 1440, height: 900 });
  await input.uploadFile(join(samples, "reentry.mp4"));
  await new Promise((r) => setTimeout(r, 7000)); // mid-clip: the returning person is on screen
  mkdirSync(resolve(web, "..", "results", "demo"), { recursive: true });
  await page.screenshot({ path: resolve(web, "..", "results", "demo", "web_app_video.png") });
  await page.waitForFunction(() => /Video finished/.test(document.getElementById("status").textContent), { timeout: 120000 });
  const events = await page.$$eval("#events li", (lis) => lis.map((l) => l.children[2].textContent).reverse());
  const stats = await page.$$eval("#stats dt", (dts) => Object.fromEntries(dts.map((d) => [d.textContent, d.nextElementSibling.textContent])));
  console.log("events:", events.join(" | "));
  console.log("stats:", JSON.stringify(stats));
  const started = events.filter((e) => e.startsWith("started")).map((e) => Number(e.split("ID ")[1]));
  check(started.length >= 3, `video: >=3 tracks started (got ${started})`);
  check(events.some((e) => e === "ended · ID 1"), "video: the person who left has track_ended");
  check(started.includes(3) && !events.includes("recovered · ID 1"), "video: re-entry got a new Track ID, not the old one");
  check(await page.$eval("#export", (b) => !b.disabled), "video: export enabled after processing");
  check(Number(stats["Processed FPS"]) > 0 || stats["Pre / inf / post p50"] !== "–", "video: performance stats populated");

  // 2. A file the browser cannot decode (MPEG-4 Part 2 written by OpenCV).
  await input.uploadFile(resolve(web, "..", "data", "sequences", "test", "crossing.mp4"));
  await page.waitForFunction(() => document.getElementById("status").classList.contains("error"), { timeout: 30000 }).catch(() => {});
  const msg = await page.$eval("#status", (e) => e.textContent);
  check(/cannot decode/.test(msg), `video: undecodable codec reported clearly ("${msg.slice(0, 80)}")`);

  // 3. Live camera through the same pipeline.
  await page.click("#tab-camera");
  await page.click("#cam-start");
  await page.waitForFunction(() => document.querySelectorAll("#events li").length >= 2, { timeout: 30000 }).catch(() => {});
  const camEvents = await page.$$eval("#events li", (lis) => lis.map((l) => l.children[2].textContent));
  check(camEvents.some((e) => e.startsWith("started")), `camera: tracks started (${camEvents.slice(0, 4).join(" | ")})`);
  await page.click("#cam-stop");
  const released = await page.evaluate(() => document.getElementById("video").srcObject === null);
  check(released, "camera: Stop releases the stream");
  check(errors.length === 0, `no page errors (${errors.slice(0, 3).join(" / ")})`);
} finally {
  await browser.close();
  server.kill();
}
if (failures.length) {
  console.error(`${failures.length} check(s) failed`);
  process.exit(1);
}
