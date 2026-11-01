# FaceTrack — lightweight face detection and temporary multi-face tracking

Detects every visible face in a video or camera stream and gives each one a **temporary
Track ID** that stays stable while the face is continuously tracked (through brief
occlusions and missed detections), and ends a short, configurable time after the face
leaves. Built to run on CPUs, in browsers and on phones, on-device.

* **Phase 1–2 (this repo):** detection + tracking, with lifecycle events
  (`track_started`, `track_updated`, `track_lost`, `track_recovered`, `track_ended`).
* **Phase 3 (not implemented):** persistent person identity. Interfaces are prepared in
  [`src/facetrack/identity.py`](src/facetrack/identity.py); see
  [docs/REPORT.md](docs/REPORT.md) §9. A returning person gets a **new** Track ID here by
  design — Track IDs are not person IDs.

## Start here

| You want | Open |
|---|---|
| The benchmark in one page | [BENCHMARK.md](BENCHMARK.md) |
| The 3-page benchmark report (all 27 detector × tracker combinations, why the suggested ResNet / ArcFace / U-Net models were not used) | [docs/BENCHMARK_REPORT.pdf](docs/BENCHMARK_REPORT.pdf) |
| The full research, method, licences and recommendation | [docs/REPORT.md](docs/REPORT.md) (tables: [docs/results_tables.md](docs/results_tables.md)) |
| Raw numbers behind the benchmark | [results/youtube/summary.json](results/youtube/summary.json) |

### Run it yourself (about 20 minutes plus downloads; Python 3.10+, Node 20+ for the web app, no GPU)

```bash
git clone <repo-url> && cd Face-Detection
python -m venv .venv && .venv/Scripts/activate        # Windows; source .venv/bin/activate elsewhere
python -m pip install --upgrade pip                     # Python 3.10's bundled pip is too old for this install
pip install -e ".[mediapipe,bench,youtube,report,dev]"
python scripts/download_models.py                       # YuNet, SCRFD-500M, BlazeFace (checksum-verified)

# 1. Watch it work: stream a YouTube video through detector + tracker (nothing is saved)
facetrack-demo --url "https://www.youtube.com/watch?v=5nGBIincOIo"

# 2. Re-run the benchmark on the two videos used in the report
python scripts/download_suggested_models.py             # optional heavy models, 305 MB, non-commercial weights
python scripts/bench_youtube.py https://www.youtube.com/watch?v=YzcawvDGe4Y https://www.youtube.com/watch?v=5nGBIincOIo
python scripts/build_benchmark_report.py                # rewrites BENCHMARK.md and docs/BENCHMARK_REPORT.*

# 3. Browser app, with the YouTube link box
cd web && npm install && npm run dev                    # http://localhost:5173
```

`python scripts/check_setup.py` tells you what is missing on your machine and how to fix it. You can also open this
folder in [Claude Code](https://claude.com/claude-code) and ask it to set the project up: `CLAUDE.md` gives it the steps,
the known problems and the rules for this repo.

**If YouTube answers "Sign in to confirm you're not a bot"** (it does this to automated lookups from
some networks, especially after many lookups), tell yt-dlp to use your own browser's login, then
run the same command again: `FACETRACK_YT_COOKIES_BROWSER=chrome` (PowerShell: `$env:FACETRACK_YT_COOKIES_BROWSER="chrome"`),
or `FACETRACK_YT_COOKIES_FILE=path/to/cookies.txt`. This applies to the demo, the benchmark and the
browser app (set it before `npm run dev`). Use your own account on your own machine and never commit cookies.

Benchmark timings depend on the machine: run step 2 on an idle, plugged-in computer and expect
different milliseconds than in the report (the ranking should hold). Only use videos you are
allowed to use. There is no `LICENSE` file yet: the repository owner has to choose one (see the
licence table below for the dependencies and pretrained weights).

## Layout

```text
src/facetrack/            Python reference implementation (detectors, trackers, benchmarks)
  sources.py              VideoFileSource / CameraSource -> Frame(image, frame_id, timestamp_ms)
  detectors/              YuNet (ONNX Runtime + OpenCV), SCRFD-500M-KPS, MediaPipe BlazeFace
  tracking/               shared lifecycle (base.py) + IoU / SORT / ByteTrack association
  pipeline.py             source-agnostic detector -> tracker -> observers loop
  identity.py             Phase-3 extension points (interfaces only)
  metrics/                COCO detection metrics (pycocotools), CLEAR-MOT / IDF1 / HOTA
  bench/                  synthetic-motion sequence generator, ONNX FLOP counter
  cli/demo.py             desktop demo (video file or webcam)
web/                      on-device browser app (TypeScript, Vite, Web Worker)
scripts/                  download, dataset preparation, every benchmark, report tables
BENCHMARK.md              one-page benchmark summary (generated)
configs/default.yaml      runtime defaults for the demo
tests/                    pytest suite (106 tests)
docs/                     REPORT.md, results_tables.md, research/published_reference.md
```

## Setup (Python 3.10+, CPU only)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[mediapipe,bench,dev]"
python scripts/download_models.py # fetches + sha256-verifies every model, derives dynamic-shape YuNet
pytest                            # 106 tests (detector tests skip if WIDER is not prepared)
```

## Run the demo

```bash
facetrack-demo --video path/to/clip.mp4                 # uploaded video (default mode)
facetrack-demo --camera 0                               # live camera, same pipeline
facetrack-demo --url "https://www.youtube.com/watch?v=VIDEO_ID"   # online video, streamed live, never saved
facetrack-demo --video clip.mp4 --detector scrfd_500m_kps --tracker sort --input-size 0
facetrack-demo --video clip.mp4 --no-display --export out/clip.jsonl --export-mot out/clip.txt --save-video out/clip.mp4
facetrack-demo --list-cameras
```

Keys: `space` play/pause · `n` step · `r` restart · `q`/`Esc` stop. `--input-size 0` runs
the detector at the native frame resolution (needed for distant faces, see the report).
Everything else is in [configs/default.yaml](configs/default.yaml).

## Run the browser app

```bash
cd web
npm install
npm run dev            # http://localhost:5173 — upload a video, or switch to Live camera
npm run build && npm run preview
npm test               # TypeScript tracker must match the Python reference frame by frame
npm run test:e2e       # headless Chrome: upload, re-entry ID, codec error, fake camera
```

**YouTube in the browser app:** paste a link in the *Uploaded video* tab. The local server
(`npm run dev` / `npm run preview`) looks the stream up with yt-dlp and relays it same-origin, so
the detector can read the frames (a YouTube iframe cannot be read). Nothing is saved; it needs
`pip install -e ".[youtube]"` in the repo root (the `.venv` is found automatically, or set
`FACETRACK_PYTHON`) and only works while the local server runs, not on a static deployment.
Use only videos you are allowed to use.

Detection runs in a Web Worker (ONNX Runtime Web for YuNet/SCRFD, MediaPipe Tasks for
BlazeFace). *Auto* accelerator mode times WebGPU and WASM on the device and keeps the
faster one, falling back to WASM where WebGPU is missing. Sample H.264 clips:
`python scripts/make_web_samples.py` → `data/samples_h264/`. The camera needs `https://`
or `localhost`.

## Reproduce the benchmarks

```bash
# data (WIDER FACE val is non-commercial research data; evaluation only)
#   put WIDER_val.zip + wider_face_split.zip in data/_download/widerface/ and unzip
python scripts/prepare_widerface.py          # COCO json, deterministic dev/test split
python scripts/build_sequences.py            # 48 tracking sequences with exact GT identities

python scripts/bench_detection_accuracy.py   # WIDER test-split AP / recall by face size
python scripts/bench_tracking.py             # IoU vs SORT vs ByteTrack on identical detections
python scripts/bench_youtube.py URL [URL ...]   # all models on streamed online videos -> results/youtube/
python scripts/bench_detection_latency.py --sustained 180   # run on an idle machine, on mains power
python scripts/export_bench_frames.py && (cd web && npm run build && npm run bench:browser)
python scripts/make_report_tables.py         # -> docs/results_tables.md
```

Each script records input checksums, model checksums, thresholds, input resolution,
runtime versions and hardware next to its results in `results/<benchmark>/`.

## Licences — read before shipping

| Component | Code | Pretrained weights |
|---|---|---|
| YuNet (OpenCV Zoo) | MIT | MIT (trained on WIDER FACE, CC BY-NC-ND — see report) |
| SCRFD (InsightFace) | MIT | **non-commercial research only** |
| MediaPipe BlazeFace | Apache-2.0 | Apache-2.0 model cards |
| ONNX Runtime / OpenCV / MediaPipe runtimes | MIT / Apache-2.0 / Apache-2.0 | — |
| SORT / ByteTrack here | clean implementations from the papers (the reference SORT code is GPL-3.0 and is not used) | — |

Details and the commercial-use recommendation: [docs/REPORT.md](docs/REPORT.md) §10.
