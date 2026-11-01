# Lightweight real-time face detection and multi-face tracking — technical report

Date: 2026-11-01 · Repository: `Face-Detection/` · Phases covered: 1 (detection) and 2 (temporary tracking).

**How to read the numbers in this report**

| Label | Meaning |
|---|---|
| **Measured** | Produced by the scripts in `scripts/` on the machine in §2. Every measured table is generated from `results/*/summary.json` into [results_tables.md](results_tables.md); nothing is hand-typed. |
| **Published** | Claimed by the model's authors or vendor; sources in [research/published_reference.md](research/published_reference.md). Never mixed into measured tables. |
| **Estimate** | Reasoned, not measured. Marked explicitly where used. |

---

## 0. Recommendation in one paragraph

**YuNet (OpenCV Zoo 2023mar, fp32) + ByteTrack + ONNX Runtime** (WASM in browsers with measured WebGPU opt-in; ORT Mobile/XNNPACK or OpenCV in apps), with the detector at **640 px for close range and native resolution for far range**, `max_lost_ms` 1000–1500, and `detect_every` 2 on weak devices.

* **YuNet** is the most accurate *shippable-candidate* detector: 0.81 / 0.93 recall on medium / large faces at 640; AP50 0.654 at native resolution, ahead of SCRFD at 640. It is the smallest model (0.23 MB, 53k parameters) and the fastest ONNX model: 5.5 ms per frame on this CPU (9.8 ms on one core), 19.7 ms in Chrome WASM.
* **ByteTrack** trades ≤ 0.015 HOTA against the plain IoU tracker for **4× less fragmentation, 5× fewer false tracks and the fewest ID switches**, which is what Phase 3's per-track identity resolution needs.
* **Not chosen:**
  * **SCRFD-500M** is more accurate (+0.09 AP50 at 640) but its weights are **non-commercial**.
  * **MediaPipe BlazeFace** is the fastest (1.9 ms short-range) and has the cleanest licence, but it is a **close-range detector**: recall 0.00 / 0.35 for faces under 5% / at 5–10% of frame height with full-range. It is the right choice for selfie or video-call distance on mobile, or as the licence-safe fallback, not for room-scale cameras.
* **Caveats.**
  * YuNet's training-data chain (WIDER FACE, InsightFace labels) needs **legal review** before commercial use (§10).
  * Tracking was benchmarked on synthetic-motion clips built from real faces, not recorded video (§4.2).
  * Phones were not measured (§11).

---

## 1. Scope

* Detect every visible face per frame (box, confidence, landmarks), on CPUs, in browsers and on phones, without a GPU and without sending frames to a server.
* Give each continuously observed face a **temporary Track ID**; keep it through brief occlusions and missed detections; end the track (with a `track_ended` event) a configurable time after the face disappears; give a returning person a **new** Track ID.
* Not in scope: persistent identity, embeddings, ArcFace, enrolment, fine-tuning. §9 specifies the interfaces Phase 3 will plug into.

## 2. Test environment (measured runs)

* **Desktop CPU:** Intel Core i7-12650H (10 cores / 16 threads), 15.7 GB RAM, Windows 11, on mains power. Runtime packages and versions are written into every `summary.json`. The laptop has an RTX 3050, which was deliberately **not** used: every provider is CPU-only.
* **Browser:** Chrome (version recorded in `results/browser_latency/summary.json`), headless, driven by `web/scripts/browser-bench.mjs`, on the same laptop.
* **Not measured here:** phones, mobile browsers, lower-end hardware, battery drain and thermal behaviour on a phone. §11 lists exactly what still needs a device run, and `web/bench.html` is the tool for it (open it on the phone and press Run).

---

## 3. Candidates

Published facts and their sources are in [research/published_reference.md](research/published_reference.md). This section summarises them alongside what we measured.

### 3.1 YuNet (OpenCV Zoo) — **included**

* **Architecture.** Tiny anchor-free detector: a depthwise-separable backbone, a "tiny FPN" neck, a decoupled head at strides 8/16/32 with one prior per cell, and 5 landmarks. The zoo's `face_detection_yunet_2023mar.onnx` is byte-identical to **YuNet-s**, the smaller variant (54.6k parameters, published), not the paper's 75.9k-parameter YuNet.
* **Measured size and cost.** 232,589 B (fp32) and 100,416 B (int8); 53,121 ONNX initializer values; GFLOPs counted from the graph: 0.17 @320², 0.52 @640×480, 0.69 @640².
* **Published accuracy** (WIDER val, original size): 0.884 / 0.866 / 0.750 (zoo eval).
* **Runtimes.** OpenCV DNN (`cv2.FaceDetectorYN`, OpenCV ≥ 4.8) and any ONNX runtime. The graph declares a fixed 640×640 input, but every Reshape is `[1,-1,C]`. We derive a copy with only symbolic H/W metadata (`scripts/download_models.py`), which lets ONNX Runtime accept any stride-32 input. The zoo has since published the same change as `2026may`.
* **Verification.** Our ONNX Runtime decoder reproduces `cv2.FaceDetectorYN` (box IoU > 0.99 for ≥ 98% of detections on a 700-face crowd; the remaining difference is OpenCV's integer-box NMS). This is covered by `tests/test_detectors.py`.
* **Licence.** MIT (code and weights). Trained on WIDER FACE (see §10).

### 3.2 SCRFD-500M-KPS (InsightFace) — **included, but not shippable commercially**

* **Architecture.** A RetinaNet-style detector found by searching how to redistribute computation. The 0.5 GF variant uses a depthwise backbone, a PAFPN neck and a shared GroupNorm head; it has 2 anchors per cell at strides 8/16/32 and 5 landmarks. The `det_500m.onnx` in InsightFace's `buffalo_sc` pack is this model.
* **Measured size and cost.** 2,524,817 B; 626,332 initializer values; GFLOPs counted from the graph: 0.26 @320², 0.77 @640×480, 1.02 @640². (Published: 0.57 M params and "500 M FLOPs @VGA". Papers count FLOPs differently; we report our own count with its method.)
* **Published accuracy** (WIDER val, VGA-bounded): 90.97 / 88.44 / 69.49.
* **Licence.** The code is MIT, but the **pretrained weights are for non-commercial research only** (stated four times by InsightFace, and in a signed `MODEL.LICENSE` in the 2026 packs). We benchmark it as an accuracy reference; it is excluded from any commercial recommendation.

### 3.3 MediaPipe Face Detector / BlazeFace — **included (short-range and full-range)**

* **Architecture.** A single-shot detector built for mobile GPUs from "BlazeBlocks" (5×5 depthwise convolutions). It uses anchors at 2 strides, "blending" instead of NMS, and 6 keypoints: eyes, nose tip, mouth centre and the ear tragions (no mouth corners).
* **Input.** Short-range uses a fixed 128×128 input, for faces within about 2 m (published). Full-range uses a fixed 192×192 input (the docs say 128; the file and the model card say 192/160×192), for faces up to about 5 m. Full-range reached the Tasks API in v0.10.33 (March 2026).
* **Measured size.** 229,746 B (short) and 1,083,786 B (full). Parameter and FLOP counts for the TFLite files were not measured; the published figures are not given.
* **Runtimes.** First-party everywhere: `@mediapipe/tasks-vision` (browser, CPU wasm or WebGL GPU delegate), MediaPipe Tasks for Android and iOS, and Python.
* **Licence.** Apache-2.0 framework and model cards. The training data is Google's own, not WIDER, which makes this the cleanest licence position of the candidates.

### 3.4 RetinaFace-MobileNet0.25 — **not introduced** (optional candidate)

The brief allowed it only if the lightweight candidates struggled with small or occluded faces. They do struggle with tiny faces (§5.2), but RetinaFace-0.25 is not the remedy. On YuNet's paper protocol it is *less* accurate than YuNet on the Hard set at every input size (published: 0.271 vs 0.395 @320, 0.541 vs 0.691 @640). It is also slower (published CPU, same hardware: 57 ms vs 16.3 ms at original size). Its common PyTorch weights have no stated licence. The measured remedy for small faces is input resolution (§5.3), not a heavier model.

### 3.5 Additionally suggested models (ResNet, ArcFace, U-Net)

| Model | What it is | Decision |
|---|---|---|
| **U-Net** | An encoder–decoder for dense per-pixel **segmentation**. It outputs a mask, not boxes, landmarks or instances, so it would need extra post-processing to count or separate faces, and it costs a full-resolution decoder pass. | Excluded from the detector benchmark. Not a face detector. |
| **ResNet** | A general **feature-extraction backbone**. It only becomes a detector inside a detection architecture (RetinaFace-R50, SCRFD's larger variants). Benchmarking a "vanilla ResNet" as a face detector is not meaningful, and ResNet-50-class backbones are about 10–100× the compute of the candidates. | Excluded. If more accuracy is ever needed, evaluate a *specific* detector (SCRFD-2.5G or RetinaFace with a light backbone), not a backbone. |
| **ArcFace** | An **additive angular-margin loss** for training face-*recognition* embeddings. It does not detect faces. | Out of scope for Phases 1–2; it is the natural basis for the Phase 3 `EmbeddingExtractor` (§9). Its 5-point alignment template matches the YuNet and SCRFD landmarks (verified: same point order). |

---

#### 3.5.1 Measured side experiment (`scripts/bench_suggested_models.py`, kept apart from the main benchmark)

Same WIDER protocol (640 px, threshold from dev, metrics on test) and the same machine. The weights are InsightFace's, so they are non-commercial. Latency is on 640×480 frames.

| Model | What it stands in for | AP50 | Recall tiny / small / medium / large | CPU e2e p50 |
|---|---|---|---|---|
| SCRFD-500M-KPS | the light baseline | 0.576 | 0.16 / 0.60 / 0.85 / 0.95 | 9.3 ms |
| **SCRFD-10G-KPS** | heavy-backbone detector (the "ResNet inside a detector" case) | **0.709** | **0.32** / 0.76 / 0.92 / 0.97 | **67.5 ms** |

* The heavy detector gains **+0.13 AP50** but costs **7× the latency** (about 15 FPS on this CPU, and mobile phones would be slower). The gain is mostly small faces (recall at 16–32 px rises 0.60 → 0.76). On medium and large faces the gap is 3–7 points.
* For comparison, simply running YuNet at full resolution gets AP50 0.654 in 18 ms at 720p (§5.2). Resolution buys most of the accuracy a heavier backbone does, at a fraction of the compute. That is the evidence behind not introducing a heavier detector.

| Recognition model (Phase 3 cost only, 112×112 crop, one face) | Size | Embedding p50, default threads | p50, 1 thread |
|---|---|---|---|
| ArcFace MobileFaceNet (`w600k_mbf`) | 13 MB | **7.4 ms** | 16.1 ms |
| ArcFace ResNet-50 (`w600k_r50`) | 166 MB | 124.7 ms | 290.1 ms |

* Both give a 512-dim embedding. The ResNet-50 model is 17× slower and 13× larger. Embedding once per track (the best few crops, not every frame) keeps even that affordable; on a phone, MobileFaceNet is the realistic choice.
* This measures **cost only**. No recognition accuracy was tested, and none of this is wired into the pipeline (Phase 3).
* **U-Net was not run.** It produces masks rather than face boxes, and no pretrained face-segmentation model was available. Producing one would mean training, which is out of scope.

## 4. Data and protocol

### 4.1 Detection: WIDER FACE val, split into dev and test

* **Data.** 3,226 manually labelled images (39,708 boxes) from WIDER FACE val, converted to COCO JSON (`scripts/prepare_widerface.py`). Boxes WIDER marks *invalid* become ignore regions (`iscrowd`).
* **Split.** Deterministic and stratified by WIDER's 61 event folders: dev has 1,595 images and 19,332 faces; test has 1,631 images and 19,780 faces. Score thresholds are chosen on **dev** (max F1 @ IoU 0.5); every reported number is on **test**.
* **Evaluator.** pycocotools, with two adaptations documented in `src/facetrack/metrics/detection.py`:
  1. Size buckets are by face *height*: tiny < 16 px, small 16–32, medium 32–96, large ≥ 96. The bucket edges come from the test data, where 37% of faces are under 16 px tall.
  2. A second bucketing by face height *as a fraction of frame height* (<5%, 5–10%, 10–20%, ≥20%). For models with a fixed input size, this relative size, not pixels, decides whether a face is detectable, so it predicts behaviour as people move away from the camera.
* **Metrics.** AP50, AP75, AP[.5:.95], precision / recall / F1 and false positives per image at the dev threshold, and recall per bucket. Recall at IoU 0.3 is also reported, because box conventions differ: MediaPipe's boxes are squarer than WIDER's (median IoU of a correct detection is 0.59–0.65, against 0.80 for SCRFD), and that difference is not a detection failure.
* **Limitation.** WIDER is photographs, not video, and it is far harder than a typical webcam scene (crowds, tiny faces, heavy blur). Absolute numbers will be higher on our own footage; the *ranking* and the *size behaviour* are what transfer.

### 4.2 Tracking: synthetic-motion sequences over real faces

* **Why synthetic.** Tracking needs every face's identity in every frame, for each situation in the brief. No consented, identity-annotated video exists for this project yet. Rather than invent numbers or wait, `scripts/build_sequences.py` builds 48 clips (24 dev, 24 test, 10,080 frames) with exact ground truth. The two generators:
  * **actors:** real WIDER face crops composited onto face-free WIDER backgrounds and moved along scripts, one per situation in the brief. The scripts cover a single person, three entering together, one leaving while another stays, a passing occluder, crossing paths, a head turn, motion-blur misses, permanent exit, re-entry after 2.8 s, far→near approach, and a 5-person far group at 1280×720 and 640×360.
  * **camera:** a virtual camera pans and zooms over real WIDER group photos, so real faces in their real context move, shrink and cross the frame edges.
* **Lighting variants.** Low light with sensor noise, backlight, and uneven illumination, on four of the clips.
* **Split and GT rules.** Dev clips come only from WIDER-dev images, test clips only from WIDER-test images. A face is an ignore region when it is < 50% visible, foreshortened past profile, or under 10 px. A visual audit of rendered frames led to two generator fixes: masks are centred on the face, and crowd photos are restricted to fully labelled ones.
* **Protocol.** Each detector runs once per clip through `VideoFileSource` (the real decode path), with score floor 0.2; detections are cached. All three trackers replay **identical** detections. The tracker detection threshold is tuned on dev (grid 0.3–0.7, by macro HOTA) and reported on test.
* **Metrics.** HOTA / DetA / AssA, IDF1, MOTA, ID switches, fragmentation, missed and false tracks, confirmation delay, termination delay, and Track-ID reuse on re-entry. They are implemented after TrackEval and cross-checked against py-motmetrics in `tests/test_metrics.py`.
* **Limitations.**
  * Compositing seams.
  * A head turn is approximated by horizontal foreshortening.
  * No real body motion or 3-D rotation.
  * Camera clips have no independent subject motion.
  * Treat the tracker *ranking* and the *failure modes* as the result. The absolute HOTA will move on real footage. §11 says how to replace these clips with recorded ones without code changes.

---

## 5. Detection results (measured)

All numbers are measured on the WIDER test split, with thresholds tuned on dev. The full tables, including per-bucket AP and recall at IoU 0.3, are in [results_tables.md](results_tables.md).

### 5.1 Accuracy at the same 640-px input

| Detector | AP50 | AP[.5:.95] | Precision | Recall | F1 | FP / image | Recall: tiny <16px / small 16–32 / medium 32–96 / large ≥96 |
|---|---|---|---|---|---|---|---|
| SCRFD-500M-KPS | **0.576** | **0.300** | 0.823 | **0.517** | **0.635** | 1.35 | 0.16 / 0.60 / 0.85 / 0.95 |
| YuNet (fp32) | 0.487 | 0.237 | 0.795 | 0.443 | 0.569 | 1.39 | 0.08 / 0.49 / 0.81 / 0.93 |
| YuNet (int8) | 0.486 | 0.234 | 0.777 | 0.448 | 0.569 | 1.56 | 0.08 / 0.50 / 0.81 / 0.93 |
| MediaPipe full-range | 0.145 | 0.048 | 0.550 | 0.145 | 0.230 | 1.44 | 0.00 / 0.00 / 0.35 / 0.77 |
| MediaPipe short-range | 0.035 | 0.013 | 0.357 | 0.032 | 0.059 | 0.70 | 0.00 / 0.00 / 0.00 / 0.42 |

Recall by **face height ÷ frame height**, which is what decides far-range behaviour. Values are at IoU 0.5; IoU 0.3 in brackets.

| Detector | < 5% | 5–10% | 10–20% | ≥ 20% |
|---|---|---|---|---|
| SCRFD @640 | 0.37 | 0.86 | 0.92 | 0.96 |
| YuNet @640 | 0.28 | 0.81 | 0.90 | 0.95 |
| YuNet @1024 (native for these 1024-px images) | 0.48 | 0.85 | 0.90 | 0.92 |
| MediaPipe full | 0.00 | 0.35 (0.45) | 0.69 (0.79) | 0.81 (0.91) |
| MediaPipe short | 0.00 | 0.00 | 0.10 (0.22) | 0.74 (0.86) |

**Reading this:**

* **The WIDER absolutes are low for every model.** 37% of WIDER faces are under 16 px, and no candidate finds those at a 640 input. On faces that fill at least 10% of the frame height, i.e. a person within a few metres of a 720p webcam, YuNet and SCRFD find **90–96%** of faces at IoU 0.5.
* **MediaPipe is a close-range detector.**
  * **Short-range** needs the face to be at least about 20% of the frame height: recall is 0.74 there and 0.10 at 10–20%. That means *selfie / video-call distance*.
  * **Full-range** works from about 10% of frame height (0.69) but is blind below 5%.
  * Part of MediaPipe's AP penalty is box convention, not missed faces: at IoU 0.3 its large-face recall rises to 0.87–0.91.
  * Because its input is a fixed 128/192 px square, giving it a higher-resolution camera does not help. A face must be a sufficient *fraction* of the frame.
* **SCRFD is the most accurate at every size.** Its margin over YuNet is +0.09 AP50 at 640, and it is largest on small faces (0.60 vs 0.49 recall at 16–32 px). It cannot be shipped commercially (§10).
* **int8 YuNet gives up no accuracy.** AP50 is 0.486 vs 0.487. Whether it is any *faster* is a separate question, answered in §5.4.

### 5.2 Input resolution is the far-range lever

| Input long side | YuNet AP50 | YuNet recall, 16–32 px faces | SCRFD AP50 | SCRFD recall, 16–32 px faces |
|---|---|---|---|---|
| 320 | 0.259 | 0.08 | 0.307 | 0.14 |
| 480 | 0.378 | 0.29 | 0.455 | 0.43 |
| 640 | 0.487 | 0.49 | 0.576 | 0.60 |
| 1024 (native) | **0.654** | **0.67** | **0.715** | **0.73** |

Going from 640 to native resolution raises YuNet's AP50 by 34% and its tiny-face recall fourfold (0.08 → 0.32). YuNet at native resolution beats SCRFD at 640. The same holds in tracking: YuNet@native tracks the 18–30 px far group that YuNet@640 misses entirely (HOTA 0.772 vs 0.744 overall). The cost is latency, which scales with pixel count (§5.4), so:

* **Close-range deployments** (faces ≥ 10% of frame height) can run at 320–480 with near-identical recall on those faces.
* **Far-range deployments** should run at native resolution. Do not cut resolution without checking the size buckets: the brief asks for this trade-off to be reported, not hidden.

### 5.3 Sample detections

_Figure: each detector on the same WIDER test images. Not committed (WIDER FACE images are CC BY-NC-ND); regenerate with `python scripts/make_visuals.py` into `docs/images/detections_grid.jpg`._

*Rows, top to bottom: YuNet@640, SCRFD@640, MediaPipe full, MediaPipe short. Columns: close group, crowd of tiny faces (51 labelled), medium faces against a busy crowd background. "found" counts all detections, including false positives. Green = ground truth, orange = detections at the dev-tuned threshold. Generated by `scripts/make_visuals.py`.*

### 5.4 Latency, CPU and memory (Python, CPU only)

Method: i7-12650H, fresh subprocess per config, 30 warm-up + 300 timed calls, 3 repetitions; median of the repetitions shown. The spread across repetitions was ≤ 0.2 ms for most configs (full table in results_tables.md).

| Config (640×480 frame unless noted) | Pre / inference / post p50 (ms) | End-to-end p50 / p95 (ms) | Cores busy | Model RAM | Load |
|---|---|---|---|---|---|
| **YuNet @640**, ORT default threads | 2.25 / 2.65 / 0.38 | **5.5 / 6.3** | 9.9 | +35 MB | 95 ms |
| **YuNet @640**, ORT **1 thread** | 1.81 / 7.63 / 0.31 | 9.8 / 11.4 | **1.0** | +32 MB | 99 ms |
| YuNet @320 | 0.43 / 1.40 / 0.32 | 2.2 / 3.5 | 9.6 | +28 MB | 119 ms |
| YuNet native, 720p frame | 7.71 / 8.83 / 0.53 | 18.0 / 19.8 | 9.9 | +57 MB | 107 ms |
| YuNet native, 1080p frame | 17.2 / 28.4 / 0.70 | 48.1 / 51.8 | 9.9 | +93 MB | 108 ms |
| YuNet **int8** @640 | 2.58 / **24.3** / 0.46 | 27.6 / 30.2 | 9.9 | +40 MB | 115 ms |
| YuNet via OpenCV `FaceDetectorYN` @640 | fused | 10.0 / 11.2 | 9.8 | +38 MB | 92 ms |
| SCRFD-500M @640 | 2.76 / 5.08 / 0.35 | 8.5 / 9.8 | 10.0 | +43 MB | 110 ms |
| SCRFD-500M @640, 1 thread | 2.05 / 16.2 / 0.25 | 18.7 / 21.1 | 1.0 | +40 MB | 116 ms |
| MediaPipe short | fused | **1.9 / 2.2** | 1.3 | +53 MB | 777 ms |
| MediaPipe full | fused | 4.9 / 5.7 | 1.2 | +60 MB | 717 ms |
| *Sustained: YuNet @640, 120 s* | | 6.0 → 6.4 ms mean per 10 s window | | | |

**Findings:**

* **Every candidate is real-time on this CPU.** The slowest default config at 640 (SCRFD) still runs at 116 FPS.
* **ONNX Runtime's default threading spins about 10 cores** to save 5 ms. On phones and laptops on battery, set `num_threads` to 1–2: YuNet still runs at about 100 FPS on one core. MediaPipe's XNNPACK uses about 1.2 cores by default, which explains its low CPU figure.
* **int8 is 9× slower at inference** (24 vs 2.7 ms) on x86 ONNX Runtime for this tiny model: quantize/dequantize overhead dominates a model with 53k parameters. Measured accuracy is equal. **Do not ship int8 without measuring it on the target ARM device.** (Published zoo numbers show int8 roughly equal to fp32 on ARM boards.)
* **The OpenCV 5 `FaceDetectorYN` path is about 2× slower** than ONNX Runtime with our decoder (10.0 vs 5.5 ms), with identical outputs.
* **Native resolution scales with pixels**: 18 ms at 720p, 48 ms at 1080p. The far-range recall from §5.2 costs 3–9× the 640 latency. With `detect_every=2` (§6.3), 720p native fits in about 9 ms per frame on average.
* **Preprocessing (resize + blob) is up to 40% of YuNet's end-to-end time.** Doing the resize on the GPU or in the camera pipeline is the next optimisation.
* **No throttling** over the 120 s sustained run on mains power. Battery and thermal behaviour on phones is **not measured** (§11).
* **Memory.** Model RAM is +28–93 MB. "RSS peak" in the full table includes the 60 benchmark frames held in memory, so read the model delta instead.
* **Startup.** About 0.1 s for ONNX Runtime and about 0.7 s for MediaPipe, plus about 7 ms for the first inference.

---

## 6. Tracking results (measured)

Full tables (all 5 detectors × 5 tracker variants, by generator, per scenario, ablations) are in [results_tables.md](results_tables.md#tracking--synthetic-motion-sequences-over-real-wider-faces-test-split). Test split, macro-averaged over 24 clips. The detector threshold was tuned on dev.

### 6.1 Trackers on identical detections

| Detector | Tracker | HOTA | IDF1 | MOTA | ID sw. | Frag | False tracks | Tracker ms (mean) |
|---|---|---|---|---|---|---|---|---|
| YuNet @640 | IoU | **0.744** | **0.870** | 0.846 | 6 | 85 | 48 | 0.14 |
| YuNet @640 | SORT | 0.737 | 0.854 | 0.845 | 6 | 69 | 39 | 0.26 |
| YuNet @640 | **ByteTrack** | 0.729 | 0.859 | **0.862** | **4** | **20** | **9** | 0.29 |
| YuNet @native | IoU | **0.772** | **0.902** | 0.878 | 6 | 94 | 47 | 0.16 |
| YuNet @native | **ByteTrack** | 0.757 | 0.892 | **0.889** | **4** | **21** | **8** | 0.38 |
| SCRFD @640 | IoU | 0.786 | 0.893 | 0.866 | 2 | 127 | 8 | 0.15 |
| SCRFD @640 | ByteTrack | 0.750 | 0.844 | 0.839 | 4 | **12** | 0 | 0.30 |
| MediaPipe full | SORT | 0.535 | 0.712 | 0.626 | 66 | 439 | 31 | 0.22 |
| MediaPipe short | SORT | 0.407 | 0.562 | 0.511 | 16 | 67 | 19 | 0.16 |

**What the numbers say:**

1. **The detector dominates.** The spread between detectors (HOTA 0.41 → 0.79) is about 10× the spread between trackers on the same detector (≤ 0.04). MediaPipe's weak small-face recall produces box flicker, which becomes ID switches (66–197) and fragmentation in the crowd and pan clips.
2. **No tracker wins every metric.**
   * The plain IoU tracker has the best HOTA/IDF1 on these clips. The motion in them is slow and smooth, and its dev-tuned threshold is lower (0.4), which buys recall.
   * ByteTrack has **4–7× less fragmentation**, the fewest false tracks (9 vs 48), the fewest ID switches and the best MOTA. On the camera-motion clips it has Frag 6 vs 65 (IoU) and 51 (SORT): its low-score second stage keeps tracks alive through blur and dips that break the other two.
   * For this product, every fragment and false track becomes a separate "visit" that Phase 3 must embed and resolve. So **continuity is worth more than the 0.015 HOTA**: ByteTrack is the recommended tracker.
3. **All trackers are cheap.** Mean 0.14–0.38 ms per frame and ≤ 2.2 ms p95 in Python (worst clip); the TypeScript port measured 0.27–0.38 ms in Chrome. Tracker cost is irrelevant next to the detector.
4. **Lifecycle behaves as specified in every run.**
   * **0 re-entries reused an old Track ID**, across all detectors and trackers. A returning person always got a new temporary ID.
   * A single missed detection never ended a track (unit-tested).
   * Termination delay is about `max_lost_ms` (907–926 ms at 1 s). The clock starts at the last ground-truth frame, which is slightly after the detector last saw the face as it left the frame.

### 6.2 Per situation (YuNet @640; IDF1 / ID switches)

| Situation from the brief | IoU | SORT | ByteTrack | Notes |
|---|---|---|---|---|
| 1. One person continuously visible | 0.954 / 0 | 0.954 / 0 | **0.994 / 0** | |
| 2. Three people entering together (+3 lighting variants) | 0.997–0.998 / 0 | same | 0.988–0.992 / 0 | |
| 3. One leaves, another stays | 0.994 / 0 | 0.992 / 0 | 0.989 / 0 | |
| 4. Temporary occlusion (an occluder slides past) | **0.985 / 0** | 0.788 / 1 | 0.788 / 1 | Kalman failure: see below |
| 5. Two people crossing (+3 lighting variants) | 0.938–0.994 / 0 | same | 0.964–0.994 / 0 | no ID swaps |
| 6. Head turned away ~1.1 s | 0.782 / 1 | 0.782 / 1 | 0.782 / 1 | Face undetectable for longer than the 1 s grace: by design (see 6.3) |
| 7. Detector misses (motion blur) during a crossing | 0.758 / 1 | 0.758 / 1 | 0.768 / 1 | The occluded face moves about 50 px while hidden |
| 8. Leaving permanently | 0.966 / 0 | 0.966 / 0 | **0.988 / 0** | ended after about 1 s |
| 9. Re-entry after 2.8 s | 0.994 / 0 | 0.993 / 0 | 0.993 / 0 | **new Track ID every time** |
| Far group, 1280×720 (faces 18–30 px), input 640 | 0.03 | 0.02 | 0.00 | detector-limited: input must be native, §5.3 |

**Failure analysis (traced frame by frame).**

* **Situation 4.** The occluder covers a *stationary* face from one side. The detected box shrinks sideways, the constant-velocity Kalman filter reads that as motion, and during the 0.6 s gap the prediction drifts off the face. On return there is no overlap, so the face gets a new ID inside the grace period. The IoU tracker has no motion model and keeps the ID.
* **Attempted fix: observation-centric recovery** (`lost_match_last_observation`, after OC-SORT). A lost track is also matched on its last *observed* box. It fixes the synthetic unit-test version of this failure (`test_lastobs_recovers_stationary_face_after_partial_occlusion`) but **not the measured clip**: there, the last observed box is the half-covered, shrunken one, which still does not overlap the full face. Across the benchmark it was mixed. It helped SCRFD + ByteTrack (HOTA 0.750 → 0.763) and hurt YuNet + ByteTrack (0.729 → 0.719). It *added* ID switches to SORT (6 → 9 with YuNet, 3 → 12 with SCRFD), because a stale box can capture a different person who steps into the spot. **Kept as an off-by-default option**; not adopted.

### 6.3 Grace period, frame skipping, frame rate (YuNet @640)

| Experiment | ByteTrack HOTA / IDF1 / ID sw. | IoU HOTA / IDF1 / ID sw. | Finding |
|---|---|---|---|
| `max_lost_ms` 500 | 0.700 / 0.804 / 10 | 0.707 / 0.799 / 13 | Too short: ordinary dropouts end tracks |
| `max_lost_ms` 1000 (default) | 0.729 / 0.859 / 4 | 0.744 / 0.870 / 6 | |
| `max_lost_ms` 1500 | 0.729 / 0.859 / 4 | **0.750 / 0.880 / 4** | Best for IoU / SORT; the end delay grows with it (1.26 s) |
| `max_lost_ms` 2000–3000 | flat | flat | No benefit, slower `track_ended`; still 0 re-entry reuses at a 2.8 s gap, but the risk grows |
| Detector every 2nd frame | 0.714 / 0.853 / 4 | 0.731 / 0.874 / 5 | **−2% HOTA for half the detector cost** |
| Detector every 3rd frame | 0.706 / 0.856 / 3 | 0.708 / 0.862 / 6 | −3 to −5% HOTA for a third of the cost |
| 15 fps input (every 2nd frame) | 0.718 / 0.854 / 4 | 0.742 / 0.876 / 5 | |
| 10 fps input | 0.713 / 0.858 / 3 | 0.734 / 0.870 / 6 | Time-based expiry holds: end delay 0.95 s vs 0.93 s at 30 fps |
| 30 fps with 40% of frames randomly dropped (jitter) | 0.722 / 0.855 / 4 | 0.740 / 0.870 / 8 | Irregular frame intervals are handled |

**Settings recommended from this:**
* `max_lost_ms` **1500** for seated, conversational scenes (covers sub-second head turns and occlusions); keep 1000 where `track_ended` latency matters more.
* `detect_every` **2** on weak devices: the tracker coasts on its predictions in between, at about 2% HOTA cost.
* ID continuity survives 10 fps and jittery delivery, which is what phones and busy browsers produce.

---

## 7. Architecture and implementation

```text
FrameSource (VideoFileSource | CameraSource)      -> Frame{image BGR, frame_id, timestamp_ms, source_id}
  └─ FacePipeline.process(frame)                     (source-agnostic; detect_every N; invalid frames coast)
       ├─ FaceDetector.detect(image)                -> [Detection{bbox xywh, confidence, keypoints}], StageTimings
       │     yunet | yunet_int8 | yunet_cv | scrfd_500m_kps | mediapipe_short | mediapipe_full
       ├─ BaseTracker.update(dets, timestamp_ms)    -> TrackerOutput{tracks (snapshots), events}
       │     shared lifecycle  +  IoUTracker | SortTracker | ByteTracker (association only)
       └─ observers: TrackObserver.on_observation / on_event   (Phase 3 hook)
  └─ viz.draw / JsonlExporter / MotExporter
```

* **Contracts.** `src/facetrack/types.py`: boxes are `[x, y, w, h]` in frame pixels; timestamps are milliseconds from the container's presentation time (video) or a monotonic clock (camera), never a frame counter. The detector contract is a single method, so swapping a model never touches the tracker, and the reverse holds too.
* **Lifecycle.** `tracking/base.py`.
  * States: `tentative → confirmed → lost → terminated`.
  * Events: `track_started`, `track_updated`, `track_lost`, `track_recovered`, `track_ended`.
  * Expiry is time-based (`max_lost_ms`, `tentative_timeout_ms`), so a phone at 12 fps and a desktop at 30 fps end tracks after the same wall time. Tests cover 10, 30 and 60 fps and jittered intervals.
  * A single missed detection moves a track to *lost*, never to *terminated*.
  * IDs come from a counter that never repeats, even across `reset()`.
* **Motion.** SORT's (centre, area, aspect) filter and ByteTrack's (centre, aspect, height) filter, both with `dt` measured in reference frames (elapsed ms / 33.3), so the published tunings apply at 30 fps and scale with the real frame interval.
* **Configuration.** All parameters live in `configs/default.yaml` / `TrackerConfig`: detection threshold, low threshold, new-track threshold, `min_hits`, `max_lost_ms`, `tentative_timeout_ms`, IoU threshold, low-IoU threshold and centre-distance threshold.
* **Robustness.**
  * Grayscale and BGRA frames are converted; empty frames coast.
  * Backwards timestamps raise an error (call `reset()` on seek).
  * Missing models, undecodable videos and unavailable cameras fail with clear messages.
  * The camera source reconnects with backoff after a stream interruption.
* **Demo.** `facetrack-demo` (`src/facetrack/cli/demo.py`) takes a video file (default) or a camera and runs the identical pipeline. It shows boxes, IDs, confidence, status, FPS and detector latency; offers play, pause, step, restart and stop; and exports JSONL, MOT text and annotated video.
* **Browser.** `web/` reimplements the same contracts in TypeScript. The trackers reproduce the Python reference frame by frame: 30 parity cases across 5 tracker variants, with IDs, boxes and events identical, in `web/tests/`. Detection runs in a Web Worker with at most one frame in flight: when the detector is busy, frames are dropped rather than queued, so output never lags behind the camera. Uploaded video is the default mode; the live camera feeds the same callback.
* **Tests.** 106 Python tests and 31 TypeScript tests. The headless-Chrome end-to-end test (`npm run test:e2e`) runs 9 checks:
  * upload an H.264 video and see tracks;
  * re-entry gets a new ID;
  * export is enabled;
  * an undecodable codec gives a clear error;
  * a fake camera fed real faces produces tracks;
  * stopping the camera releases the device;
  * no page errors.

---

## 8. Deployment

### 8.1 Browser (implemented: `web/`)

* **Threading.** Inference runs in a **Web Worker**, never on the UI thread. MediaPipe's own docs warn that its `detect*` calls block the calling thread. At most one frame is in flight; frames that arrive while the worker is busy are dropped and counted ("Dropped" in the UI). This keeps the output aligned with the camera instead of building a queue.
* **Accelerator selection** (`web/src/detector/worker.ts`):
  1. With *Auto*, if `navigator.gpu` exposes an adapter, the worker creates **both** a WebGPU and a WASM session, times 5 inferences on each, and keeps the faster. MediaPipe gets the same treatment with its GPU and CPU delegates.
  2. Without an adapter, or if WebGPU session creation fails, the worker uses WASM.
  3. The reason for the choice is shown in the UI tooltip.

  WebGPU is never assumed. The published support matrix (research §7) still lacks it on Firefox Android and older iOS, and §8.3 shows it is not automatically faster for models this small.
* **WASM threads.** These need cross-origin isolation. The dev and preview servers send COOP/COEP headers, and production hosting must send them too. Without isolation ORT falls back to 1 thread, which the UI reports.
* **Measured download cost** (brotli-11, the transfer size when served compressed):

| Asset | Raw | Brotli |
|---|---|---|
| YuNet model (`.onnx`) | 0.23 MB | 0.20 MB |
| SCRFD-500M model | 2.52 MB | 2.31 MB |
| BlazeFace short / full model (`.tflite`) | 0.23 / 1.08 MB | 0.19 / 0.91 MB |
| ONNX Runtime Web wasm, CPU-only build | 14.2 MB | **2.36 MB** |
| ONNX Runtime Web wasm, WebGPU-capable (JSEP) build | 28.3 MB | **3.95 MB** |
| MediaPipe Tasks Vision wasm | 13.5 MB | **2.75 MB** |
| App + worker JS (both runtimes' JS) | 0.59 MB | (not measured) |

  The runtime, not the model, dominates first load. Only the selected backend's wasm is fetched.
* **Camera.** Device selection, permission denial, "in use", "not found" and a mid-stream `ended` track each produce a specific message. *Stop* stops every `MediaStreamTrack`, and the e2e test asserts the stream is released. The page must be served over `https://` (or `localhost`).
* **Uploaded video.** Timestamps come from `requestVideoFrameCallback().mediaTime` (presentation time), with a `requestAnimationFrame` fallback. Browsers do not expose a container FPS, so the UI shows the FPS *measured* from frame timestamps. Codecs: H.264, VP9 and AV1 in MP4 or WebM play in current browsers. MPEG-4 Part 2 (OpenCV's `mp4v`) does not decode in Chrome; the app detects that and says so (tested), and `scripts/make_web_samples.py` transcodes such files.

### 8.2 Mobile apps (not built; evidence-based plan)

| Path | Detector | Effort | Notes |
|---|---|---|---|
| **MediaPipe Tasks (Android/iOS)** | BlazeFace short/full | Lowest: first-party SDKs, camera helpers, GPU delegate | Same model and decoder as the browser. iOS ≥ 15. Far-range limits as measured in §5. |
| **ONNX Runtime Mobile** (`onnxruntime-android`, `onnxruntime-objc`) | YuNet (or SCRFD) | Medium: we supply pre/post-processing (port `yunet.py`, about 150 lines) | ORT recommends the XNNPACK EP for float models. **NNAPI is deprecated as of Android 15** (Android docs), so do not build on it. CoreML EP is available on iOS 13+ but still marked "preview". The prebuilt AAR is about 24 MB; a custom op-reduced build is much smaller (published). |
| **OpenCV `FaceDetectorYN`** (OpenCV Android/iOS SDK) | YuNet | Low–medium: one call does pre/post-processing | The SDK is large. Use it if the app already ships OpenCV. |

Same runtime everywhere is not a goal: the detector contract is identical, so each platform can use its best runtime.

### 8.3 Measured browser latency (Chrome 156, headless, same laptop; Web Worker; 640×480 frames)

| Backend | Accelerator | End-to-end p50 / p95 (ms) | Inference p50 (ms) |
|---|---|---|---|
| YuNet @640 | WASM, 4 threads | **19.7 / 29.7** | 13.4 |
| YuNet @640 | WebGPU (Intel Iris Xe iGPU) | 54.3 / 67.6 | 51.0 |
| YuNet @320 | WASM | **9.5 / 14.4** | 5.2 |
| SCRFD @640 | WASM / WebGPU | 43.2 / 52.1 | 35.8 / 46.5 |
| MediaPipe short | CPU / GPU (WebGL) | **7.0** / 8.2 | 6.3 / 7.5 |
| MediaPipe full | CPU / GPU (WebGL) | 15.2 / **13.0** | 14.5 / 12.4 |

**Findings:**

* **WebGPU was slower than WASM** on this device for both ONNX models: 2.7× slower for YuNet. Kernel dispatch and readback overhead dominate a 0.5 GFLOP model, which is exactly why *Auto* measures before choosing instead of preferring WebGPU. On a discrete or Apple GPU the result may differ, so measure there with `web/bench.html`.
* **The browser costs about 3–4×** native Python ONNX Runtime (19.7 vs 5.5 ms for YuNet). About 6 ms of that is canvas `getImageData` plus the JavaScript NCHW conversion, which a WebGL/WebGPU preprocessing shader would remove.
* **Everything except SCRFD fits a 30 fps budget** on this laptop, inside a worker, with the UI thread free. In the end-to-end app run, with YuNet @640 under load, the UI showed 23–26 processed FPS from a 30 fps video. The difference was dropped, not queued.
* **Mobile browsers were not measured.** Open `/bench.html` on the target phones; it runs the same configs and prints the same table.

---

## 9. Phase 3 integration (designed, not implemented)

The tracker already emits everything identity matching needs. The interfaces are in `src/facetrack/identity.py`:

1. **Collect evidence while the face is visible.** A `TrackObserver.on_observation(frame, track, detection)` call arrives for every confirmed, matched track, with the frame pixels. An `EvidenceCollector` scores each crop with a `FaceQualityScorer` (size, sharpness, frontal pose from the keypoints, detector confidence) and keeps the best *k* per Track ID, embedding them with an `EmbeddingExtractor` (an ArcFace-family model, aligned on the 5 keypoints).
2. **Finalise on `track_ended`.** The collector passes the track's `FaceEvidence` list to an `IdentityMatcher`. It compares the evidence with an `IdentityStore` of enrolled people and returns an `IdentityDecision` (person ID or `UNKNOWN`, a score and the evidence count).
3. **Many Track IDs → one person.** A person who leaves and returns gets a new Track ID from the tracker (verified: 0 Track-ID reuses on re-entry across all runs). The matcher maps both Track IDs to the same person ID.
4. **Never recognise from a frame after the track ended.** By then the face is gone; evidence is captured during the track.
5. **Keypoint caveat.** BlazeFace has no mouth corners. A MediaPipe-detector pipeline needs a different alignment template or a landmark model before ArcFace-style alignment. YuNet and SCRFD output the 5-point ArcFace order directly.

---

## 10. Licensing

Sources and exact quotes are in [research/published_reference.md](research/published_reference.md) §6, §8 and §9. **This is not legal advice**; items 1–3 need legal review before a commercial launch.

| Item | Code | Weights / data | Risk for commercial use |
|---|---|---|---|
| **SCRFD (`det_500m.onnx`)** | MIT (README); training code Apache-2.0 | **Non-commercial research only**, stated 4× including a signed `MODEL.LICENSE` | **Blocked** without a licence from InsightFace. Kept here only as an accuracy reference. |
| **YuNet 2023mar** | MIT (zoo) / BSD-3 (training repo) | MIT; but trained on **WIDER FACE (CC BY-NC-ND)** with "labelv2" annotations that "come from SCRFD", and InsightFace states its annotations *and models trained with them* are non-commercial | **Unresolved by the sources**: permissive weights licence, restrictive training-data chain. Needs legal review. |
| **MediaPipe BlazeFace** | Apache-2.0 | Apache-2.0 model cards; trained on Google's consented in-house images, **not WIDER** | **Lowest licence risk.** The model card states that "surveillance or identity recognition is explicitly out of scope". That is an intended-use statement, not a licence term, but it must be weighed for Phase 3. |
| WIDER FACE (our evaluation set) | — | CC BY-NC-ND | Internal research evaluation only; do not ship or redistribute. |
| Trackers in this repo | Our own implementations from the papers | — | The reference SORT code is **GPL-3.0** and was *not* used or copied; ByteTrack's reference code is MIT. |
| ONNX Runtime / OpenCV / MediaPipe runtimes; pycocotools, SciPy, py-motmetrics | MIT / Apache-2.0 / Apache-2.0; BSD-2, BSD-3, MIT | — | None. |

**If commercial use is intended:**

1. Do not ship SCRFD weights.
2. Get a legal opinion on YuNet's WIDER/InsightFace training chain.
3. If that opinion is negative, the fallback is MediaPipe BlazeFace (Apache-2.0, consented data). Accept its measured far-range weakness (§5) or mitigate it, e.g. by running the full-range model on tiles or crops of a high-resolution frame (an *estimate*, not measured here).
4. The longer-term clean option is retraining YuNet's open training code (BSD-3) on commercially licensed or consented data with our own annotations.

---

## 11. What still needs real devices or real footage

These were not measured, and nothing in this report should be read as measuring them:

1. **Phones and mobile browsers.** Run `web/bench.html` (latency per backend and accelerator) and the app itself on 2–3 target phones, including one low-end Android. Record device, OS, browser version and power state. Measure **battery drain and thermal throttling** with a 10-minute sustained camera session (OS battery stats before and after, plus FPS over time from the exported JSONL).
2. **Consented, annotated video of our real setting.** Follow [DATA_PROTOCOL.md](DATA_PROTOCOL.md): about 30 clips across scenarios A–E, at least two devices, split by session, pre-annotated by the pipeline and corrected in CVAT, imported with `scripts/import_mot_annotations.py`. The tracking benchmark then reports recorded clips separately (`generator = recorded`). Re-check the operating points (`det_threshold`, `max_lost_ms`) there, and decide whether the far-range use case needs native resolution.
3. **Mobile app runtimes** (§8.2): ORT Mobile with the XNNPACK EP vs MediaPipe Tasks, on device. Re-test int8 YuNet on ARM before rejecting it everywhere.
4. **Legal review** of the YuNet training-data chain (§10) before any commercial launch.

**Engineering next steps, in order of measured payoff:**
* GPU or shader preprocessing in the browser (about 6 ms of the 20 ms).
* `num_threads=1–2` defaults on mobile.
* A head-turn-aware grace period (1.5 s).
* A light motion model for stationary faces (velocity damping when lost) to fix the occluder failure in §6.2. The last-observation variant tried here did not fix it.
