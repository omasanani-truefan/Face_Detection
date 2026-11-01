# Published reference: lightweight face detectors, trackers, runtimes

Retrieved **2026-11-01**. This section collects only what authors and vendors publish. No local benchmarks were run; local measurements belong in a separate report.

**Labels used throughout**

- **PUBLISHED**: the number appears at the cited URL (paper, official README, official docs or model card).
- **DERIVED**: read directly from an official artifact, such as an HTTP `Content-Length`, a Git LFS pointer, an ONNX/TFLite tensor header, a ZIP central directory, an official annotation file or a sha256. No inference was run. These are not author claims. They are included only where they settle an ambiguity in the published material.
- **not verified**: I could not confirm the value from a source I fetched.

GPU, CPU and mobile hardware are marked explicitly on every speed number. Latencies from different sources are **not comparable**: hardware, runtime, input size and whether pre/post-processing is included all differ.

---

## 0. Summary matrix

All values are PUBLISHED unless tagged DERIVED. Each cell carries its source link.

| Model (file) | Params | FLOPs @ input | File size | WIDER FACE val E / M / H (protocol) | Published speed (hardware, runtime) | Landmarks | Code license | Weights license | Browser path | Mobile path |
|---|---|---|---|---|---|---|---|---|---|---|
| **YuNet, opencv_zoo `face_detection_yunet_2023mar.onnx`** (byte-identical to libfacedetection.train `yunet_s_640_640.onnx`, DERIVED via sha256 `8f2383e4…`) | YuNet-s: 54,608 ([paper Tab.1](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | YuNet-s: 96 M @320×320; 386 M @640×640 ([paper Tab.1](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | 232,589 B fp32; 100,416 B int8; 122,489 B int8bq (DERIVED, [LFS pointers](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)) | 0.8844 / 0.8656 / 0.7503 (opencv_zoo `tools/eval`, original image size, conf 0.3, NMS 0.45) ([zoo README](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)); paper YuNet-s @original size: 0.887 / 0.871 / 0.768 ([paper](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | **CPU** 0.69 ms mean @160×120, i7-12700K, OpenCV 4.10 DNN ([zoo benchmark](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)); **CPU** YuNet-s 1.4 ms @320×320, i7-12700K, ONNX Runtime ([paper Tab.4](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | 5 | MIT (zoo dir) ([LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE)); BSD-3 (training repo) | MIT ("All files in this directory") ([zoo README](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md)) | OpenCV.js `FaceDetectorYN` binding is whitelisted ([opencv_js.config.py](https://github.com/opencv/opencv/blob/4.x/platforms/js/opencv_js.config.py)); generic ONNX → ORT Web (no official demo verified) | OpenCV DNN; libfacedetection C++ (NEON) ([README](https://github.com/ShiqiYu/libfacedetection)); TFLite *export script* only ([train README](https://github.com/ShiqiYu/libfacedetection.train)) |
| **YuNet-n (paper headline model)**, libfacedetection.train `yunet_n_*.onnx` | 75,856 ([paper](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | 149 M @320×320; 595 M @640×640 ([paper Tab.1](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | 317,466 B (`yunet_n_640_640.onnx`, DERIVED via [GitHub API](https://github.com/ShiqiYu/libfacedetection.train/tree/master/onnx)) | 0.892 / 0.883 / 0.811 (single-scale, original size, conf 0.01, NMS 0.45) ([paper](https://link.springer.com/article/10.1007/s11633-023-1423-y), [train README](https://github.com/ShiqiYu/libfacedetection.train)) | **CPU** 1.6 ms @320×320, i7-12700K, ONNX Runtime ([paper abstract & Tab.4](https://link.springer.com/article/10.1007/s11633-023-1423-y)) | 5 | BSD-3-Clause ([LICENSE](https://github.com/ShiqiYu/libfacedetection.train/blob/master/LICENSE)) | BSD-3 repo; no separate weights license stated | same as above | same as above |
| **SCRFD_500M_KPS** (`det_500m.onnx` in `buffalo_s`, `buffalo_sc`, `cheetah_s`) | 0.57 M ([SCRFD README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)) | 500 M @VGA 640×480 ([SCRFD README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)) | `det_500m.onnx` 2,524,817 B (DERIVED from [release ZIP](https://github.com/deepinsight/insightface/releases/tag/model-zoo) directory) | 90.97 / 88.44 / 69.49 (single-scale, bounded by VGA) ([SCRFD README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)) | **GPU** 3.6 ms @VGA, NVIDIA 2080Ti ([README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) + [paper Tab.4](https://arxiv.org/abs/2105.04714)); **CPU** (SCRFD-0.5GF, non-KPS) 28.3 ms @640×480, 11.4 ms @320×240, Ryzen 9 3950X single-thread PyTorch ([README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)) | 5 | MIT per README; `detection/scrfd/LICENSE` is Apache-2.0 | **Non-commercial research only** ([README](https://github.com/deepinsight/insightface#license)) | generic ONNX → ORT Web (no official browser demo) | generic ONNX → ORT Mobile; third-party ncnn Android demo listed in README |
| SCRFD_2.5G_KPS (reference) | 0.82 M | 2.5 G @VGA | `det_2.5g.onnx` in `buffalo_m`: 3,292,009 B (DERIVED; KPS variant not verified) | 93.80 / 92.02 / 77.13 (same protocol) | **GPU** 4.3 ms @VGA, 2080Ti | 5 | as above | Non-commercial | as above | as above |
| **MediaPipe BlazeFace short-range** (`blaze_face_short_range.tflite`) | not verified (not published) | not verified (not published) | 229,746 B (DERIVED, HTTP); "224KB" ([model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)) | **Not published on WIDER FACE.** Paper reports AP 98.61% on a private frontal-camera dataset ([arXiv 1907.05047](https://arxiv.org/abs/1907.05047)) | **Mobile** Pixel 6 whole-task pipeline: CPU 2.94 ms, GPU 7.41 ms ([docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector)); **mobile CPU** ~275 FPS Pixel 2 single-core XNNPACK ([model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)); **mobile GPU** 0.6 ms iPhone XS, TFLite GPU fp16 ([paper Tab.1](https://arxiv.org/abs/1907.05047)) | 6 | Apache-2.0 ([repo](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)) | Apache-2.0 ([model card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)) | Official: `@mediapipe/tasks-vision`, CPU or GPU delegate ([setup](https://ai.google.dev/edge/mediapipe/solutions/setup_web)) | Official: MediaPipe Tasks Android and iOS |
| MediaPipe BlazeFace full-range / sparse full-range | not verified | not verified | 1,083,786 B / 676,746 B (DERIVED, HTTP); "1.1MB" / "680KB" (model cards) | Not published on WIDER FACE | **Mobile GPU** ~90 FPS Pixel 3, ~165 FPS Pixel 4; **mobile CPU** ~50 FPS Pixel 3 single-core XNNPACK, ~145 FPS (full) / ~160 FPS (sparse) Pixel 4 XNNPACK ([full card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Full%20Range).pdf), [sparse card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Sparse%20Model%20Card%20(Full%20Range).pdf)) | 6 | Apache-2.0 | Apache-2.0 (model cards) | Tasks API (support added in v0.10.33, see §3) | Tasks API |
| **RetinaFace-MobileNet0.25** (biubug6 `mobilenet0.25_Final.pth`) | 426,608 ([YuNet paper Tab.1](https://link.springer.com/article/10.1007/s11633-023-1423-y)); 0.44 M ([SCRFD README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)) | 245 M @320×320 (YuNet paper); 0.802 G @VGA (SCRFD README) | "1.7M" ([Pytorch_Retinaface README](https://github.com/biubug6/Pytorch_Retinaface)); "1MB" ([paper](https://arxiv.org/abs/1905.00641)) | 90.70 / 88.16 / 73.82 (original scale, single scale); 88.67 / 87.09 / 80.99 ("same parameter with Mxnet") ([README](https://github.com/biubug6/Pytorch_Retinaface)); 87.78 / 81.16 / 47.32 @VGA ([SCRFD README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)) | **GPU** 1.4 ms @VGA, Tesla P40; **CPU** 5.5 ms multi-thread / 17.2 ms single-thread @VGA, i7-6700K; **ARM** 61.2 ms @VGA, RK3399; all via TVM ([paper Tab.5](https://arxiv.org/abs/1905.00641)); **GPU** 7.9 ms @VGA, 2080Ti (SCRFD README) | 5 | MIT ([LICENSE.MIT](https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT)) | Not stated (weights on Google Drive/Baidu) | ONNX export script only (`convert_to_onnx.py`) | not verified |
| **SORT** (tracker) | n/a | n/a | n/a | n/a. MOT15 test MOTA 33.4 (**pedestrians**) ([arXiv 1602.00763](https://arxiv.org/abs/1602.00763)) | **CPU** 260 Hz tracking component, single core of Intel i7 2.5 GHz (paper) | n/a | **GPL-3.0** ([LICENSE](https://github.com/abewley/sort/blob/master/LICENSE)) | n/a | n/a | n/a |
| **ByteTrack** (tracker) | n/a | n/a | n/a | n/a. MOT17 test MOTA 80.3 / IDF1 77.3 / HOTA 63.1 (**pedestrians**) ([README](https://github.com/ifzhang/ByteTrack)) | **GPU** 29.6 FPS on MOT17, *including YOLOX-X detector*, V100, FP16 ([README](https://github.com/ifzhang/ByteTrack), [paper](https://arxiv.org/abs/2110.06864)) | n/a | MIT ([LICENSE](https://github.com/ifzhang/ByteTrack/blob/main/LICENSE)) | n/a | n/a | n/a |

### Key surprises (details in the sections below)

1. **The opencv_zoo "YuNet" 2023mar ONNX is the smaller YuNet-s variant, not the paper's 75,856-parameter YuNet.** DERIVED: `face_detection_yunet_2023mar.onnx` is byte-identical (sha256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`) to [`libfacedetection.train/onnx/yunet_s_640_640.onnx`](https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_s_640_640.onnx). This is consistent with the AP published in the [libfacedetection README](https://github.com/ShiqiYu/libfacedetection) (0.887/0.871/0.768), which equals the paper's YuNet-s original-size row.
2. **The headline AP in the opencv_zoo YuNet README is stale.** The first line says 0.834/0.824/0.708. Those numbers are the pre-2023 model's, matching the [libfacedetection.train README at commit a61a428](https://github.com/ShiqiYu/libfacedetection.train/blob/a61a428929148171b488f024b5d6774f93cdbc13/README.md) that the "Model source" link points to. The same README's table for the 2023mar file reads 0.8844/0.8656/0.7503 today. In June 2023 it read [0.8871/0.8710/0.7681](https://github.com/opencv/opencv_zoo/blob/f12e12798e/models/face_detection_yunet/README.md). The change is not explained.
3. **The MediaPipe docs table lists full-range and sparse as 128×128 input.** The official `.tflite` files take **1×192×192×3** (DERIVED from the TFLite header). The model cards say **160x192**. Full-range support only reached the Tasks API in March 2026.
4. **InsightFace pretrained models, SCRFD `det_500m.onnx` included, are non-commercial research only.** The new 2026 packs also carry a signed `MODEL.LICENSE` with `"grant": "non-commercial"`.
5. **SORT is GPL-3.0.**
6. **WIDER FACE is CC BY-NC-ND.** YuNet, SCRFD and RetinaFace are all trained on it. YuNet's current training annotations ("labelv2") come from SCRFD/InsightFace, whose README states that "the training data containing the annotation (and the models trained with these data)" are non-commercial.

---

## 1. YuNet

### Sources fetched
- opencv_zoo model dir and README: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet (README commit `26cc381e4d`, 2026-05-22)
- opencv_zoo YuNet LICENSE: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE
- opencv_zoo benchmark: https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md, config https://github.com/opencv/opencv_zoo/blob/main/benchmark/config/face_detection_yunet.yaml, timer https://github.com/opencv/opencv_zoo/blob/main/benchmark/utils/timer.py
- opencv_zoo eval: https://github.com/opencv/opencv_zoo/blob/main/tools/eval/eval.py, https://github.com/opencv/opencv_zoo/blob/main/tools/eval/datasets/widerface.py
- Paper: Wu, Peng, Yu, "YuNet: A Tiny Millisecond-level Face Detector", *Machine Intelligence Research* 20(5):656–665, 2023, https://link.springer.com/article/10.1007/s11633-023-1423-y (open access, CC BY 4.0 per [Crossref metadata](https://api.crossref.org/works/10.1007/s11633-023-1423-y))
- Training repo: https://github.com/ShiqiYu/libfacedetection.train (README and LICENSE)
- C++ library: https://github.com/ShiqiYu/libfacedetection
- Hugging Face mirror: https://huggingface.co/opencv/face_detection_yunet

### Architecture (PUBLISHED, paper §3.2 and Fig. 4)
- **Backbone**, 5 stages built from depthwise-separable units. A *DWUnit* is a 1×1 conv followed by a 3×3 depthwise conv with BN+ReLU; a *DWBlock* is two DWUnits.
  - Stage 0 is "ConvHead": a 3×3 conv with stride 2, then a DWUnit, 16 channels.
  - Stage 1 is a max-pool followed by DWBlocks that raise the channels to 64, ending at 1/4 resolution.
  - Stages 2–4 each apply a DWBlock (64 ch) and a max-pool. Their outputs are at 1/8, 1/16 and 1/32.
- **Neck**: "TFPN" (tiny FPN). This is FPN with each 3×3 conv replaced by a DWUnit, using a top-down path with upsampling. The paper says it reduces parameters "to 12% from that of FPN".
- **Head**: a decoupled head on each of the 3 levels, with DWUnit branches for cls (1), obj (1), bbox (4) and kps (10, i.e. **5 landmarks**).
- **Anchor scheme**: **anchor-free**, one candidate per location instead of RetinaFace's 2 or more. Positive assignment uses simOTA. Losses are EIoU (box), SmoothL1 (landmarks) and cross-entropy (cls/obj).
- Strides are 8/16/32. The ONNX output names (`cls_8`, `cls_16`, `cls_32`, …) are listed in the [train README](https://github.com/ShiqiYu/libfacedetection.train). DERIVED: the 2023mar file outputs 6400/1600/400 rows at a 640×640 input, i.e. one prediction per cell.
- Training (paper): from scratch, SGD, batch 16×2 on two 1080Ti GPUs, 560 epochs, random-crop scale range [0.5, 1.5].

### Model files (opencv_zoo)
| File | Size (DERIVED, LFS pointer / HF API) | Notes (PUBLISHED, zoo README) |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232,589 B | "fixed input shape"; for OpenCV 4.x DNN |
| `face_detection_yunet_2023mar_int8.onnx` | 100,416 B | int8 quantized |
| `face_detection_yunet_2023mar_int8bq.onnx` | 122,489 B | block-quantized int8, `block_size=64` |
| `face_detection_yunet_2026may.onnx` | 229,738 B | **new default**, "dynamic input shape (symbolic `height` and `width` dims), compatible with OpenCV 5.x ONNX Runtime engine (`OPENCV_FORCE_DNN_ENGINE=4`)", "re-exported from `face_detection_yunet_2023mar.onnx`" |

- DERIVED from the ONNX header of 2023mar: input `input` is `[1, 3, 640, 640]`, opset 11, 12 outputs. The README notes that "OpenCV 4.x DNN infers on the exact shape of input image" despite the fixed shape.
- PUBLISHED detectable face size: "faces of pixels between around 10x10 to 300x300 due to the training scheme" (zoo README).
- DERIVED: the 2023mar file is identical to `yunet_s_640_640.onnx` in the training repo (see Surprise 1). For the YuNet variants it publishes, the training repo carries `yunet_n_320_320.onnx` (317,434 B), `yunet_n_640_640.onnx` (317,466 B), `yunet_n_dynamic.onnx` (322,906 B), `yunet_s_320_320.onnx` (232,557 B), `yunet_s_640_640.onnx` (232,589 B), `yunet_s_dynamic.onnx` (237,823 B), `weights/yunet_n.pth` (747,497 B) and `weights/yunet_s.pth` (560,340 B). Sizes come from the GitHub contents API.
- DERIVED (for orientation only; BN folding changes the counts): the float initializers in the ONNX graphs total 53,112 elements for 2023mar/yunet_s and 73,848 for yunet_n. The paper's 54,608 and 75,856 count trainable parameters.

### Which OpenCV version does `face_detection_yunet_2023mar.onnx` need?
- PUBLISHED: opencv_zoo [PR #151 "update yunet to v2"](https://github.com/opencv/opencv_zoo/pull/151), which introduced the 2023mar model, states: "This PR requires https://github.com/opencv/opencv/pull/23020 and should be merged along with the next OpenCV release (probably 4.8.0)."
- PUBLISHED: [opencv/opencv PR #23020 "upgrade FaceDetectorYN to v2"](https://github.com/opencv/opencv/pull/23020) was merged 2023-04-06 with **milestone 4.8.0**. DERIVED via the GitHub compare API: the merge commit is contained in tag 4.8.0 (released 2023-06-28) and not in 4.7.0.
- **Answer: OpenCV ≥ 4.8.0** for `cv.FaceDetectorYN` with the 2023mar model. The current zoo `demo.py` asserts `opencv-python >= 4.10.0`, but that check belongs to the demo or benchmark. It is not a stated minimum for the model.
- With OpenCV 5.x's ONNX Runtime DNN engine, the README directs you to the 2026may dynamic-shape file instead.
- opencv_zoo benchmark numbers were produced with "opencv-python & opencv-contrib-python (current 4.10.0)".

### Published accuracy (WIDER FACE val)
| Source | Model | Easy | Medium | Hard | Protocol |
|---|---|---|---|---|---|
| [zoo README](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md) table | 2023mar fp32 | 0.8844 | 0.8656 | 0.7503 | `tools/eval`: original image size (`setInputSize(img_shape)`), confThreshold 0.3, NMS 0.45, topK 5000; python port of WiderFace-Evaluation, IoU 0.5 ([eval.py](https://github.com/opencv/opencv_zoo/blob/main/tools/eval/eval.py), [widerface.py](https://github.com/opencv/opencv_zoo/blob/main/tools/eval/datasets/widerface.py)) |
| same | 2023mar int8bq ("block") | 0.8845 | 0.8652 | 0.7504 | same |
| same | 2023mar int8 ("quant") | 0.8810 | 0.8629 | 0.7503 | same |
| same README, first sentence | (stale, see Surprise 2) | 0.834 | 0.824 | 0.708 | not stated |
| [paper Tab.1](https://link.springer.com/article/10.1007/s11633-023-1423-y) | YuNet (75,856 p) | 0.836 / 0.899 / **0.892** | 0.747 / 0.869 / **0.883** | 0.395 / 0.691 / **0.811** | input 320×320 / 640×640 / **original size (~1000 px)**; no TTA; conf 0.01; NMS 0.45 |
| same | YuNet-s (54,608 p) | 0.785 / 0.876 / **0.887** | 0.668 / 0.834 / **0.871** | 0.309 / 0.591 / **0.768** | same |
| [train README](https://github.com/ShiqiYu/libfacedetection.train) | released YuNet_n | 0.892 | 0.883 | 0.811 | `--mode origin` (not otherwise described) |
| [libfacedetection README](https://github.com/ShiqiYu/libfacedetection) | C++ lib model | 0.887 | 0.871 | 0.768 | "scales=[1.], confidence_threshold=0.02, floating point" |

The paper's comparison rows from Table 1 use the same protocol and are useful for cross-model context:

| Model | Params | FLOPs @320 | E/M/H @320 | Lat. @320 | FLOPs @640 | E/M/H @640 | Lat. @640 | E/M/H @orig | Lat. @orig |
|---|---|---|---|---|---|---|---|---|---|
| SCRFD-0.5g | 631,410 | 195 M | .850/.754/.372 | 3.4 ms | 779 M | .907/.882/.684 | 17.8 ms | .892/.885/.820 | 25.0 ms |
| RetinaFace-0.25 | 426,608 | 245 M | .765/.611/.271 | 4.2 ms | 981 M | .893/.831/.541 | 22.0 ms | .907/.883/.742 | 57.0 ms |
| YOLO5Face-n | 446,376 | 185 M | .858/.793/.445 | 7.2 ms | 741 M | .907/.880/.734 | 20.1 ms | n/a (static ONNX) | n/a |
| YuNet | 75,856 | 149 M | .836/.747/.395 | 2.2 ms | 595 M | .899/.869/.691 | 11.3 ms | .892/.883/.811 | 16.3 ms |
| YuNet-s | 54,608 | 96 M | .785/.668/.309 | 1.9 ms | 386 M | .876/.834/.591 | 8.7 ms | .887/.871/.768 | 13.8 ms |

Latency in that table is **CPU**: all models converted to ONNX, ONNX Runtime on an Intel i7-12700K, with NumPy pre/post-processing. It is averaged over the whole WIDER val set. The paper does not state a thread count. Row order was cross-checked: FLOPs @640 ≈ 4× FLOPs @320 for every row, and the YuNet/YuNet-s original-size rows match the abstract and Table 2.

### Published speed
**Paper, Table 4 (CPU)**: i7-12700K, ONNX Runtime, average of a 1000-iteration loop on one image.

| Input | YuNet latency (FPS) | YuNet-s latency (FPS) |
|---|---|---|
| 224×224 | 1.1 ms (909) | 0.9 ms (1,111) |
| 320×320 | 1.6 ms (625) | 1.4 ms (714) |
| 640×480 | 9.6 ms (104) | 7.8 ms (128) |
| 1280×960 | 36.2 ms (28) | 29.3 ms (34) |

**opencv_zoo benchmark** ([README](https://github.com/opencv/opencv_zoo/blob/main/benchmark/README.md)) for the YuNet rows:
- **Input size [160, 120]** ([config](https://github.com/opencv/opencv_zoo/blob/main/benchmark/config/face_detection_yunet.yaml): images `group.jpg`, `concerts.jpg`, `dance.jpg`; warmup 30, repeat 10).
- Times are in ms, from `cv.TickMeter.getTimeMilli` ([timer.py](https://github.com/opencv/opencv_zoo/blob/main/benchmark/utils/timer.py)), and include "preprocess, forward and postprocess". Batch size is 1.
- OpenCV is opencv-python 4.10.0.

| Board / hardware | Backend / target | Model | mean | median | min |
|---|---|---|---|---|---|
| Intel Core i7-12700K | OpenCV / **CPU** | 2023mar | 0.69 | 0.70 | 0.68 |
| Intel Core i7-12700K | OpenCV / **CPU** | 2023mar_int8 | 0.79 | 0.80 | 0.68 |
| Raspberry Pi 4B (Cortex-A72 @1.5 GHz) | **CPU** | 2023mar | 6.23 | 6.27 | 6.18 |
| Raspberry Pi 4B | **CPU** | int8 | 6.68 | 6.73 | 6.18 |
| Jetson Nano B01 (A57) | **CPU** | 2023mar | 5.62 | 5.54 | 5.52 |
| Jetson Nano B01 | **CPU** | int8 | 6.14 | 6.24 | 5.52 |
| Jetson Nano B01 (128-core Maxwell) | CUDA / **GPU FP32** | 2023mar | 10.99 | 10.71 | 9.64 |
| Jetson Nano B01 | CUDA / **GPU FP16** | 2023mar | 25.05 | 25.05 | 24.95 |
| Khadas VIM3 (A311D) | **CPU** | 2023mar | 4.62 | 4.62 | 4.53 |
| Khadas VIM3 | **CPU** | int8 | 5.24 | 5.29 | 4.53 |
| Khadas VIM3 | TIM-VX / **NPU** | int8 | 5.24 | 7.45 | 4.77 |
| Atlas 200 DK (8× A55) | **CPU** | 2023mar | 7.82 | 7.82 | 7.77 |
| Atlas 200 DK | **CPU** | int8 | 8.57 | 8.77 | 7.77 |
| Atlas 200 DK (Ascend 310) | CANN / **NPU** | **2022mar** (older model, as printed) | 2.24 | 2.21 | 2.19 |
| Toybrick RV1126 (4× A7) | **CPU** | 2023mar | 56.78 | 56.74 | 56.46 |
| Toybrick RV1126 | **CPU** | int8 | 51.16 | 51.41 | 45.18 |
| Khadas Edge2 (RK3588) | **CPU** | 2023mar | 2.30 | 2.29 | 2.26 |
| Khadas Edge2 | **CPU** | int8 | 2.70 | 2.73 | 2.26 |
| Horizon Sunrise X3 PI (4× A53) | **CPU** | 2023mar | 10.56 | 10.69 | 10.46 |
| Horizon Sunrise X3 PI | **CPU** | int8 | 12.45 | 12.60 | 10.46 |
| MAIX-III AX-PI (4× A7) | **CPU** | 2023mar | 83.95 | 83.76 | 83.62 |
| MAIX-III AX-PI | **CPU** | int8 | 79.35 | 79.92 | 75.47 |
| StarFive VisionFive 2 (RISC-V JH7110) | **CPU** | 2023mar | 41.13 | 41.07 | 41.06 |
| StarFive VisionFive 2 | **CPU** | int8 | 37.43 | 37.83 | 34.35 |
| Khadas VIM4 (A311D2) | **CPU** | 2023mar | 4.27 | 4.33 | 4.17 |
| Khadas VIM4 | **CPU** | int8 | 4.58 | 4.58 | 4.17 |
| Jetson Orin Nano (6× A78AE) | **CPU** | 2023mar | 2.59 | 2.62 | 2.50 |
| Jetson Orin Nano | **CPU** | int8 | 2.98 | 2.97 | 2.50 |
| Jetson Orin Nano (Ampere 1024-core) | CUDA / **GPU FP32** | 2023mar | 5.23 | 5.27 | 5.17 |
| Jetson Orin Nano | CUDA / **GPU FP16** | 2023mar | 5.00 | 5.04 | 4.92 |
| Atlas 200I DK A2 (4× 1.0 GHz) | **CPU** | 2023mar | 6.67 | 6.80 | 5.17 |
| Atlas 200I DK A2 | **CPU** | int8 | 8.70 | 9.22 | 5.17 |

[libfacedetection README](https://github.com/ShiqiYu/libfacedetection) (C++ library, its own embedded model, **CPU**, AVX2, i7-7820X): 640×480 50.02 ms single-thread / 6.55 ms with 16 threads; 320×240 13.09 / 1.82 ms; 160×120 3.61 / 0.57 ms. On Raspberry Pi 4B, 4 threads: 320×240 105.73 ms single-thread / 32.98 ms multi-thread.

### Landmarks, formats, downloads, runtimes
- **Landmarks**: 5 (right eye, left eye, nose tip, right and left mouth corner, per [demo.py](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/demo.py)).
- **ONNX**: yes, official.
- **TFLite**: the training repo provides an export command (`yunet_train.cli.export_tflite`). No published `.tflite` file was found.
- **Weights URLs** (verified to download):
  - https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  - `…/face_detection_yunet_2023mar_int8.onnx`, `…/face_detection_yunet_2023mar_int8bq.onnx`, `…/face_detection_yunet_2026may.onnx`
  - HF mirror: https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx (232,589 B). Note the opencv_zoo root README now says "This Project is now hosted in Hugging Face".
  - Training repo: https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/onnx/yunet_n_640_640.onnx, https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/weights/yunet_n.pth
- **Browser**: `FaceDetectorYN` is in the OpenCV.js whitelist in both [4.x](https://github.com/opencv/opencv/blob/4.x/platforms/js/opencv_js.config.py) and [5.x](https://github.com/opencv/opencv/blob/5.x/platforms/js/opencv_js.config.py). Running the ONNX through ORT Web is technically possible, but no official YuNet browser demo was verified.
- **Mobile**: per its README, libfacedetection compiles on ARM with NEON. OpenCV DNN `FaceDetectorYN` works on any OpenCV build. A specific OpenCV Android/iOS SDK test is not verified.

### Licenses
- **Code and weights, opencv_zoo dir**: "All files in this directory are licensed under [MIT License]" (zoo README). The LICENSE text reads "MIT License, Copyright (c) 2020 Shiqi Yu". The opencv_zoo repo as a whole is Apache-2.0, with the note "Please refer to licenses of different models".
- **Training repo** libfacedetection.train: BSD-3-Clause, "Copyright (c) 2022-2026, Shiqi Yu". The repo includes `.pth` and `.onnx` files and states no separate weights license.
- **libfacedetection**: "License Agreement For libfacedetection (3-clause BSD License)". GitHub classifies it as "Other".
- **Training data**: WIDER FACE (see §6). The train README states "The `labelv2` annotations come from SCRFD". The older README (commit a61a428) says "We relabled the WIDER Face train set using RetinaFace".

---

## 2. SCRFD (InsightFace)

### Sources fetched
- https://github.com/deepinsight/insightface/tree/master/detection/scrfd (README and `LICENSE`)
- Paper: Guo, Deng, Lattas, Zafeiriou, "Sample and Computation Redistribution for Efficient Face Detection", ICLR 2022, https://arxiv.org/abs/2105.04714
- https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- https://github.com/deepinsight/insightface/blob/master/python-package/README.md and https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md (insightface 2.1 on PyPI)
- Root README license section: https://github.com/deepinsight/insightface#license
- Releases: https://github.com/deepinsight/insightface/releases/tag/model-zoo (published 2026-09-03) and https://github.com/deepinsight/insightface/releases/tag/v0.7 (2023-04-02)

### Architecture (PUBLISHED, paper §4–5)
- The base design is RetinaNet-style. The **backbone** comes from computation-redistribution search:
  - basic residual blocks for 2.5GF and 10GF;
  - bottleneck residual blocks for 34GF;
  - depthwise convs for the 0.5GF "mobile regime".
- **Neck**: PAFPN (top-down plus bottom-up).
- **Head**: stacked 3×3 convs, shared across the 3 strides, with weight sharing and Group Norm.
- **Anchors**: {16, 32} at stride 8, {64, 128} at stride 16, {256, 512} at stride 32; aspect ratio 1.0; nothing at stride 4. DERIVED: `det_500m.onnx` emits 12800/3200/800 rows at a 640×640 input, which is 2 anchors per cell.
- Positive assignment is ATSS. Losses are Generalised Focal Loss and DIoU.
- Training uses "Sample Redistribution", a random crop range of [0.3, 2.0]. The paper also notes that "InsightFace is a nonprofit face project."
- `_KPS` variants add **5 keypoints** (README).

### Published table ([SCRFD README](https://github.com/deepinsight/insightface/tree/master/detection/scrfd), mirrored in [model_zoo README](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md))
"mAP, FLOPs and inference latency are all evaluated on VGA resolution." The model_zoo README adds: "evaluated with single scale testing, VGA resolution."

| Name | Easy | Medium | Hard | FLOPs | Params (M) | Infer (ms) |
|---|---|---|---|---|---|---|
| SCRFD_500M | 90.57 | 88.12 | 68.51 | 500M | 0.57 | 3.6 |
| SCRFD_1G | 92.38 | 90.57 | 74.80 | 1G | 0.64 | 4.1 |
| SCRFD_2.5G | 93.78 | 92.16 | 77.87 | 2.5G | 0.67 | 4.2 |
| SCRFD_10G | 95.16 | 93.87 | 83.05 | 10G | 3.86 | 4.9 |
| SCRFD_34G | 96.06 | 94.92 | 85.29 | 34G | 9.80 | 11.7 |
| SCRFD_500M_KPS | 90.97 | 88.44 | 69.49 | 500M | 0.57 | 3.6 |
| SCRFD_2.5G_KPS | 93.80 | 92.02 | 77.13 | 2.5G | 0.82 | 4.3 |
| SCRFD_10G_KPS | 95.40 | 94.01 | 82.80 | 10G | 4.23 | 5.0 |

- **Hardware for "Infer(ms)"**: the README doesn't say. The paper's Table 4 caption does: "'Infer' refers to network inference latency with VGA resolution (640 × 480) images on **NVIDIA 2080TI**" (**GPU**). "#Flops denote … multiply-adds."
- **Protocol** (paper §5.1): "We only employ single-scale testing, bounded by the VGA resolution (640 × 480)." The README's evaluation script is `tools/test_widerface.py`, which it describes as "a pure python evaluation script without Matlab".
- **CPU** (README, "X64 CPU Performance of SCRFD-0.5GF", AMD Ryzen 9 3950X, "simple PyTorch CPU inference by setting `OMP_NUM_THREADS=1` (no mkldnn)"):
  - Original size: E/M/H 90.91 / 89.49 / 82.03, no time given.
  - 640×480: **28.3 ms**, 90.57 / 88.12 / 68.51.
  - 320×240: **11.4 ms**, AP not given.
- The YuNet paper's measurement of SCRFD-0.5g (ONNX Runtime CPU, i7-12700K) gives 631,410 params and 195 M FLOPs @320. See the §1 table.

### Which model packs contain `det_500m.onnx`, and where to download them
The pack tables in the [python-package model guide](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md) and the [model_zoo README](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) list detectors by family. File names and sizes are DERIVED from each release ZIP's central directory.

| Pack | Detector (PUBLISHED) | Detector file (DERIVED) | ZIP size (DERIVED, GitHub API) | GitHub release URL |
|---|---|---|---|---|
| `buffalo_sc` | SCRFD-500MF | `det_500m.onnx` 2,524,817 B (plus `w600k_mbf.onnx`) | 14,969,382 B | https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip |
| `buffalo_s` | SCRFD-500MF | `det_500m.onnx` 2,524,817 B | 127,607,557 B | https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_s.zip |
| `cheetah_s` (new in 2.1) | "identical to those in `buffalo_s`" | `det_500m.onnx` 2,524,817 B | 27,116,086 B | https://github.com/deepinsight/insightface/releases/download/model-zoo/cheetah_s.zip |
| `buffalo_m` | SCRFD-2.5GF | `det_2.5g.onnx` 3,292,009 B | 275,951,529 B | https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_m.zip |
| `buffalo_l` (default) | SCRFD-10GF | `det_10g.onnx` 16,923,827 B | 288,621,354 B | https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip |
| `antelopev2` | SCRFD-10GF | `scrfd_10g_bnkps.onnx` 16,923,827 B | 360,662,982 B | https://github.com/deepinsight/insightface/releases/download/model-zoo/antelopev2.zip |
| `raccoon_s` / `raccoon_l` | not stated | `det_10g_wo.onnx` 16,940,585 B | 36,573,350 / 255,140,324 B | `…/model-zoo/raccoon_s.zip`, `…/model-zoo/raccoon_l.zip` |

- The same buffalo/antelope ZIPs, with identical byte sizes, also exist under the older tag `v0.7`, e.g. https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip. The `model-zoo` release notes say: "Legacy model files mirrored here are identical to their previously released versions unless explicitly stated otherwise."
- PUBLISHED: auto-download in insightface ≥0.3.3 uses the `model-zoo` release. Only `buffalo_l` is marked "Auto: Y" in the legacy pack table; the others are manual.
- DERIVED from the ONNX header: `det_500m.onnx` has input `[1, 3, ?, ?]` (dynamic H/W), opset 11, and 9 outputs, including 10-wide keypoint tensors. **It is therefore a KPS (5-landmark) model.** The README itself states: "Generated onnx model can accept dynamic input as default."
- The python lib's SCRFD defaults to "Auto: 128x128 + 640x640" detection size (python-package README).
- **Original .pth weights** for SCRFD_500M_KPS:
  - OneDrive: https://1drv.ms/u/s!AswpsDO2toNKri_NDM0GIkPpkE2f?e=JkebJo (SCRFD README)
  - Google Drive: https://drive.google.com/file/d/1TXvKmfLTTxtk7tMd2fEf-iWtAljlWDud/view?usp=sharing (model_zoo README)
- **TFLite**: not verified (none official).
- **Browser/mobile**: no official browser runtime; the insightface `web-demos/` directory holds a Python CLI and a Discord bot. The python lib uses ONNX Runtime, picking providers in the order "CoreMLExecutionProvider → CUDAExecutionProvider → CPUExecutionProvider". The SCRFD README lists third-party demos: [ncnn-android-scrfd](https://github.com/nihui/ncnn-android-scrfd), plus MNN, TNN, NCNN, ORT and TensorRT ports.

### License wording (exact quotes)
- Root README ([#license](https://github.com/deepinsight/insightface#license)):
  > "The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage.
  > The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.
  > Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy(which is for non-commercial research purposes only)."

  Under "`2025-11-24 Update:`" it continues: "For open-sourced face recognition models (e.g., buffalo_l package), please contact recognition-oss-pack@insightface.ai for licensing."
- model_zoo README: "**ALL models are available for non-commercial research purposes only.**"
- python-package README: "The library code is released under the **MIT License**, for academic and commercial use. **The pretrained models provided with this library are for non-commercial research only**, whether downloaded automatically or manually."
- python-package model guide: "Model licenses apply separately from the library's MIT license." It also says: "Keep the signed `MODEL.LICENSE` included with the package; public packages carry a non-commercial grant."
- DERIVED from the files themselves: `MODEL.LICENSE` in `raccoon_s.zip` and `cheetah_s.zip` (the latter ships `det_500m.onnx`) contains `"issuer": "InsightFace"`, `"grant": "non-commercial"`, plus a signature.
- **Code-license nuance**: GitHub detects **no repository-level LICENSE file** for deepinsight/insightface (the license API returns none); MIT is asserted only in the READMEs. The `detection/scrfd/` training code ships its own `LICENSE`, which is **Apache-2.0** with header "Copyright 2018-2019 Open-MMLab" (it is mmdetection-based).

---

## 3. MediaPipe Face Detector / BlazeFace

### Sources fetched
- Overview: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector. The requested https://developers.google.com/edge/mediapipe/solutions/vision/face_detector also returns HTTP 200 with the same guide.
- Platform guides: `…/face_detector/web_js`, `…/android`, `…/ios`, `…/python`
- Setup guides: https://ai.google.dev/edge/mediapipe/solutions/setup_web, `setup_android`, `setup_ios`, `setup_python`
- Model cards:
  - [Short Range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)
  - [Full Range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Full%20Range).pdf)
  - [Sparse Full Range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Sparse%20Model%20Card%20(Full%20Range).pdf)
- Paper: Bazarevsky et al., "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs", https://arxiv.org/abs/1907.05047
- Tasks graph source: https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_detector/face_detector_graph.cc

### Models offered by the current Tasks API
The overview lists three models, each with a download link, as "variants of BlazeFace":

| Model | `.tflite` URL (PUBLISHED link; DERIVED size from HTTP HEAD) | Docs "Input shape" | Actual input tensor (DERIVED) | Model card input | Architecture (card) | Intended range (card, "out-of-scope") |
|---|---|---|---|---|---|---|
| BlazeFace (short-range) | https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite (229,746 B; last-modified 2023-04-26) | 128 x 128, float16 | `[1,128,128,3]`; outputs `[1,896,16]`, `[1,896,1]` | 128x128, values in [-1, 1] | "SSD-like with a custom encoder" | "Detecting people too far away from the camera (e.g. further than **2 meters**)" is out of scope |
| BlazeFace (full-range) | https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite (1,083,786 B; last-modified 2026-03-11) | 128 x 128 (**docs error**) | `[1,192,192,3]`; outputs `[1,2304,16]`, `[1,2304,1]` | "160x192" | "CenterNet-like with a custom encoder" | "further than **5 meters**" is out of scope |
| BlazeFace Sparse (full-range) | https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range_sparse.tflite (676,746 B; last-modified 2026-03-11) | 128 x 128 (**docs error**) | `[1,192,192,3]` | "160x192" | CenterNet-like; "Most of the 1x1 convolutions … pruned to have 70% of zeros"; docs: "roughly 60% smaller" | "further than 5 meters" |

- **Full-range in the Tasks API is new.** [Commit af06e59](https://github.com/google-ai-edge/mediapipe/commit/af06e59f48), "Add support for FULL_RANGE face detection task", landed 2026-03-06. DERIVED via the GitHub compare API: the commit is in release **v0.10.33** (2026-03-23) and not in v0.10.32. The graph source hard-codes `kShortRangeImageSize = 128` and `kLongRangeImageSize = 192`, which confirms a 192 px full-range input.
- Anchors in the graph source, for legacy models without metadata:
  - Short-range: 4 layers, strides 8/16/16/16, aspect 1.0, fixed anchor size. That gives 896 anchors, matching the paper's "2 anchors per pixel" at 16×16 plus "6 anchors at 8×8".
  - Full-range: 1 layer, stride 4 at 192 px, giving 2,304 anchors.
- **Keypoints: 6** ("left eye, right eye, nose tip, mouth, left eye tragion, and right eye tragion"; docs and model cards).
- Config defaults (docs): `min_detection_confidence` 0.5, `min_suppression_threshold` 0.3. Running modes are IMAGE, VIDEO and LIVE_STREAM. On Web only IMAGE and VIDEO exist.
- Model-card limitation: "Produces only up to a given limit (e.g. 10) of detections."
- Short-range card: "Face bounding box sides should be at least 20% of the corresponding image sides". The paper uses ">20% of the image area" for the frontal model and 5% for the rear model.

### Paper architecture (PUBLISHED, frontal model)
- 128×128 RGB input.
- A 2D conv, then 5 single BlazeBlocks, then 6 double BlazeBlocks, with 5×5 depthwise kernels.
- Maximum depth 96 channels; lowest spatial resolution 8×8.
- SSD-derived anchor scheme ending at 8×8, with 1:1 anchors only.
- A "blending" tie-resolution strategy replaces NMS. The paper reports it cut jitter by 40% (front camera) and 30% (rear camera).
- Trained on 66K images; evaluated on a private, geographically diverse set of 2K images.

### Published speed
- **Docs, "Task benchmarks"**: "average latency on Pixel 6 using CPU / GPU" for the whole pipeline. BlazeFace (short-range): **CPU 2.94 ms, GPU 7.41 ms** (**mobile**). No full-range row is published.
- **Model cards (mobile)**:
  - Short-range: "~275FPS on Pixel 2 single-core CPU with XNNPACK".
  - Full-range: "~90FPS on Pixel 3 GPU, ~165FPS on Pixel 4 GPU, ~50FPS on Pixel 3 single-core CPU with XNNPACK, ~145FPS on Pixel 4 CPU with XNNPACK".
  - Sparse: same as full-range except "~160FPS on Pixel 4 CPU".
- **Paper, Table 2 (mobile GPU, TFLite GPU fp16, frontal model)**: iPhone 7 1.8 ms, iPhone XS 0.6 ms, Pixel 3 3.4 ms, Huawei P20 5.8 ms, Galaxy S9+ (SM-G965U1) 3.7 ms.
- **Accuracy**: the paper reports AP 98.61% on a *private* frontal dataset (MobileNetV2-SSD baseline 97.95%), and median keypoint error 7.4% of inter-ocular distance. **No WIDER FACE numbers are published by Google.** The "BlazeFace-FPN-SSH (paddle)" row in the InsightFace model zoo (91.9 / 89.8 / 81.7%) is a different PaddleDetection model, not MediaPipe's.
- Parameter count and FLOPs: **not verified** (not published in the paper or the cards).

### Runtimes
- **Web**: `@mediapipe/tasks-vision` from npm (latest 1.2.0, 2026-10-27; npm license field Apache-2.0).
  - `BaseOptions.delegate`: "Enables hardware acceleration through a device delegate … Default value: CPU", with options `[CPU, GPU]` ([setup_web](https://ai.google.dev/edge/mediapipe/solutions/setup_web)).
  - Supported platforms: "Chrome or Safari browser".
  - `detect()` and `detectForVideo()` "run synchronously and block the user interface thread"; the docs suggest web workers.
- **Android**: delegate `[CPU, GPU]`, GPU via `BaseOptions.builder().useGpu()` ([setup_android](https://ai.google.dev/edge/mediapipe/solutions/setup_android)).
- **iOS**: "at least iOS 15.0", Xcode 14.1+, 64-bit only: arm64 devices; arm64 and x86_64 simulators ([setup_ios](https://ai.google.dev/edge/mediapipe/solutions/setup_ios)).
- **Python**: "Supports version 3.9 and later"; Windows, Mac, Linux, Raspberry OS 64-bit.

### Licenses
- **Framework**: Apache-2.0 ([LICENSE](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE); PyPI `mediapipe` 1.2.0 says "Apache 2.0").
- **Weights**: each model card states "LICENSED UNDER Apache License, Version 2.0".
- **Training data** (model card): "trained and evaluated on consented images of people using a mobile AR application". It is **not** trained on WIDER FACE.
- **Intended-use statement** (all cards; not a license term): "Any form of surveillance or identity recognition is explicitly out of scope and not enabled by this technology."

---

## 4. RetinaFace-MobileNet0.25 (optional)

### Sources fetched
- Paper: https://arxiv.org/abs/1905.00641 (CVPR 2020)
- InsightFace page: https://github.com/deepinsight/insightface/tree/master/detection/retinaface
- PyTorch port: https://github.com/biubug6/Pytorch_Retinaface (README, `data/config.py`, `test_widerface.py`, `LICENSE.MIT`)

### Architecture
- **Paper**:
  - Full model: FPN P2–P6, independent context modules (SSH-style, with DCN), and a shared 1×1 multi-task head for score, box, **5 landmarks** and an optional dense 3D mesh.
  - Light-weight variant: "7 × 7 convolution with stride=4 on the input image, tile dense anchors on P3, P4 and P5 … and remove deformable layers"; "MobileNet-0.25 …, size of 1MB, and AP 78.2% on the WIDER FACE hard set".
- **Pytorch_Retinaface `cfg_mnet`**: `min_sizes` [[16, 32], [64, 128], [256, 512]], `steps` [8, 16, 32], `image_size` 640 for training, `out_channel` 64.

### Published numbers
- **Pytorch_Retinaface README**: "Model size only 1.7M, when Retinaface use mobilenet0.25 as backbone net." WIDER val, single scale, MobileNet0.25:

  | Style | Easy | Medium | Hard |
  |---|---|---|---|
  | Pytorch (same parameter with Mxnet) | 88.67 | 87.09 | 80.99 |
  | Pytorch (original image scale) | 90.70 | 88.16 | 73.82 |
  | Mxnet | 88.72 | 86.97 | 79.19 |
  | Mxnet (original image scale) | 89.58 | 87.11 | 69.12 |

  The README does not define "same parameter with Mxnet". In `test_widerface.py`, the non-`origin_size` path resizes the short side to 1600 px, capping the long side at 2150. The defaults are `confidence_threshold` 0.02 and NMS 0.4.
- **InsightFace RetinaFace page**: third-party "RetinaFace-MobileNet0.25 … WiderFace validation mAP: Hard 82.5. (model size: 1.68Mb)". The model_zoo README notes "In RetinaFace, mAP was evaluated with multi-scale testing."
- **SCRFD README, at VGA**: RetinaFace MobileNet0.25 87.78 / 81.16 / 47.32, 0.44 M params, 0.802 GFLOPs, 7.9 ms (2080Ti **GPU**).
- **YuNet paper, Table 1** (ONNX Runtime, i7-12700K **CPU**): 426,608 params, 245 MFLOPs @320.
  - @320: .765 / .611 / .271, 4.2 ms
  - @640: .893 / .831 / .541, 22.0 ms
  - original size: .907 / .883 / .742, 57.0 ms
- **RetinaFace paper, Table 5** ("Inference time (ms)", via TVM):

  | Input | MobileNet-0.25 GPU (Tesla P40) | CPU-m (multi-thread, i7-6700K) | CPU-1 (single-thread, i7-6700K) | ARM (RK3399, A72×2) |
  |---|---|---|---|---|
  | VGA 640×480 | 1.4 | 5.5 | 17.2 | 61.2 |
  | HD 1920×1080 | 6.1 | 50.3 | 130.4 | 434.3 |
  | 4K 4096×2160 | 25.6 | – | – | – |
- **Landmarks**: 5.
- **ONNX**: export script `convert_to_onnx.py` only; no published ONNX file. **TFLite**: not verified.
- **Weights**: `mobilenet0.25_Final.pth` lives in a Google Drive folder, https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1, with a Baidu mirror (password `fstq`). **No direct file URL is published.**
- **Licenses**: code is MIT ("Copyright (c) 2019", `LICENSE.MIT`). The README says nothing about a weights license. Training annotations come from the Baidu link `pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA`, the **same link** the InsightFace RetinaFace page gives for "our annotations".

---

## 5. Trackers (published results are pedestrian MOT, **not faces**)

### SORT
- Paper: https://arxiv.org/abs/1602.00763 (ICIP 2016). Repo: https://github.com/abewley/sort
- **Method**: Kalman filter with state (u, v, s, r) and constant velocity, plus the Hungarian algorithm on an IoU cost matrix with a minimum IoU threshold. Appearance features are not used.
- **Published results**, MOT benchmark 2D MOT 2015 test (**pedestrians**, FrRCNN(VGG16) detections): MOTA 33.4, MOTP 72.1, FAF 1.3, MT 11.7%, ML 30.9%, FP 7,318, FN 32,615, ID sw 1,001, Frag 1,764 (paper Table 2).
- **Speed**: "The tracking component runs at 260 Hz on single core of an Intel i7 2.5GHz machine with 16 GB memory" (**CPU**, tracker only, detector excluded).
- README caveat: "A significant proportion of SORT's accuracy is attributed to the detections."
- **License**: "SORT is released under the GPL License … If you require a permissive license contact Alex (alex@bewley.ai)." The LICENSE file is **GNU GPL v3.0**.

### ByteTrack
- Paper: https://arxiv.org/abs/2110.06864 (ECCV 2022). Repo: https://github.com/ifzhang/ByteTrack
- **Method (BYTE)**:
  - Kalman prediction, then two-stage association. High-score detections are matched first (IoU or Re-ID, Hungarian). Remaining tracks are then matched to *low-score* detections by IoU only.
  - Defaults (paper): detection score threshold 0.6; matches with IoU < 0.2 rejected; lost tracks kept for 30 frames.
  - The detector is YOLOX-X.
- **Published results (README)**, MOT test sets, **pedestrians**:
  - MOT17: MOTA 80.3, IDF1 77.3, HOTA 63.1, IDs 2,196, FPS 29.6.
  - MOT20: MOTA 77.8, IDF1 75.2, HOTA 61.3, IDs 1,223, FPS 13.7.
- **Speed definition**: FPS "is measured with FP16-precision and batch size of 1 on a single GPU" (V100) and includes the YOLOX detector. No tracker-only latency is published.
- Ablation on MOT17 val: BYTE vs SORT association is MOTA 76.6 vs 74.6, IDF1 79.3 vs 76.9.
- **License**: MIT ("Copyright (c) 2021 Yifu Zhang").

**Caveat for this report**: no face-tracking benchmark numbers were found for either tracker in official sources. The MOT numbers measure pedestrian detection plus association, with a heavy detector, and do not transfer to faces.

---

## 6. Dataset: WIDER FACE

Sources: http://shuoyang1213.me/WIDERFACE/; the technical report https://arxiv.org/abs/1511.06523; annotation ZIP http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip; HF card https://huggingface.co/datasets/CUHK-CSE/wider_face (linked from the official site as a download mirror).

- **Size (PUBLISHED)**: "32,203 images and label 393,703 faces", organized in 61 event classes. Each class is split "randomly … 40%/10%/50% data as training, validation and testing sets". Test ground truth is not released.
- **Validation set size (DERIVED** by counting the official `wider_face_val_bbx_gt.txt`): **3,226 images, 39,708 face boxes**, of which **585** carry `invalid = 1`.
  - Train: 12,880 images, 159,420 boxes (2,399 invalid).
  - Test file list: 16,097 images.
  - The three splits sum to 32,203.
- **"Invalid" faces**:
  - The annotation `readme.txt` defines a per-box attribute, `invalid: false->0(valid image) true->1(invalid image)`, in the format `x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose`.
  - The technical report says: "Similar to the PASCAL VOC dataset, we assign an 'Ignore' flag to the face which is very difficult to be recognized due to low resolution and small scale (10 pixels or less)."
  - The sources do not explicitly say that the `invalid` attribute *is* the paper's 'Ignore' flag.
- **Difficulty levels**: the RetinaFace paper describes them as "Based on the detection rate of EdgeBox, three levels of difficulty (i.e. Easy, Medium and Hard) are defined by incrementally incorporating hard samples". The YuNet paper notes "Hard covers Medium and Easy".
- **Terms of use**: the official site's "License" section reads "**Creative Common License (cc by-nc-nd)**", with no version stated. The HF dataset card gives "Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)".
- **Image provenance**: per the report, images come from the WIDER dataset and were "retrieved using search engines like Google and Bing".

### What sources say about models trained on WIDER FACE (no speculation)
| Model | Trained on WIDER FACE? (source) | What the source says about weights / commercial use |
|---|---|---|
| YuNet | Yes ([train README](https://github.com/ShiqiYu/libfacedetection.train): "YuNet face detection training on WIDER Face"; annotations "labelv2 … come from SCRFD") | opencv_zoo: all files MIT. libfacedetection.train: BSD-3. Neither mentions dataset terms. |
| SCRFD | Yes (paper; README data prep) | InsightFace: "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only." |
| RetinaFace (InsightFace) | Yes (paper) | Same InsightFace policy. |
| RetinaFace (Pytorch_Retinaface) | Yes (README), using InsightFace's landmark annotations (same Baidu link) | Repo is MIT; nothing is said about weights or dataset terms. |
| MediaPipe BlazeFace | **No**. Model card: consented images from a mobile AR app | Apache-2.0 model card. |

Only InsightFace explicitly extends non-commercial terms to models trained on its annotated data. No other source addresses whether WIDER FACE's NC-ND terms carry over to trained weights.

---

## 7. Runtimes

### ONNX Runtime Web
Sources: https://onnxruntime.ai/docs/get-started/with-javascript/web.html, https://onnxruntime.ai/docs/tutorials/web/, https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html, https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html, https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html, https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html. npm `onnxruntime-web` latest is 1.31.1 (2026-10-26), MIT.

- **EPs**: "webgl, webgpu or webnn (with deviceType set to gpu) for GPU processing, and WebAssembly (wasm, alias to cpu) or webnn (with deviceType set to cpu) for CPU processing. All ONNX operators are supported by WASM but only a subset are currently supported by WebGL, WebGPU and WebNN."
- **Published support matrix** (get-started page, verbatim):

| EP | Chrome/Edge (Windows) | Chrome/Edge (Android) | Chrome/Edge (macOS) | Chrome/Edge (iOS) | Safari (macOS) | Safari (iOS) | Firefox (Windows) | Node.js |
|---|---|---|---|---|---|---|---|---|
| WebAssembly (CPU) | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ [1] |
| WebGPU | ✔️ [2] | ✔️ [3] | ✔️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| WebGL | ✔️ [4] | ✔️ [4] | ✔️ [4] | ✔️ [4] | ✔️ [4] | ✔️ [4] | ✔️ [4] | ❌ |
| WebNN | ✔️ [5] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

[1] Node.js supports only single-threaded wasm. [2] "WebGPU requires Chromium v113 or later on Windows. Float16 support requires Chrome v121 or later, and Edge v122 or later." [3] "WebGPU requires Chromium v121 or later on Windows" (sic; this is the Android column). [4] "WebGL support is in maintenance mode." [5] WebNN requires the `--enable-features=WebMachineLearningNeuralNetwork` flag.

- **Caveats (PUBLISHED)**:
  - The get-started page still labels WebGPU an "experimental feature", imported via `onnxruntime-web/webgpu`.
  - The WebGPU tutorial says Firefox has it "behind a flag" and Safari only in "Technology Preview". Both statements are **out of date** relative to browser release notes (below).
  - WASM multi-threading is enabled "only when the browser supports WebAssembly multi-threading and crossOriginIsolated mode is enabled".
  - "The proxy worker cannot work with WebGPU EP."
  - The browser path currently ships the JSEP-based WebGPU backend; the native Dawn-based WebGPU EP is separate (WebGPU EP page).

### WebGPU browser status (per caniuse data updated 2026-10-29 and official release notes)
caniuse source: https://github.com/Fyrd/caniuse/blob/main/features-json/webgpu.json (backs https://caniuse.com/webgpu).

| Browser | Status | Official source and date |
|---|---|---|
| Chrome / Edge desktop | Supported since **113**. caniuse notes: Linux not enabled by default (113–143); from 144, "Support on Linux depends on hardware and drivers". | [Chrome blog](https://developer.chrome.com/blog/webgpu-release), last updated 2023-04-06: "available in Chrome 113 on ChromeOS devices with Vulkan support, Windows devices with Direct3D 12 support, and macOS" |
| Chrome Android | Supported (caniuse) | [Chrome blog](https://developer.chrome.com/blog/new-in-webgpu-121), last updated 2024-01-18: "enabled by default in Chrome 121 on devices running Android 12 and greater powered by Qualcomm and ARM GPUs" |
| Firefox desktop | **Partial**. 141: "Only enabled by default on Windows". 145+: Windows plus macOS 26 Tahoe on Apple Silicon. | [Firefox 141 notes](https://www.mozilla.org/en-US/firefox/141.0/releasenotes/) (2025-07-22): "Enabled the WebGPU API … on Windows." [Firefox 145 notes](https://www.mozilla.org/en-US/firefox/145.0/releasenotes/) (2025-11-11): "available on macOS 26 (Tahoe) on Apple Silicon" |
| Firefox Android | Not enabled; flag only (caniuse) | – |
| Safari macOS | **Partial** from 26.0: "only being enabled by default on macOS 26 Tahoe or later" (caniuse) | [WebKit blog](https://webkit.org/blog/17333/webkit-features-in-safari-26-0/) (2025-09-15): "WebKit for Safari 26.0 adds support for WebGPU" |
| Safari iOS / iPadOS | Supported from **26.0** (caniuse) | same WebKit post (covers iOS 26, iPadOS 26, macOS 26, visionOS 26) |
| Samsung Internet | Supported from 24 (caniuse) | – |

### MediaPipe Tasks Vision JS
See §3. `@mediapipe/tasks-vision` 1.2.0 offers `delegate: "CPU" | "GPU"`, default CPU. Platforms: "Chrome or Safari browser". Calls are synchronous on the calling thread. The docs do not say which web GPU API (WebGL or WebGPU) backs the GPU delegate: **not verified**.

### ONNX Runtime Mobile / NNAPI / CoreML
Sources: https://onnxruntime.ai/docs/tutorials/mobile/, https://onnxruntime.ai/docs/execution-providers/, https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html, https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html, https://developer.android.com/ndk/guides/neuralnetworks/migration-guide, https://developer.android.com/ndk/guides/neuralnetworks

- **Packages**: `onnxruntime-android`, `onnxruntime-c` / `onnxruntime-objc` (iOS). "Applications that run on Android also have support for NNAPI and XNNPACK"; "Applications that run on iOS also have support for CoreML and XNNPACK".
- ORT guidance: "If the model is quantized, start with the CPU Execution Provider. If the model is not quantized start with XNNPACK."
- Published binary size, ORT 1.18.0: the prebuilt AAR is 24,415,212 B; a custom ResNet50-only build is 7,532,309 B.
- **NNAPI EP**: "requires Android devices with Android 8.1 or higher … recommended … Android 9 or higher". The ORT page as fetched says nothing about deprecation.
- **Android official docs**: "The Neural Networks API (NNAPI) is deprecated. It was introduced in Android 8.1 … and **deprecated in Android 15**." The NNAPI guide adds: "we expect the majority of devices in the future to use the CPU backend … we recommend migrating to alternative solutions, for example the TF Lite GPU runtime."
- **CoreML EP**: "requires iOS devices with iOS 13 or higher, or Mac computers with macOS 10.15 or higher".
  - `MLProgram` format "Requires Core ML 5 or later (iOS 15+ or macOS 12+)"; the default is `NeuralNetwork`.
  - `OrtSessionOptionsAppendExecutionProvider_CoreML` is deprecated in ORT 1.20.0.
  - The ORT EP summary table still labels it "CoreML (preview)".

---

## 8. Licenses of runtime dependencies

| Dependency | License (PUBLISHED) | Source |
|---|---|---|
| OpenCV | **Apache-2.0 since 4.5.0**. [PR #18073](https://github.com/opencv/opencv/pull/18073), "changed OpenCV license from BSD to Apache 2 license", milestone 4.5.0, merged 2020-08-17; 4.5.0 released 2020-10-12. Before that: BSD. | https://github.com/opencv/opencv/blob/5.x/LICENSE; PyPI `opencv-python` 5.0.0.93: "Apache 2.0" |
| opencv_zoo (repo) | Apache-2.0, "Please refer to licenses of different models" (YuNet dir: MIT) | https://github.com/opencv/opencv_zoo/blob/main/LICENSE |
| ONNX Runtime (incl. Web/Mobile) | MIT | https://github.com/microsoft/onnxruntime/blob/main/LICENSE; npm `onnxruntime-web` MIT |
| MediaPipe | Apache-2.0 | https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE; npm `@mediapipe/tasks-vision` Apache-2.0 |
| py-motmetrics (`motmetrics` 1.4.0) | MIT ("Copyright (c) 2017-2020 Christoph Heindl") | https://github.com/cheind/py-motmetrics/blob/develop/LICENSE |
| TrackEval | MIT ("Copyright (c) 2020 Jonathon Luiten") | https://github.com/JonathonLuiten/TrackEval/blob/master/LICENSE. Note: PyPI `trackeval` 1.3.0 is published from a **fork**, https://github.com/kovalp/TrackEval (MIT on PyPI). |
| pycocotools | BSD 2-clause ("FreeBSD"; "Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin") | https://github.com/cocodataset/cocoapi/blob/master/license.txt. Note: PyPI `pycocotools` 2.0.11 is published from the fork https://github.com/ppwwyyxx/cocoapi (license "FreeBSD"). |
| SciPy | BSD-3-Clause | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| libfacedetection | 3-clause BSD (custom header) | https://github.com/ShiqiYu/libfacedetection |
| SORT | GPL-3.0 | https://github.com/abewley/sort/blob/master/LICENSE |
| ByteTrack | MIT | https://github.com/ifzhang/ByteTrack/blob/main/LICENSE |

---

## 9. Licensing risks

These are ranked by how directly an official source restricts use. They report what the sources say and are not legal advice.

1. **SCRFD / InsightFace weights: explicitly non-commercial.** This covers `det_500m.onnx`, `det_2.5g.onnx`, `det_10g.onnx`, every buffalo/antelope/raccoon/cheetah pack, and the original `.pth` files. Four separate official statements apply: the root README, the model_zoo README, the python-package README and model guide, and the signed `MODEL.LICENSE` (`"grant": "non-commercial"`) inside the 2026 packs. MIT applies only to library *code*. The repo also has no top-level LICENSE file, and the `detection/scrfd/` training code is Apache-2.0. The 2025-11-24 README update names contact addresses for commercial licensing. **Shipping SCRFD weights in a commercial product requires a separate license from InsightFace.**
2. **SORT is GPL-3.0.** Distributing software that incorporates `sort.py` triggers GPL-3.0 copyleft obligations. The README offers a permissive license on request (alex@bewley.ai). ByteTrack (MIT) has no such restriction.
3. **WIDER FACE is CC BY-NC-ND** (site: "Creative Common License (cc by-nc-nd)"; HF card: CC BY-NC-ND 4.0). YuNet, SCRFD and both RetinaFace releases are trained on it.
   - Only InsightFace states that the restriction extends to trained models.
   - opencv_zoo (MIT), libfacedetection(.train) (BSD-3) and Pytorch_Retinaface (MIT) license their weights permissively and do not address the dataset terms.
   - Commercial use of WIDER-trained weights is therefore **unresolved by the sources** and should go to legal review.
   - Using the WIDER val set for an internal evaluation is also subject to the NC-ND terms.
4. **Annotation provenance couples YuNet and Pytorch_Retinaface to InsightFace data.**
   - YuNet's current training repo uses "`labelv2` annotations [that] come from SCRFD". The pre-2023 README says the train set was "relabled … using RetinaFace".
   - Pytorch_Retinaface's landmark annotations use the same Baidu link as InsightFace's "our annotations".
   - InsightFace states that "the training data containing the annotation (and the models trained with these data)" are non-commercial.
   - Neither the YuNet nor the Pytorch_Retinaface weights license mentions this.
5. **Pytorch_Retinaface weights have no stated license.** They are hosted on Google Drive and Baidu; only the code is MIT.
6. **MediaPipe BlazeFace has the lowest licensing risk among the candidates.** The framework and each model card are Apache-2.0, and training data is described as consented and in-house, not WIDER. The model cards do add an intended-use statement: "Any form of surveillance or identity recognition is explicitly out of scope." That is not a license term, but it should be noted if the product does identity work.
7. **Packaging forks.** The PyPI `trackeval` (kovalp fork) and `pycocotools` (ppwwyyxx fork) packages are not published by the original authors. Their licenses are still permissive (MIT and BSD-2).
8. **Version and identity drift (not licensing, but affects citations).** The opencv_zoo "YuNet" 2023mar file is YuNet-s, and the README's headline AP is stale; see Surprises 1–2. Cite the paper's YuNet-s rows, or the zoo's own `tools/eval` table, for that file, not the 75,856-parameter / 81.1% Hard headline.
