# Benchmark summary

Face detectors and trackers compared on two real YouTube videos (streamed, never saved), CPU only
(12th Gen Intel(R) Core(TM) i7-12650H). **Full 3-page report: [docs/BENCHMARK_REPORT.pdf](docs/BENCHMARK_REPORT.pdf)**
· raw numbers: [results/youtube/summary.json](results/youtube/summary.json).

| Detector | ms/frame (median) | A: faces/frame | A: F1 vs ref | B: faces/frame | B: F1 vs ref |
|---|---|---|---|---|---|
| YuNet @640 | 5.1 | 3.3 | 0.63 | 5.0 | 0.76 |
| YuNet @native | 22.4 | 5.8 | 0.81 | 8.4 | 0.72 |
| YuNet int8 @640 | 24.4 | 3.3 | 0.64 | 5.1 | 0.75 |
| YuNet OpenCV @640 | 8.2 | 3.3 | 0.63 | 5.0 | 0.76 |
| SCRFD-500M @640 | 7.8 | 3.1 | 0.66 | 4.0 | 0.76 |
| SCRFD-500M @native | 25.4 | 5.2 | 0.89 | 5.8 | 0.88 |
| BlazeFace short | 3.0 | 0.0 | 0.00 | 0.7 | 0.15 |
| BlazeFace full | 6.0 | 0.3 | 0.03 | 2.0 | 0.50 |
| SCRFD-10G @640 (ResNet-class) | 46.6 | 3.9 | 0.79 | 5.7 | 0.81 |

*A* = street footage with small faces (900 frames), *B* = selfie advert with larger faces (750 frames),
1280×720. Reference detector: SCRFD-10G at native resolution (5.5 / 5.8 faces per frame).
**F1 vs ref is a proxy, not accuracy**: there are no human labels for these videos and the reference favours the SCRFD rows.
Each of the 9 detector setups was also run with IoU, SORT and ByteTrack (27 combinations per video; table in the PDF).

## Result

* **Default: YuNet + ByteTrack on ONNX Runtime.** 5.1 ms per frame at 640 px, 0.2 MB, MIT licence.
  Resolution matters more than model size: at full 720p YuNet finds 73% of the
  reference's faces under 24 px on A, against 25% at 640 px (22 ms).
* **BlazeFace** misses distant faces (0.0 / 0.3 faces per frame on A for short / full range).

## The suggested ResNet, ArcFace and U-Net models

* **ResNet:** a backbone, not a detector. The closest real detector (SCRFD-10G) takes 47 ms per frame, over the 33 ms budget
  for 30 fps, and agrees less with the reference than a small model run at native resolution.
* **ArcFace:** a recognition model; it cannot find faces. On the faces in these videos the ResNet-50 version takes
  914 ms per frame (27× the budget); MobileFaceNet 79 ms.
  It is viable once per track (Phase 3), not once per frame.
* **U-Net:** not run. It outputs masks, not face boxes, and no pretrained face model exists to benchmark.

## Reproduce

```bash
pip install -e ".[mediapipe,bench,youtube,report]"
python scripts/download_models.py
python scripts/download_suggested_models.py        # 305 MB, non-commercial InsightFace weights
python scripts/bench_youtube.py URL_A URL_B         # about 15 minutes; run on an idle, plugged-in machine
python scripts/build_benchmark_report.py            # rewrites this file and docs/BENCHMARK_REPORT.*
```

Limits: two 30-second clips, one machine, no phone timings. Measured and proxy numbers are kept apart in the PDF.
