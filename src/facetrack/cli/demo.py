"""Desktop demo: uploaded video (default) or live camera through the same pipeline.

    facetrack-demo --video path/to/clip.mp4
    facetrack-demo --camera 0
    facetrack-demo --video clip.mp4 --no-display --export out/clip.jsonl   # headless batch

Keys while the window is focused:  space = play/pause   n = step one frame (paused)
                                   r = restart video    q / Esc = stop
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import yaml

from facetrack.detectors import DETECTOR_NAMES, create_detector
from facetrack.export import JsonlExporter, MotExporter
from facetrack.pipeline import FacePipeline
from facetrack.sources import CameraSource, SourceError, VideoFileSource, list_cameras
from facetrack.tracking import TRACKERS, TrackerConfig, create_tracker
from facetrack.viz import draw

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="facetrack-demo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", type=str, help="video file (mp4/avi/mov/mkv/webm)")
    src.add_argument("--url", help="online video (e.g. YouTube) played as a stream, not saved; needs the [youtube] extra")
    src.add_argument("--camera", type=int, help="camera index")
    src.add_argument("--list-cameras", action="store_true")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--detector", choices=DETECTOR_NAMES)
    p.add_argument("--tracker", choices=tuple(TRACKERS))
    p.add_argument("--input-size", type=int, help="detector input long side in px; 0 = native frame size")
    p.add_argument("--score-threshold", type=float)
    p.add_argument("--detect-every", type=int)
    p.add_argument("--camera-width", type=int)
    p.add_argument("--camera-height", type=int)
    p.add_argument("--export", type=Path, help="write per-frame JSONL (detections, tracks, events, timings)")
    p.add_argument("--export-mot", type=Path, help="write MOTChallenge-format tracks")
    p.add_argument("--save-video", type=Path, help="write the annotated video")
    p.add_argument("--no-display", action="store_true", help="headless: process as fast as possible")
    p.add_argument("--realtime", action="store_true", help="pace video playback to its timestamps")
    p.add_argument("--show-detections", action="store_true", help="also draw raw detector boxes")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if args.list_cameras:
        print("cameras:", list_cameras() or "none found")
        return 0
    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    dcfg, tcfg, pcfg = cfg.get("detector", {}), dict(cfg.get("tracker", {})), cfg.get("pipeline", {})

    det_name = args.detector or dcfg.get("name", "yunet")
    trk_name = args.tracker or tcfg.pop("name", "bytetrack")
    tcfg.pop("name", None)
    try:
        detector = create_detector(
            det_name,
            score_threshold=args.score_threshold if args.score_threshold is not None else dcfg.get("score_threshold"),
            nms_threshold=dcfg.get("nms_threshold"),
            input_long_side=args.input_size if args.input_size is not None else dcfg.get("input_long_side"),
            num_threads=dcfg.get("num_threads"),
        )
    except (FileNotFoundError, ImportError, ValueError) as e:
        print(f"error: could not initialise detector {det_name!r}: {e}", file=sys.stderr)
        return 2
    tracker = create_tracker(trk_name, TrackerConfig.from_dict(tcfg))
    pipeline = FacePipeline(detector, tracker, detect_every=args.detect_every or pcfg.get("detect_every", 1))

    try:
        if args.camera is not None:
            source = CameraSource(args.camera, args.camera_width, args.camera_height)
        else:
            if args.url:
                from facetrack.download import resolve_stream_url

                args.video = resolve_stream_url(args.url)
                args.realtime = True  # a stream plays at its own speed, not as fast as we can decode
            if args.video is None:
                print("error: pass --video PATH, --url URL or --camera INDEX", file=sys.stderr)
                return 2
            source = VideoFileSource(args.video)
    except (SourceError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    info = source.info
    print(f"source: {info.kind} {info.uri} {info.width}x{info.height} fps={info.fps} frames={info.frame_count}")
    print(f"detector: {det_name}  tracker: {trk_name}  detect_every: {pipeline.detect_every}")
    header = {"source": info.to_dict(), "detector": detector.describe(), "tracker": {"name": trk_name, **vars(tracker.cfg)},
              "detect_every": pipeline.detect_every}
    jsonl = JsonlExporter(args.export, header) if args.export else None
    mot = MotExporter(args.export_mot) if args.export_mot else None
    writer = None

    window = "facetrack" if not args.no_display else None
    if window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    paused, step = False, False
    proc_times: deque[float] = deque(maxlen=30)
    det_times: deque[float] = deque(maxlen=30)
    wall_start, first_ts = time.perf_counter(), None
    last = None
    n = 0
    try:
        while True:
            if not paused or step:
                try:
                    frame = source.read()
                except SourceError as e:
                    print(f"source failed: {e}", file=sys.stderr)
                    break
                if frame is None:
                    break
                step = False
                if args.realtime and args.video:
                    first_ts = frame.timestamp_ms if first_ts is None else first_ts
                    delay = (frame.timestamp_ms - first_ts) / 1000 - (time.perf_counter() - wall_start)
                    if delay > 0:
                        time.sleep(delay)
                res = pipeline.process(frame)
                n += 1
                proc_times.append(res.total_ms)
                if res.detector_timings:
                    det_times.append(res.detector_timings.total_ms)
                if jsonl:
                    jsonl.add(res)
                if mot:
                    mot.add(res)
                for e in res.events:
                    if e.type.value != "track_updated" and args.verbose:
                        print(f"[{e.timestamp_ms / 1000:8.2f}s] {e.type.value:16s} id={e.track_id}")
                last = res
                if window or args.save_video:
                    avg = sum(proc_times) / len(proc_times)
                    hud = {
                        "source": f"{info.kind} {info.width}x{info.height}",
                        "detector": f"{det_name} ({(sum(det_times) / len(det_times)) if det_times else 0:.1f} ms)",
                        "tracker": trk_name,
                        "pipeline": f"{avg:.1f} ms/frame = {1000 / avg if avg else 0:.0f} FPS capacity",
                        "time": f"{frame.timestamp_ms / 1000:.2f}s  frame {frame.frame_id}",
                        "tracks": str(len(res.tracks.visible())),
                    }
                    vis = draw(res, hud, show_raw=args.show_detections)
                    if args.save_video:
                        if writer is None:
                            args.save_video.parent.mkdir(parents=True, exist_ok=True)
                            writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                                     info.fps or 30, (vis.shape[1], vis.shape[0]))
                        writer.write(vis)
                    if window:
                        cv2.imshow(window, vis)
            if window:
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("n"):
                    step = True
                elif key == ord("r") and isinstance(source, VideoFileSource):
                    source.restart()
                    pipeline.reset()
                    wall_start, first_ts = time.perf_counter(), None
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        end_ts = last.frame.timestamp_ms if last else 0.0
        final = pipeline.finish(end_ts, last.frame.frame_id if last else -1)
        source.release()
        detector.close()
        if writer:
            writer.release()
        if window:
            cv2.destroyAllWindows()
        summary = {"frames": n, "mean_pipeline_ms": sum(proc_times) / len(proc_times) if proc_times else None,
                   "tracks_created": tracker.ids_issued}
        if jsonl:
            jsonl.close(final, summary)
            print(f"wrote {args.export}")
        if mot:
            mot.close()
            print(f"wrote {args.export_mot}")
    print(f"processed {n} frames; {tracker.ids_issued} track IDs issued")
    return 0


if __name__ == "__main__":
    sys.exit(main())
