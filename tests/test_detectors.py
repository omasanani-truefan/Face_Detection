"""Detector backends: contract, decoder parity, resolutions. Skips if models are absent."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from facetrack.detectors import DETECTOR_NAMES, create_detector
from facetrack.models import MODELS, model_path
from facetrack.tracking.matching import iou_matrix

ROOT = Path(__file__).resolve().parents[1]
WIDER_DEV = ROOT / "data" / "widerface" / "dev.json"


def available(name: str) -> bool:
    try:
        create_detector(name).close()
        return True
    except (FileNotFoundError, ImportError):
        return False


pytestmark = pytest.mark.skipif(not all(model_path(k).exists() for k in MODELS), reason="run scripts/download_models.py")


def pick_image(min_rel_height: float):
    """The WIDER dev image with the most faces, all at least ``min_rel_height`` of the
    frame height, so every face is inside the detector's operating range."""
    if not WIDER_DEV.exists():
        pytest.skip("run scripts/prepare_widerface.py")
    gt = json.loads(WIDER_DEV.read_text())
    root = ROOT / gt["info"]["image_root"]
    heights = {i["id"]: i["height"] for i in gt["images"]}
    per_img: dict[int, list] = {}
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            per_img.setdefault(a["image_id"], []).append(a["bbox"])
    ok = {k: v for k, v in per_img.items() if len(v) >= 2 and min(b[3] for b in v) >= min_rel_height * heights[k]}
    img_id = max(ok, key=lambda k: (len(ok[k]), -k))
    im = next(i for i in gt["images"] if i["id"] == img_id)
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in ok[img_id]], float)
    return cv2.imread(str(root / im["file_name"])), boxes


@pytest.fixture(scope="module")
def face_image():
    return pick_image(0.15)


# Measured operating range (docs/REPORT.md): BlazeFace short-range needs faces at
# roughly >= 20-35% of frame height; everything else works from ~10%.
MIN_REL_HEIGHT = {"mediapipe_short": 0.30}


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_blank_frame_has_no_faces(name):
    with create_detector(name) as d:
        dets, t = d.detect(np.zeros((480, 640, 3), np.uint8))
    assert dets == [] and t.total_ms >= 0


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_finds_faces_in_operating_range_and_returns_contract(name):
    img, gt = pick_image(MIN_REL_HEIGHT.get(name, 0.15))
    with create_detector(name, score_threshold=0.3) as d:
        dets, _ = d.detect(img)
        info = d.describe()
    assert info["model_sha256"] and info["runtime"]
    h, w = img.shape[:2]
    for det in dets:
        x, y, bw, bh = det.bbox
        assert 0 <= x and 0 <= y and x + bw <= w + 1e-6 and y + bh <= h + 1e-6
        assert 0 <= det.confidence <= 1
        assert det.keypoints is not None and len(det.keypoints) == len(d.keypoint_names)
    found = iou_matrix(gt, np.array([d.xyxy for d in dets]).reshape(-1, 4)).max(1) >= 0.3 if dets else np.zeros(len(gt), bool)
    assert found.mean() >= 0.6, found


def test_yunet_ort_decoder_matches_opencv():
    """Our NumPy decoder over ONNX Runtime must reproduce cv2.FaceDetectorYN on a crowded
    image. Exact equality is not expected: OpenCV runs NMS on integer (Rect2i) boxes and
    we use floats, so suppression between tiny overlapping faces can differ; detections
    within 0.02 of the threshold are also excluded (a 1e-7 score difference can flip them)."""
    if not WIDER_DEV.exists():
        pytest.skip("run scripts/prepare_widerface.py")
    gt = json.loads(WIDER_DEV.read_text())
    root = ROOT / gt["info"]["image_root"]
    counts: dict[int, int] = {}
    for a in gt["annotations"]:
        counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
    im = next(i for i in gt["images"] if i["id"] == max(counts, key=counts.get))
    img = cv2.imread(str(root / im["file_name"]))
    with create_detector("yunet", score_threshold=0.5, input_long_side=640) as a, \
         create_detector("yunet_cv", score_threshold=0.5, input_long_side=640) as b:
        da = [d for d in a.detect(img)[0] if d.confidence >= 0.52]
        db = [d for d in b.detect(img)[0] if d.confidence >= 0.52]
    assert len(da) > 20 and abs(len(da) - len(db)) <= 0.02 * len(db)
    iou = iou_matrix(np.array([d.xyxy for d in da]), np.array([d.xyxy for d in db]))
    assert np.mean(iou.max(1) > 0.99) >= 0.98 and np.mean(iou.max(0) > 0.99) >= 0.98


@pytest.mark.parametrize("name", ["yunet", "scrfd_500m_kps"])
@pytest.mark.parametrize("shape", [(240, 320), (480, 640), (720, 1280), (1080, 1920), (1280, 720)])
def test_any_input_resolution_maps_back_to_frame(name, shape, face_image):
    img, _ = face_image
    frame = cv2.resize(img, (shape[1], shape[0]))
    with create_detector(name, score_threshold=0.5) as d:
        dets, _ = d.detect(frame)
    assert dets, "expected faces after resize"
    for det in dets:
        x, y, w, h = det.bbox
        assert x + w <= shape[1] + 1e-6 and y + h <= shape[0] + 1e-6


def test_missing_model_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_models"):
        create_detector("yunet", model_path=str(tmp_path / "missing.onnx"))


def test_unknown_detector_name():
    with pytest.raises(KeyError):
        create_detector("resnet")
