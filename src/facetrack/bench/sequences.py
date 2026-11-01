"""Synthetic-motion tracking sequences built from real WIDER FACE faces, with exact
per-frame ground-truth identities.

Why synthetic: identity-annotated face video recorded with consent does not exist for
this project yet, and the tracking benchmark needs exact identities for every frame of
every scenario in the spec (entering, leaving, occlusion, crossing, head turns, missed
detections, re-entry). Two generators:

* **actors** -- face "sprites" cropped from WIDER (real faces, real photo context) are
  composited onto face-free WIDER backgrounds and moved along scripted paths. Exact
  boxes, occlusion and identity are known by construction. Limitation: compositing seams
  and the absence of real 3-D head rotation (a head turn is approximated by horizontal
  foreshortening).
* **camera** -- a virtual camera pans / zooms over a real WIDER crowd photo, so real
  faces in their real context move, shrink and cross the frame edges (the mobile /
  hand-held case). Limitation: no independent subject motion.

Ground truth per face: ``track_gt_id`` (one per continuous appearance -- a person who
leaves and returns gets a new one, matching the Phase 1-2 contract), ``person_id``
(stable across appearances, for Phase 3), ``visibility``, and ``ignore`` when the face
is < 50% visible, foreshortened past profile, or below 10 px.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MIN_VISIBLE = 0.5
MIN_SQUEEZE = 0.45  # width factor below which a "turned" head is treated as not a visible face
MIN_FACE_PX = 10


# --------------------------------------------------------------------------- assets
@dataclass
class Actor:
    image: np.ndarray
    alpha: np.ndarray  # float32 0..1, same HxW as image
    face: tuple[float, float, float, float]  # x1, y1, x2, y2 in crop coordinates
    person_id: str
    source: str


def _overlaps(box, boxes, margin=0.0) -> bool:
    x1, y1, x2, y2 = box
    for bx, by, bw, bh in boxes:
        if bx - margin < x2 and bx + bw + margin > x1 and by - margin < y2 and by + bh + margin > y1:
            return True
    return False


def load_actors(coco: dict, img_root: Path, n: int, rng: np.random.Generator, min_face_h: int = 110) -> list[Actor]:
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    images = {i["id"]: i for i in coco["images"]}
    cands = []
    for a in coco["annotations"]:
        at = a["attributes"]
        x, y, w, h = a["bbox"]
        if a["iscrowd"] or h < min_face_h or at["blur"] or at["occlusion"] or at["pose"] or at["illumination"]:
            continue
        im = images[a["image_id"]]
        cx, cy = x + w / 2, y + h * 0.55
        cw, ch = w * 1.8, h * 2.0
        crop = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        if crop[0] < 0 or crop[1] < 0 or crop[2] > im["width"] or crop[3] > im["height"]:
            continue  # a border-clipped crop would put the face off-centre in its mask
        others = [b["bbox"] for b in by_img[a["image_id"]] if b["id"] != a["id"]]
        if _overlaps(crop, others):
            continue  # a second face inside the sprite would be an unlabelled face
        cands.append((a, crop))
    rng.shuffle(cands)
    actors = []
    for a, (x1, y1, x2, y2) in cands[:n]:
        img = cv2.imread(str(img_root / images[a["image_id"]]["file_name"]))
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        crop = img[y1:y2, x1:x2].copy()
        h, w = crop.shape[:2]
        fx, fy, fw, fh = a["bbox"]
        alpha = np.zeros((h, w), np.float32)
        # Head-sized ellipse centred on the face: keeps hair and jaw line, drops most of
        # the source photo's background around the head.
        fcx, fcy = fx - x1 + fw / 2, fy - y1 + fh / 2
        cv2.ellipse(alpha, (int(fcx), int(fcy - 0.05 * fh)), (int(fw * 0.68), int(fh * 0.85)), 0, 0, 360, 1.0, -1)
        k = max(3, int(min(h, w) * 0.08) | 1)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        actors.append(Actor(crop, alpha, (fx - x1, fy - y1, fx - x1 + fw, fy - y1 + fh),
                            f"wider-{a['id']}", images[a["image_id"]]["file_name"]))
    if len(actors) < n:
        raise RuntimeError(f"only {len(actors)} usable actor faces (wanted {n})")
    return actors


def load_backgrounds(coco: dict, img_root: Path, n: int, size: tuple[int, int], rng: np.random.Generator) -> list[np.ndarray]:
    """Face-free windows of WIDER images (no GT box, valid or not, within 10 px)."""
    by_img: dict[int, list] = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a["bbox"])
    W, H = size
    out = []
    imgs = list(coco["images"])
    rng.shuffle(imgs)
    for im in imgs:
        boxes = by_img.get(im["id"], [])
        for _ in range(25):
            ww = rng.uniform(0.45, 0.8) * im["width"]
            hh = ww * H / W
            if hh > im["height"]:
                continue
            x0 = rng.uniform(0, im["width"] - ww)
            y0 = rng.uniform(0, im["height"] - hh)
            if not _overlaps((x0, y0, x0 + ww, y0 + hh), boxes, margin=10):
                img = cv2.imread(str(img_root / im["file_name"]))
                win = img[int(y0):int(y0 + hh), int(x0):int(x0 + ww)]
                out.append(cv2.resize(win, (W, H), interpolation=cv2.INTER_AREA))
                break
        if len(out) >= n:
            return out
    raise RuntimeError("not enough face-free background windows")


# --------------------------------------------------------------------------- scripting
@dataclass
class Appearance:
    """One continuous on-screen appearance of an actor (one GT track)."""

    actor: int
    keys: list[tuple[float, float, float, float]]  # (t_sec, cx, cy, face_h) in frame px
    z: int = 0
    squeeze: list[tuple[float, float]] = field(default_factory=list)  # (t_sec, width factor)
    blur: list[tuple[float, float]] = field(default_factory=list)  # (t_start, t_end) motion blur bursts
    sway: float = 0.03

    def state(self, t: float, seed: int) -> tuple[float, float, float, float, float] | None:
        ts = [k[0] for k in self.keys]
        if t < ts[0] or t > ts[-1]:
            return None
        cx = np.interp(t, ts, [k[1] for k in self.keys])
        cy = np.interp(t, ts, [k[2] for k in self.keys])
        fh = np.interp(t, ts, [k[3] for k in self.keys])
        ph = (seed % 97) / 97 * 2 * np.pi
        cx += self.sway * fh * np.sin(2 * np.pi * 0.45 * t + ph)
        cy += 0.5 * self.sway * fh * np.sin(2 * np.pi * 0.31 * t + 2 * ph)
        sq = float(np.interp(t, [s[0] for s in self.squeeze], [s[1] for s in self.squeeze])) if self.squeeze else 1.0
        bl = next((1.0 for a, b in self.blur if a <= t <= b), 0.0)
        return cx, cy, fh, sq, bl


@dataclass
class Occluder:
    keys: list[tuple[float, float]]  # (t_sec, x_center) -- a vertical pole sweeping horizontally
    width: float
    color: tuple[int, int, int] = (70, 80, 90)


@dataclass
class Scenario:
    name: str
    tags: list[str]
    duration: float
    size: tuple[int, int]
    appearances: list[Appearance]
    occluders: list[Occluder] = field(default_factory=list)
    n_actors: int = 0


def scenarios(W: int = 640, H: int = 480) -> list[Scenario]:
    """The tracking situations from the spec, scripted at 640x480 (far ones at 1280x720)."""
    F = 150  # close-range face height (px) at 480p: ~1 m from a laptop webcam
    S = []
    S.append(Scenario("single_continuous", ["situation1", "close"], 6, (W, H),
                      [Appearance(0, [(0, 320, 220, F), (3, 300, 230, F * 1.05), (6, 340, 215, F)], sway=0.06)]))
    S.append(Scenario("multi_enter", ["situation2", "close", "enter"], 7, (W, H), [
        Appearance(0, [(0.0, -60, 200, 110), (1.2, 140, 210, 110), (7, 150, 215, 110)]),
        Appearance(1, [(0.0, W + 60, 230, 115), (1.2, 480, 225, 115), (7, 470, 230, 115)]),
        Appearance(2, [(0.0, 320, H + 70, 100), (1.2, 320, 300, 100), (7, 315, 290, 100)]),
    ]))
    S.append(Scenario("one_leaves", ["situation3", "leave"], 7, (W, H), [
        Appearance(0, [(0, 180, 220, 130), (7, 190, 225, 130)]),
        Appearance(1, [(0, 460, 220, 125), (3.0, 470, 220, 125), (4.2, W + 90, 230, 125)]),
    ]))
    S.append(Scenario("temporary_occlusion", ["situation4", "occlusion"], 7, (W, H), [
        Appearance(0, [(0, 190, 230, 130), (7, 200, 230, 130)]),
        Appearance(1, [(0, 450, 220, 125), (7, 440, 225, 125)]),
    ], occluders=[Occluder([(1.5, -80), (5.5, W + 80)], 150)]))
    S.append(Scenario("crossing", ["situation5", "crossing"], 6, (W, H), [
        Appearance(0, [(0, 90, 220, 120), (6, 560, 235, 128)], z=1),
        Appearance(1, [(0, 560, 255, 105), (6, 80, 245, 100)], z=0),
    ]))
    S.append(Scenario("turn_away", ["situation6", "pose"], 6, (W, H), [
        Appearance(0, [(0, 220, 225, 135), (6, 230, 225, 135)], squeeze=[(0, 1), (2.0, 1), (2.5, 0.12), (3.2, 0.12), (3.7, 1), (6, 1)]),
        Appearance(1, [(0, 470, 215, 120), (6, 460, 220, 120)]),
    ]))
    S.append(Scenario("detector_miss_blur", ["situation7", "motion_blur"], 6, (W, H), [
        Appearance(0, [(0, 160, 230, 125), (6, 480, 225, 125)], blur=[(1.5, 1.7), (3.2, 3.5)]),
        Appearance(1, [(0, 470, 210, 115), (6, 170, 220, 115)], blur=[(4.0, 4.25)], z=1),
    ]))
    S.append(Scenario("leave_permanent", ["situation8", "leave"], 7, (W, H), [
        Appearance(0, [(0, 200, 225, 130), (2.0, 210, 225, 130), (3.0, -110, 240, 130)]),
        Appearance(1, [(0, 470, 215, 120), (7, 460, 220, 120)]),
    ]))
    S.append(Scenario("reentry", ["situation9", "reentry"], 9, (W, H), [
        Appearance(0, [(0, 220, 225, 130), (1.8, 230, 225, 130), (2.6, -110, 230, 130)]),
        Appearance(0, [(5.4, -110, 230, 130), (6.2, 200, 225, 130), (9, 210, 225, 130)]),  # same person, new GT track
        Appearance(1, [(0, 470, 215, 120), (9, 460, 220, 120)]),
    ]))
    S.append(Scenario("approach", ["scale", "far_to_near"], 6, (W, H), [
        Appearance(0, [(0, 330, 180, 22), (5, 320, 240, 220), (6, 320, 245, 230)], sway=0.02),
    ]))
    FW, FH = 1280, 720
    S.append(Scenario("far_group", ["far", "multi", "enter", "leave"], 8, (FW, FH), [
        Appearance(0, [(0, 300, 300, 26), (8, 380, 305, 24)], sway=0.05),
        Appearance(1, [(0, 620, 280, 22), (8, 600, 285, 23)], sway=0.05),
        Appearance(2, [(0, 900, 320, 30), (8, 940, 310, 28)], sway=0.05),
        Appearance(3, [(1.0, -30, 360, 20), (8, 520, 350, 19)], sway=0.03),
        Appearance(4, [(0, 1100, 260, 18), (3.5, 1320, 265, 18)], sway=0.03),
    ]))
    for s in S:
        s.n_actors = max(a.actor for a in s.appearances) + 1
    return S


# --------------------------------------------------------------------------- rendering
def _paste(frame: np.ndarray, sprite: np.ndarray, alpha: np.ndarray, x0: int, y0: int, cover: np.ndarray) -> None:
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    fx1, fy1, fx2, fy2 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    sx1, sy1 = fx1 - x0, fy1 - y0
    s = sprite[sy1:sy1 + fy2 - fy1, sx1:sx1 + fx2 - fx1].astype(np.float32)
    a = alpha[sy1:sy1 + fy2 - fy1, sx1:sx1 + fx2 - fx1][..., None]
    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (s * a + roi * (1 - a)).astype(np.uint8)
    cover[fy1:fy2, fx1:fx2] = np.maximum(cover[fy1:fy2, fx1:fx2], a[..., 0])


def _visible_fraction(box, W, H, cover: np.ndarray | None) -> float:
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0
    inside = (cx2 - cx1) * (cy2 - cy1) / area
    if cover is None:
        return inside
    occ = float(cover[int(cy1):int(np.ceil(cy2)), int(cx1):int(np.ceil(cx2))].mean()) if cy2 - cy1 >= 1 and cx2 - cx1 >= 1 else 0.0
    return inside * (1 - occ)


def render_actor_sequence(sc: Scenario, actors: list[Actor], background: np.ndarray, fps: float, seed: int):
    W, H = sc.size
    bg = cv2.resize(background, (W, H), interpolation=cv2.INTER_AREA)
    n = int(round(sc.duration * fps))
    # GT track ids are per appearance; person ids per actor.
    for k in range(n):
        t = k / fps
        frame = bg.copy()
        faces = []
        drawn = []  # (z, face box, cover-before-this-actor?) computed after all pastes
        covers = []
        order = sorted(range(len(sc.appearances)), key=lambda i: sc.appearances[i].z)
        for ai in order:
            ap = sc.appearances[ai]
            st = ap.state(t, seed + ai)
            if st is None:
                continue
            cx, cy, fh, sq, bl = st
            act = actors[ap.actor]
            fx1, fy1, fx2, fy2 = act.face
            scale = fh / (fy2 - fy1)
            sw = max(2, int(round(act.image.shape[1] * scale * sq)))
            shh = max(2, int(round(act.image.shape[0] * scale)))
            sprite = cv2.resize(act.image, (sw, shh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            alpha = cv2.resize(act.alpha, (sw, shh), interpolation=cv2.INTER_LINEAR)
            if bl:
                kl = max(3, int(fh * 0.35) | 1)
                kern = np.zeros((kl, kl), np.float32)
                kern[kl // 2, :] = 1.0 / kl
                sprite = cv2.filter2D(sprite, -1, kern)
            fcx = ((fx1 + fx2) / 2) * scale * sq
            fcy = ((fy1 + fy2) / 2) * scale
            x0, y0 = int(round(cx - fcx)), int(round(cy - fcy))
            box = (x0 + fx1 * scale * sq, y0 + fy1 * scale, x0 + fx2 * scale * sq, y0 + fy2 * scale)
            cover = np.zeros((H, W), np.float32)
            _paste(frame, sprite, alpha, x0, y0, cover)
            covers.append(cover)
            drawn.append((ai, box, sq))
        occ_cover = np.zeros((H, W), np.float32)
        for oc in sc.occluders:
            ts = [p[0] for p in oc.keys]
            if ts[0] <= t <= ts[-1]:
                xc = np.interp(t, ts, [p[1] for p in oc.keys])
                x1, x2 = int(xc - oc.width / 2), int(xc + oc.width / 2)
                cx1, cx2 = max(0, x1), min(W, x2)
                if cx2 > cx1:
                    grad = np.linspace(0.8, 1.2, cx2 - cx1)[None, :, None]
                    frame[:, cx1:cx2] = np.clip(np.array(oc.color, np.float32) * grad, 0, 255).astype(np.uint8)
                    occ_cover[:, cx1:cx2] = 1.0
        # Visibility: covered by any later-drawn (higher z) actor or by an occluder.
        for idx, (ai, box, sq) in enumerate(drawn):
            above = occ_cover.copy()
            for c in covers[idx + 1:]:
                above = np.maximum(above, (c > 0.5).astype(np.float32))
            vis = _visible_fraction(box, W, H, above)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            ignore = vis < MIN_VISIBLE or sq < MIN_SQUEEZE or (y2 - y1) < MIN_FACE_PX
            faces.append({"track_gt_id": ai + 1, "person_id": actors[sc.appearances[ai].actor].person_id,
                          "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                          "visibility": round(vis, 3), "ignore": bool(ignore)})
        yield frame, round(t * 1000, 3), faces


def render_camera_sequence(img: np.ndarray, anns: list[dict], path: list[tuple[float, float, float, float]],
                           duration: float, fps: float, size: tuple[int, int]):
    """``path`` keys are (t_sec, cx, cy, window_width) in source-image pixels."""
    W, H = size
    n = int(round(duration * fps))
    ts = [p[0] for p in path]
    for k in range(n):
        t = k / fps
        cx, cy, ww = (np.interp(t, ts, [p[i] for p in path]) for i in (1, 2, 3))
        s = W / ww
        x0, y0 = cx - ww / 2, cy - (H / s) / 2
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
        frame = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        faces = []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            box = ((bx - x0) * s, (by - y0) * s, (bx + bw - x0) * s, (by + bh - y0) * s)
            vis = _visible_fraction(box, W, H, None)
            if vis <= 0:
                continue
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            ignore = bool(a["iscrowd"] or vis < MIN_VISIBLE or bh * s < MIN_FACE_PX)
            faces.append({"track_gt_id": a["id"], "person_id": f"wider-{a['id']}",
                          "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                          "visibility": round(vis, 3), "ignore": ignore})
        yield frame, round(t * 1000, 3), faces


# --------------------------------------------------------------------------- lighting
def apply_lighting(frame: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    f = frame.astype(np.float32) / 255.0
    H, W = f.shape[:2]
    if kind == "low_light":
        f = np.power(f, 1.6) * 0.35
        f += rng.normal(0, 0.025, f.shape).astype(np.float32)  # sensor noise at high gain
    elif kind == "backlight":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        glow = np.exp(-(((xx - W * 0.5) / (W * 0.45)) ** 2 + ((yy - H * 0.15) / (H * 0.5)) ** 2))
        f = f * 0.45 + glow[..., None] * 0.75  # subjects darkened, bright source behind
    elif kind == "uneven":
        ramp = np.linspace(0.18, 1.25, W, dtype=np.float32)[None, :, None]
        f = f * ramp
    else:
        raise ValueError(kind)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def write_sequence(out_dir: Path, name: str, frames_iter, fps: float, meta: dict, lighting: str | None = None,
                   seed: int = 0, codec: str = "mp4v") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / f"{name}.mp4"
    rng = np.random.default_rng(seed)
    writer = None
    gt_frames = []
    for k, (frame, ts, faces) in enumerate(frames_iter):
        if lighting:
            frame = apply_lighting(frame, lighting, rng)
        if writer is None:
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*codec), fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError(f"cannot open video writer for {video}")
        writer.write(frame)
        gt_frames.append({"frame_id": k, "timestamp_ms": ts, "faces": faces})
    writer.release()
    gt = {**meta, "name": name, "video": video.name, "fps": fps, "lighting": lighting or "normal",
          "num_frames": len(gt_frames), "frames": gt_frames}
    (out_dir / f"{name}.json").write_text(json.dumps(gt))
    return gt
