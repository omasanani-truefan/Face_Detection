"""Multi-object tracking metrics: CLEAR-MOT, IDF1, HOTA, plus lifecycle timings.

The three metric families follow TrackEval's reference implementations
(github.com/JonathonLuiten/TrackEval, ``metrics/clear.py``, ``identity.py``, ``hota.py``)
step for step; TrackEval itself is not a dependency because its released code uses
NumPy aliases removed in NumPy 1.24. MOTA / IDF1 are cross-checked against
``py-motmetrics`` in the tests.

Input format, per frame: ``gt[f] = [(gt_id, xyxy, ignore)]`` and
``pred[f] = [(track_id, xyxy)]``. Ignore boxes (heavily occluded / mostly out of frame
faces) follow MOTChallenge practice: predictions matched to them are dropped, and they
never count as misses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from facetrack.tracking.matching import iou_matrix

EPS = np.finfo(float).eps
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)

GtFrame = list[tuple[int, np.ndarray, bool]]
PredFrame = list[tuple[int, np.ndarray]]


@dataclass
class SequenceData:
    """Dense per-frame id/similarity arrays after ignore filtering, with ids re-indexed."""

    gt_ids: list[np.ndarray]
    pred_ids: list[np.ndarray]
    sims: list[np.ndarray]
    num_gt_ids: int
    num_pred_ids: int
    gt_id_map: dict[int, int]
    pred_id_map: dict[int, int]
    num_gt_dets: int = 0
    num_pred_dets: int = 0
    frames: list[int] = field(default_factory=list)


def prepare(gt: dict[int, GtFrame], pred: dict[int, PredFrame], frames: list[int], ignore_iou: float = 0.5) -> SequenceData:
    gt_map: dict[int, int] = {}
    pr_map: dict[int, int] = {}
    gid_l, pid_l, sim_l = [], [], []
    for f in frames:
        g = gt.get(f, [])
        p = pred.get(f, [])
        keep_g = [x for x in g if not x[2]]
        ign_g = [x for x in g if x[2]]
        pboxes = np.array([b for _, b in p], dtype=float).reshape(-1, 4)
        drop = set()
        if ign_g and len(p):
            # A prediction is dropped only if it matches an ignore box *and* no real face better.
            iboxes = np.array([b for _, b, _ in ign_g], dtype=float).reshape(-1, 4)
            gboxes = np.array([b for _, b, _ in keep_g], dtype=float).reshape(-1, 4)
            iou_i = iou_matrix(pboxes, iboxes)
            iou_g = iou_matrix(pboxes, gboxes)
            r, c = linear_sum_assignment(-iou_i)
            for ri, ci in zip(r, c):
                best_real = iou_g[ri].max() if iou_g.shape[1] else 0.0
                if iou_i[ri, ci] >= ignore_iou and iou_i[ri, ci] > best_real:
                    drop.add(ri)
        p = [x for k, x in enumerate(p) if k not in drop]
        gids = np.array([gt_map.setdefault(i, len(gt_map)) for i, _, _ in keep_g], dtype=int)
        pids = np.array([pr_map.setdefault(i, len(pr_map)) for i, _ in p], dtype=int)
        gb = np.array([b for _, b, _ in keep_g], dtype=float).reshape(-1, 4)
        pb = np.array([b for _, b in p], dtype=float).reshape(-1, 4)
        gid_l.append(gids)
        pid_l.append(pids)
        sim_l.append(iou_matrix(gb, pb))
    return SequenceData(gid_l, pid_l, sim_l, len(gt_map), len(pr_map), gt_map, pr_map,
                        sum(len(x) for x in gid_l), sum(len(x) for x in pid_l), list(frames))


def clear_mot(d: SequenceData, threshold: float = 0.5) -> dict[str, float]:
    tp = fn = fp = idsw = 0
    motp_sum = 0.0
    prev_id = np.full(d.num_gt_ids, np.nan)  # last tracker id ever matched (for IDSW)
    prev_step = np.full(d.num_gt_ids, np.nan)  # tracker id matched on the previous frame
    gt_count = np.zeros(d.num_gt_ids)
    matched_count = np.zeros(d.num_gt_ids)
    frag_count = np.zeros(d.num_gt_ids)
    for gids, pids, sim in zip(d.gt_ids, d.pred_ids, d.sims):
        if len(gids) == 0:
            fp += len(pids)
            prev_step[:] = np.nan
            continue
        if len(pids) == 0:
            fn += len(gids)
            gt_count[gids] += 1
            prev_step[:] = np.nan
            continue
        score = (pids[None, :] == prev_step[gids][:, None]).astype(float)
        score = 1000 * score + sim
        score[sim < threshold - EPS] = 0
        r, c = linear_sum_assignment(-score)
        ok = score[r, c] > EPS
        r, c = r[ok], c[ok]
        mg, mp = gids[r], pids[c]
        prev_m = prev_id[mg]
        idsw += int(np.sum(~np.isnan(prev_m) & (mp != prev_m)))
        gt_count[gids] += 1
        matched_count[mg] += 1
        not_prev = np.isnan(prev_step)
        prev_id[mg] = mp
        prev_step[:] = np.nan
        prev_step[mg] = mp
        frag_count += not_prev & ~np.isnan(prev_step)
        n = len(r)
        tp += n
        fn += len(gids) - n
        fp += len(pids) - n
        motp_sum += float(sim[r, c].sum())
    ratio = matched_count[gt_count > 0] / gt_count[gt_count > 0]
    num_gt = max(1, d.num_gt_dets)
    return {
        "MOTA": (tp - fp - idsw) / num_gt,
        "MOTP": motp_sum / max(1, tp),
        "IDSW": idsw,
        "Frag": float(np.sum(frag_count[frag_count > 0] - 1)),
        "TP": tp, "FP": fp, "FN": fn,
        "MT": int(np.sum(ratio > 0.8)), "PT": int(np.sum((ratio >= 0.2) & (ratio <= 0.8))), "ML": int(np.sum(ratio < 0.2)),
        "num_gt_tracks": int(np.sum(gt_count > 0)),
    }


def identity(d: SequenceData, threshold: float = 0.5) -> dict[str, float]:
    ng, nt = d.num_gt_ids, d.num_pred_ids
    if ng == 0 or nt == 0:
        idfn = float(d.num_gt_dets)
        idfp = float(d.num_pred_dets)
        return {"IDF1": 0.0, "IDP": 0.0, "IDR": 0.0, "IDTP": 0.0, "IDFN": idfn, "IDFP": idfp}
    potential = np.zeros((ng, nt))
    gt_count = np.zeros(ng)
    pr_count = np.zeros(nt)
    for gids, pids, sim in zip(d.gt_ids, d.pred_ids, d.sims):
        mr, mc = np.nonzero(sim >= threshold - EPS)
        potential[gids[mr], pids[mc]] += 1
        gt_count[gids] += 1
        pr_count[pids] += 1
    fp_mat = np.zeros((ng + nt, ng + nt))
    fn_mat = np.zeros((ng + nt, ng + nt))
    fp_mat[ng:, :nt] = 1e10
    fn_mat[:ng, nt:] = 1e10
    for g in range(ng):
        fn_mat[g, :nt] = gt_count[g]
        fn_mat[g, nt + g] = gt_count[g]
    for t in range(nt):
        fp_mat[:ng, t] = pr_count[t]
        fp_mat[t + ng, t] = pr_count[t]
    fn_mat[:ng, :nt] -= potential
    fp_mat[:ng, :nt] -= potential
    r, c = linear_sum_assignment(fn_mat + fp_mat)
    idfn = float(fn_mat[r, c].sum())
    idfp = float(fp_mat[r, c].sum())
    idtp = float(gt_count.sum() - idfn)
    return {"IDF1": idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn), "IDP": idtp / max(1.0, idtp + idfp),
            "IDR": idtp / max(1.0, idtp + idfn), "IDTP": idtp, "IDFN": idfn, "IDFP": idfp}


def hota(d: SequenceData) -> dict[str, float]:
    na = len(HOTA_ALPHAS)
    ng, nt = d.num_gt_ids, d.num_pred_ids
    if d.num_pred_dets == 0 or d.num_gt_dets == 0:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "LocA": 1.0}
    potential = np.zeros((ng, nt))
    gt_count = np.zeros((ng, 1))
    pr_count = np.zeros((1, nt))
    for gids, pids, sim in zip(d.gt_ids, d.pred_ids, d.sims):
        if len(gids) and len(pids):
            denom = sim.sum(0)[None, :] + sim.sum(1)[:, None] - sim
            s = np.zeros_like(sim)
            m = denom > EPS
            s[m] = sim[m] / denom[m]
            potential[gids[:, None], pids[None, :]] += s
        gt_count[gids] += 1
        pr_count[0, pids] += 1
    global_score = potential / np.maximum(EPS, gt_count + pr_count - potential)
    matches = [np.zeros((ng, nt)) for _ in range(na)]
    TP, FN, FP, LOC = np.zeros(na), np.zeros(na), np.zeros(na), np.zeros(na)
    for gids, pids, sim in zip(d.gt_ids, d.pred_ids, d.sims):
        if len(gids) == 0:
            FP += len(pids)
            continue
        if len(pids) == 0:
            FN += len(gids)
            continue
        score = global_score[gids[:, None], pids[None, :]] * sim
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(HOTA_ALPHAS):
            ok = sim[r, c] >= alpha - EPS
            ar, ac = r[ok], c[ok]
            n = len(ar)
            TP[a] += n
            FN[a] += len(gids) - n
            FP[a] += len(pids) - n
            if n:
                LOC[a] += sim[ar, ac].sum()
                matches[a][gids[ar], pids[ac]] += 1
    AssA = np.zeros(na)
    for a in range(na):
        mc = matches[a]
        ass = mc / np.maximum(1, gt_count + pr_count - mc)
        AssA[a] = np.sum(mc * ass) / max(1, TP[a])
    DetA = TP / np.maximum(1, TP + FN + FP)
    H = np.sqrt(DetA * AssA)
    LocA = np.maximum(1e-10, LOC) / np.maximum(1e-10, TP)
    return {"HOTA": float(H.mean()), "DetA": float(DetA.mean()), "AssA": float(AssA.mean()), "LocA": float(LocA.mean()),
            "DetRe": float((TP / np.maximum(1, TP + FN)).mean()), "DetPr": float((TP / np.maximum(1, TP + FP)).mean())}


def lifecycle(gt: dict[int, GtFrame], pred: dict[int, PredFrame], frames: list[int], ts_ms: dict[int, float],
              track_end_ms: dict[int, float], threshold: float = 0.5) -> dict[str, float | None]:
    """Face-tracking-specific timings and track counts.

    * confirmation delay: GT appearance start -> first frame a confirmed track overlaps it.
    * termination delay: GT appearance end -> ``track_ended`` of the track that covered it
      most (only for appearances that end before the sequence does).
    * missed tracks: GT appearances never covered; false tracks: track IDs that never
      cover any GT face.
    """
    first_gt: dict[int, int] = {}
    last_gt: dict[int, int] = {}
    first_hit: dict[int, int] = {}
    cover: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    pred_hit: set[int] = set()
    pred_all: set[int] = set()
    for f in frames:
        g = [x for x in gt.get(f, []) if not x[2]]
        p = pred.get(f, [])
        for gid, _, _ in g:
            first_gt.setdefault(gid, f)
            last_gt[gid] = f
        pred_all.update(pid for pid, _ in p)
        if not g or not p:
            continue
        iou = iou_matrix(np.array([b for _, b, _ in g]).reshape(-1, 4), np.array([b for _, b in p]).reshape(-1, 4))
        r, c = linear_sum_assignment(-iou)
        for ri, ci in zip(r, c):
            if iou[ri, ci] >= threshold:
                gid, pid = g[ri][0], p[ci][0]
                first_hit.setdefault(gid, f)
                cover[gid][pid] += 1
                pred_hit.add(pid)
    confirm = [ts_ms[first_hit[g]] - ts_ms[first_gt[g]] for g in first_gt if g in first_hit]
    seq_end = frames[-1]
    term = []
    for g, lf in last_gt.items():
        if lf >= seq_end or g not in cover:
            continue
        pid = max(cover[g].items(), key=lambda kv: kv[1])[0]
        if pid in track_end_ms:
            term.append(track_end_ms[pid] - ts_ms[lf])
    return {
        "confirmation_delay_ms_mean": float(np.mean(confirm)) if confirm else None,
        "confirmation_delay_ms_p95": float(np.percentile(confirm, 95)) if confirm else None,
        "termination_delay_ms_mean": float(np.mean(term)) if term else None,
        "missed_tracks": sum(1 for g in first_gt if g not in first_hit),
        "false_tracks": len(pred_all - pred_hit),
        "gt_tracks": len(first_gt),
        "pred_tracks": len(pred_all),
    }


def evaluate_sequence(gt: dict[int, GtFrame], pred: dict[int, PredFrame], frames: list[int],
                      ts_ms: dict[int, float] | None = None, track_end_ms: dict[int, float] | None = None) -> dict:
    d = prepare(gt, pred, frames)
    out = {**clear_mot(d), **identity(d), **hota(d)}
    if ts_ms is not None:
        out.update(lifecycle(gt, pred, frames, ts_ms, track_end_ms or {}))
    return out
