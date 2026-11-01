from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``a`` (N x 4 xyxy) and ``b`` (M x 4 xyxy)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def normalized_center_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Centre distance divided by the mean diagonal of the two boxes (scale-free)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ca = (a[:, :2] + a[:, 2:]) / 2
    cb = (b[:, :2] + b[:, 2:]) / 2
    d = np.linalg.norm(ca[:, None] - cb[None], axis=2)
    diag_a = np.linalg.norm(a[:, 2:] - a[:, :2], axis=1)
    diag_b = np.linalg.norm(b[:, 2:] - b[:, :2], axis=1)
    return d / np.maximum((diag_a[:, None] + diag_b[None]) / 2, 1e-6)


def hungarian(cost: np.ndarray, max_cost: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal assignment, then reject pairs whose cost exceeds ``max_cost``."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return matches, [r for r in range(cost.shape[0]) if r not in mr], [c for c in range(cost.shape[1]) if c not in mc]


def greedy(score: np.ndarray, min_score: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Repeatedly take the highest-scoring remaining pair -- the IoU tracker's matcher."""
    matches: list[tuple[int, int]] = []
    if score.size:
        s = score.copy()
        while True:
            r, c = np.unravel_index(np.argmax(s), s.shape)
            if s[r, c] < min_score:
                break
            matches.append((int(r), int(c)))
            s[r, :] = -np.inf
            s[:, c] = -np.inf
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return matches, [r for r in range(score.shape[0]) if r not in mr], [c for c in range(score.shape[1]) if c not in mc]
