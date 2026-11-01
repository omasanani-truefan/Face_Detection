"""Constant-velocity Kalman box models, time-aware.

Both SORT and ByteTrack were published with a per-*frame* motion model (dt = 1). We
run on phones and browsers whose frame interval varies, so ``dt`` here is the elapsed
time expressed in *reference frames* (``elapsed_ms / ref_frame_ms``). At the reference
rate this reproduces the published filters exactly; at other rates velocity is still in
px/reference-frame, so a 15 fps stream predicts twice as far per step as a 30 fps one.
"""

from __future__ import annotations

import numpy as np


class _LinearKF:
    """Minimal linear KF over an ``n``-dim state with the first ``m`` dims measured."""

    def __init__(self, x: np.ndarray, P: np.ndarray, m: int):
        self.x = x.astype(np.float64)
        self.P = P.astype(np.float64)
        n = len(x)
        self.m = m
        self.H = np.eye(m, n)

    def _F(self, dt: float) -> np.ndarray:
        n = len(self.x)
        F = np.eye(n)
        k = n - self.m  # number of velocity components
        F[:k, self.m : self.m + k] = np.eye(k) * dt
        return F

    def predict(self, dt: float, Q: np.ndarray) -> None:
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P


def xyxy_to_xysr(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def xysr_to_xyxy(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class SortKalman:
    """SORT's filter (Bewley et al. 2016): state [u, v, s, r, du, dv, ds], s = area,
    r = aspect ratio held constant. Noise constants are the reference implementation's.
    """

    def __init__(self, box_xyxy: np.ndarray):
        P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        x = np.zeros(7)
        x[:4] = xyxy_to_xysr(box_xyxy)
        self.kf = _LinearKF(x, P, m=4)
        self._Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
        self._R = np.diag([1.0, 1.0, 10.0, 10.0])

    def predict(self, dt: float) -> np.ndarray:
        # A shrinking box must not predict a negative area.
        if self.kf.x[2] + self.kf.x[6] * dt <= 0:
            self.kf.x[6] = 0.0
        # Process noise grows with the step so long gaps widen uncertainty.
        self.kf.predict(dt, self._Q * max(dt, 1e-3))
        return self.box()

    def update(self, box_xyxy: np.ndarray) -> None:
        self.kf.update(xyxy_to_xysr(box_xyxy), self._R)

    def box(self) -> np.ndarray:
        return xysr_to_xyxy(self.kf.x[:4])


class XyahKalman:
    """ByteTrack / DeepSORT filter: state [cx, cy, a, h, vcx, vcy, va, vh] with noise
    proportional to box height, so a 30 px face and a 300 px face get comparable
    *relative* uncertainty -- useful when far and near faces share a frame.
    """

    std_pos = 1.0 / 20
    std_vel = 1.0 / 160

    def __init__(self, box_xyxy: np.ndarray):
        z = self._to_xyah(box_xyxy)
        h = z[3]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        self.kf = _LinearKF(np.r_[z, np.zeros(4)], np.diag(np.square(std)), m=4)

    @staticmethod
    def _to_xyah(b: np.ndarray) -> np.ndarray:
        w, h = b[2] - b[0], max(b[3] - b[1], 1e-6)
        return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])

    def predict(self, dt: float) -> np.ndarray:
        h = self.kf.x[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
               self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        self.kf.predict(dt, np.diag(np.square(std)) * max(dt, 1e-3))
        return self.box()

    def update(self, box_xyxy: np.ndarray) -> None:
        h = self.kf.x[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]
        self.kf.update(self._to_xyah(box_xyxy), np.diag(np.square(std)))

    def box(self) -> np.ndarray:
        cx, cy, a, h = self.kf.x[:4]
        h = max(h, 1e-6)
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
