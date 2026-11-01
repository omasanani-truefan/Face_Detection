// Time-aware constant-velocity Kalman box models; mirror src/facetrack/tracking/kalman.py.
// dt is elapsed time in *reference frames* (elapsedMs / refFrameMs).

import type { Xyxy } from "./matching.ts";

type Mat = number[][];

const eye = (n: number, m = n): Mat => Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => (i === j ? 1 : 0)));
const diag = (d: number[]): Mat => d.map((v, i) => d.map((_, j) => (i === j ? v : 0)));
const T = (a: Mat): Mat => a[0].map((_, j) => a.map((r) => r[j]));
const mul = (a: Mat, b: Mat): Mat => a.map((r) => b[0].map((_, j) => r.reduce((s, v, k) => s + v * b[k][j], 0)));
const add = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((v, j) => v + b[i][j]));
const sub = (a: Mat, b: Mat): Mat => a.map((r, i) => r.map((v, j) => v - b[i][j]));
const scale = (a: Mat, s: number): Mat => a.map((r) => r.map((v) => v * s));
const mv = (a: Mat, x: number[]): number[] => a.map((r) => r.reduce((s, v, k) => s + v * x[k], 0));

/** Solve A X = B for X (A square, small) by Gauss-Jordan with partial pivoting. */
function solve(A: Mat, B: Mat): Mat {
  const n = A.length;
  const M = A.map((r, i) => [...r, ...B[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    for (let k = 0; k < M[c].length; k++) M[c][k] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f) for (let k = 0; k < M[r].length; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r) => r.slice(n));
}

class LinearKF {
  x: number[];
  P: Mat;
  private H: Mat;
  private m: number;
  constructor(x: number[], P: Mat, m: number) {
    this.x = x;
    this.P = P;
    this.m = m;
    this.H = eye(m, x.length);
  }
  private F(dt: number): Mat {
    const n = this.x.length;
    const F = eye(n);
    for (let i = 0; i < n - this.m; i++) F[i][this.m + i] = dt;
    return F;
  }
  predict(dt: number, Q: Mat): void {
    const F = this.F(dt);
    this.x = mv(F, this.x);
    this.P = add(mul(mul(F, this.P), T(F)), Q);
  }
  update(z: number[], R: Mat): void {
    const Hx = mv(this.H, this.x);
    const y = z.map((v, i) => v - Hx[i]);
    const S = add(mul(mul(this.H, this.P), T(this.H)), R);
    const K = T(solve(S, mul(this.H, this.P)));
    this.x = this.x.map((v, i) => v + K[i].reduce((s, k, j) => s + k * y[j], 0));
    this.P = mul(sub(eye(this.x.length), mul(K, this.H)), this.P);
  }
}

export interface MotionModel {
  predict(dt: number): Xyxy;
  update(box: Xyxy): void;
  box(): Xyxy;
}

const toXysr = (b: Xyxy) => {
  const w = b[2] - b[0];
  const h = b[3] - b[1];
  return [b[0] + w / 2, b[1] + h / 2, w * h, w / Math.max(h, 1e-6)];
};

export class SortKalman implements MotionModel {
  private kf: LinearKF;
  private Q = diag([1, 1, 1, 1, 0.01, 0.01, 1e-4]);
  private R = diag([1, 1, 10, 10]);
  constructor(b: Xyxy) {
    this.kf = new LinearKF([...toXysr(b), 0, 0, 0], diag([10, 10, 10, 10, 1e4, 1e4, 1e4]), 4);
  }
  predict(dt: number): Xyxy {
    if (this.kf.x[2] + this.kf.x[6] * dt <= 0) this.kf.x[6] = 0;
    this.kf.predict(dt, scale(this.Q, Math.max(dt, 1e-3)));
    return this.box();
  }
  update(b: Xyxy): void {
    this.kf.update(toXysr(b), this.R);
  }
  box(): Xyxy {
    const [x, y, s0, r0] = this.kf.x;
    const s = Math.max(s0, 1e-6);
    const r = Math.max(r0, 1e-6);
    const w = Math.sqrt(s * r);
    const h = s / w;
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2];
  }
}

const toXyah = (b: Xyxy) => {
  const w = b[2] - b[0];
  const h = Math.max(b[3] - b[1], 1e-6);
  return [b[0] + w / 2, b[1] + h / 2, w / h, h];
};

export class XyahKalman implements MotionModel {
  private kf: LinearKF;
  private static sp = 1 / 20;
  private static sv = 1 / 160;
  constructor(b: Xyxy) {
    const z = toXyah(b);
    const h = z[3];
    const { sp, sv } = XyahKalman;
    const std = [2 * sp * h, 2 * sp * h, 1e-2, 2 * sp * h, 10 * sv * h, 10 * sv * h, 1e-5, 10 * sv * h];
    this.kf = new LinearKF([...z, 0, 0, 0, 0], diag(std.map((s) => s * s)), 4);
  }
  predict(dt: number): Xyxy {
    const h = this.kf.x[3];
    const { sp, sv } = XyahKalman;
    const std = [sp * h, sp * h, 1e-2, sp * h, sv * h, sv * h, 1e-5, sv * h];
    this.kf.predict(dt, scale(diag(std.map((s) => s * s)), Math.max(dt, 1e-3)));
    return this.box();
  }
  update(b: Xyxy): void {
    const h = this.kf.x[3];
    const sp = XyahKalman.sp;
    const std = [sp * h, sp * h, 1e-1, sp * h];
    this.kf.update(toXyah(b), diag(std.map((s) => s * s)));
  }
  box(): Xyxy {
    const [cx, cy, a] = this.kf.x;
    const h = Math.max(this.kf.x[3], 1e-6);
    const w = a * h;
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }
}
