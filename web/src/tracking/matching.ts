// Association primitives; mirror src/facetrack/tracking/matching.py.

export type Xyxy = [number, number, number, number];

export function iouMatrix(a: Xyxy[], b: Xyxy[]): number[][] {
  return a.map((p) =>
    b.map((q) => {
      const iw = Math.max(0, Math.min(p[2], q[2]) - Math.max(p[0], q[0]));
      const ih = Math.max(0, Math.min(p[3], q[3]) - Math.max(p[1], q[1]));
      const inter = iw * ih;
      const union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter;
      return union > 0 ? inter / Math.max(union, 1e-9) : 0;
    }),
  );
}

export function centerDistance(a: Xyxy[], b: Xyxy[]): number[][] {
  return a.map((p) =>
    b.map((q) => {
      const d = Math.hypot((p[0] + p[2]) / 2 - (q[0] + q[2]) / 2, (p[1] + p[3]) / 2 - (q[1] + q[3]) / 2);
      const da = Math.hypot(p[2] - p[0], p[3] - p[1]);
      const db = Math.hypot(q[2] - q[0], q[3] - q[1]);
      return d / Math.max((da + db) / 2, 1e-6);
    }),
  );
}

export interface Assignment {
  matches: [number, number][];
  unmatchedRows: number[];
  unmatchedCols: number[];
}

function split(n: number, m: number, matches: [number, number][]): Assignment {
  const r = new Set(matches.map(([i]) => i));
  const c = new Set(matches.map(([, j]) => j));
  return {
    matches,
    unmatchedRows: [...Array(n).keys()].filter((i) => !r.has(i)),
    unmatchedCols: [...Array(m).keys()].filter((j) => !c.has(j)),
  };
}

/** Optimal assignment (Kuhn-Munkres with potentials, O(n^2 m)), then gate by maxCost.
 * `cols` is explicit: a matrix with zero rows cannot carry its column count, and
 * losing it would make every detection look matched when there are no tracks. */
export function hungarian(cost: number[][], maxCost: number, cols: number): Assignment {
  const n0 = cost.length;
  const m0 = cols;
  if (!n0 || !m0) return split(n0, m0, []);
  const transpose = n0 > m0;
  const C = transpose ? cost[0].map((_, j) => cost.map((row) => row[j])) : cost;
  const n = C.length;
  const m = C[0].length;
  const INF = Number.POSITIVE_INFINITY;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1);
  const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(INF);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const matches: [number, number][] = [];
  for (let j = 1; j <= m; j++) {
    if (!p[j]) continue;
    const r = p[j] - 1;
    const c = j - 1;
    const [row, col] = transpose ? [c, r] : [r, c];
    if (cost[row][col] <= maxCost) matches.push([row, col]);
  }
  matches.sort((a, b) => a[0] - b[0]);
  return split(n0, m0, matches);
}

/** Repeatedly take the highest-scoring remaining pair (the IoU tracker's matcher). */
export function greedy(score: number[][], minScore: number, cols: number): Assignment {
  const n = score.length;
  const m = cols;
  const s = score.map((r) => r.slice());
  const matches: [number, number][] = [];
  for (;;) {
    let best = -Infinity;
    let bi = -1;
    let bj = -1;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++)
        if (s[i][j] > best) {
          best = s[i][j];
          bi = i;
          bj = j;
        }
    if (bi < 0 || best < minScore) break;
    matches.push([bi, bj]);
    for (let j = 0; j < m; j++) s[bi][j] = -Infinity;
    for (let i = 0; i < n; i++) s[i][bj] = -Infinity;
  }
  return split(n, m, matches);
}
