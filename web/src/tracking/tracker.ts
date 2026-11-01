// Track lifecycle + the three association strategies; mirrors src/facetrack/tracking/
// base.py and trackers.py (same states, events, parameters and time-based expiry).
// Config keys are camelCase here; the Python names are the snake_case equivalents.

import type { Detection, Point } from "../types.ts";
import { type MotionModel, SortKalman, XyahKalman } from "./kalman.ts";
import { centerDistance, greedy, hungarian, iouMatrix, type Xyxy } from "./matching.ts";

export type TrackState = "tentative" | "confirmed" | "lost" | "terminated";
export type EventType = "track_started" | "track_updated" | "track_lost" | "track_recovered" | "track_ended";

export interface TrackerConfig {
  detThreshold: number;
  lowDetThreshold: number;
  newTrackThreshold: number | null;
  minHits: number;
  tentativeTimeoutMs: number;
  maxLostMs: number;
  iouThreshold: number;
  lowIouThreshold: number;
  centerDistanceThreshold: number;
  refFrameMs: number;
  emitUpdates: boolean;
  /** Observation-centric recovery: a lost track may match on its last observed box and
   * restarts its motion model (see Python TrackerConfig.lost_match_last_observation). */
  lostMatchLastObservation: boolean;
}

export const DEFAULT_TRACKER_CONFIG: TrackerConfig = {
  detThreshold: 0.5,
  lowDetThreshold: 0.1,
  newTrackThreshold: null,
  minHits: 3,
  tentativeTimeoutMs: 300,
  maxLostMs: 1000,
  iouThreshold: 0.3,
  lowIouThreshold: 0.5,
  centerDistanceThreshold: 0.6,
  refFrameMs: 1000 / 30,
  emitUpdates: true,
  lostMatchLastObservation: false,
};

export interface Track {
  trackId: number;
  state: TrackState;
  box: Xyxy;
  confidence: number;
  firstSeenMs: number;
  lastSeenMs: number;
  keypoints?: Point[];
  hits: number;
  confirmedMs: number | null;
  matched: boolean;
}

export interface TrackEvent {
  event: EventType;
  trackId: number;
  timestampMs: number;
  frameId: number;
  track: TrackSnapshot;
}

export interface TrackSnapshot {
  track_id: number;
  bbox: [number, number, number, number];
  confidence: number;
  status: TrackState;
  first_seen: number;
  last_seen: number;
  hits: number;
}

export interface TrackerOutput {
  frameId: number;
  timestampMs: number;
  tracks: Track[];
  events: TrackEvent[];
  detectorRan: boolean;
}

interface Live extends Track {
  motion: MotionModel | null;
  obsBox: Xyxy; // last detection box this track was matched to
}

interface Association {
  matches: [number, number][];
  newTrackCandidates: number[];
}

export function snapshot(t: Track): TrackSnapshot {
  const r = (v: number, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
  return {
    track_id: t.trackId,
    bbox: [r(t.box[0]), r(t.box[1]), r(t.box[2] - t.box[0]), r(t.box[3] - t.box[1])],
    confidence: r(t.confidence, 4),
    status: t.state,
    first_seen: r(t.firstSeenMs, 3),
    last_seen: r(t.lastSeenMs, 3),
    hits: t.hits,
  };
}

/** Confirmed tracks matched at the latest detector update (plus lost ones if asked). */
export function visible(out: TrackerOutput, includeLost = false): Track[] {
  return out.tracks.filter((t) => (t.state === "confirmed" && t.matched) || (includeLost && t.state === "lost"));
}

const detBox = (d: Detection): Xyxy => [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]];

export abstract class BaseTracker {
  abstract readonly name: string;
  readonly cfg: TrackerConfig;
  protected tracks: Live[] = [];
  private nextId = 1;
  private lastTs: number | null = null;

  constructor(cfg: Partial<TrackerConfig> = {}) {
    this.cfg = { ...DEFAULT_TRACKER_CONFIG, ...cfg };
  }

  protected abstract associate(tracks: Live[], dets: Detection[]): Association;
  protected newMotion(_box: Xyxy): MotionModel | null {
    return null;
  }
  protected minDetScore(): number {
    return this.cfg.detThreshold;
  }

  get idsIssued(): number {
    return this.nextId - 1;
  }

  /** Forget live tracks (video restarted/seeked). IDs keep increasing. */
  reset(): void {
    this.tracks = [];
    this.lastTs = null;
  }

  update(detections: Detection[], timestampMs: number, frameId = -1): TrackerOutput {
    const events: TrackEvent[] = [];
    this.advance(timestampMs);
    const cfg = this.cfg;
    const dets = detections.filter((d) => d.confidence >= this.minDetScore());
    const assoc = this.associate(this.tracks, dets);
    const matched = new Set<number>();
    for (const [ti, di] of assoc.matches) {
      const t = this.tracks[ti];
      const d = dets[di];
      matched.add(ti);
      const box = detBox(d);
      if (t.motion) {
        if (t.state === "lost" && cfg.lostMatchLastObservation) t.motion = this.newMotion(box);
        else t.motion.update(box);
        t.box = t.motion!.box();
      } else t.box = box;
      t.obsBox = box;
      t.confidence = d.confidence;
      t.keypoints = d.keypoints;
      t.lastSeenMs = timestampMs;
      t.matched = true;
      t.hits += 1;
      if (t.state === "tentative" && t.hits >= cfg.minHits) this.confirm(t, timestampMs, frameId, events);
      else if (t.state === "lost") {
        t.state = "confirmed";
        events.push(this.event("track_recovered", t, timestampMs, frameId));
      } else if (t.state === "confirmed" && cfg.emitUpdates) events.push(this.event("track_updated", t, timestampMs, frameId));
    }
    this.tracks.forEach((t, ti) => {
      if (matched.has(ti)) return;
      t.matched = false;
      if (t.state === "confirmed") {
        // One miss never ends a track: it goes LOST and gets the grace period.
        t.state = "lost";
        events.push(this.event("track_lost", t, timestampMs, frameId));
      }
    });
    const newThr = cfg.newTrackThreshold ?? cfg.detThreshold;
    for (const di of assoc.newTrackCandidates) {
      const d = dets[di];
      if (d.confidence < newThr) continue;
      const box = detBox(d);
      const t: Live = {
        trackId: this.nextId++, state: "tentative", box, confidence: d.confidence, firstSeenMs: timestampMs,
        lastSeenMs: timestampMs, keypoints: d.keypoints, hits: 1, confirmedMs: null, matched: true, motion: this.newMotion(box),
        obsBox: box,
      };
      this.tracks.push(t);
      if (cfg.minHits <= 1) this.confirm(t, timestampMs, frameId, events);
    }
    this.expire(timestampMs, frameId, events);
    return { frameId, timestampMs, tracks: this.tracks.map(copyTrack), events, detectorRan: true };
  }

  /** Advance motion without a detector result (frame skipping). No lifecycle changes. */
  predict(timestampMs: number, frameId = -1): TrackerOutput {
    this.advance(timestampMs);
    return { frameId, timestampMs, tracks: this.tracks.map(copyTrack), events: [], detectorRan: false };
  }

  /** Stream stopped: end every announced track. */
  finish(timestampMs: number, frameId = -1): TrackEvent[] {
    const events: TrackEvent[] = [];
    for (const t of this.tracks)
      if (t.state === "confirmed" || t.state === "lost") {
        t.state = "terminated";
        events.push(this.event("track_ended", t, timestampMs, frameId));
      }
    this.tracks = [];
    return events;
  }

  /** IoU with predicted boxes; lost tracks may use their last observed box instead. */
  protected matchIou(tracks: Live[], db: Xyxy[]): number[][] {
    const iou = iouMatrix(tracks.map((t) => t.box), db);
    if (this.cfg.lostMatchLastObservation)
      tracks.forEach((t, i) => {
        if (t.state !== "lost") return;
        const o = iouMatrix([t.obsBox], db)[0];
        iou[i] = iou[i].map((v, j) => Math.max(v, o[j]));
      });
    return iou;
  }

  private advance(ts: number): void {
    if (this.lastTs !== null && ts < this.lastTs)
      throw new Error(`timestamps must not go backwards (${ts} < ${this.lastTs}); call reset() on seek`);
    const dt = this.lastTs === null ? 1 : (ts - this.lastTs) / this.cfg.refFrameMs;
    this.lastTs = ts;
    if (dt <= 0) return;
    for (const t of this.tracks) if (t.motion) t.box = t.motion.predict(dt);
  }

  private confirm(t: Live, ts: number, frameId: number, events: TrackEvent[]): void {
    t.state = "confirmed";
    t.confirmedMs = ts;
    events.push(this.event("track_started", t, ts, frameId));
  }

  private expire(ts: number, frameId: number, events: TrackEvent[]): void {
    this.tracks = this.tracks.filter((t) => {
      const unseen = ts - t.lastSeenMs;
      if (t.state === "tentative" && !t.matched && unseen > this.cfg.tentativeTimeoutMs) return false;
      if (t.state === "lost" && unseen > this.cfg.maxLostMs) {
        t.state = "terminated";
        events.push(this.event("track_ended", t, ts, frameId));
        return false;
      }
      return true;
    });
  }

  private event(event: EventType, t: Track, ts: number, frameId: number): TrackEvent {
    return { event, trackId: t.trackId, timestampMs: ts, frameId, track: snapshot(t) };
  }
}

function copyTrack(t: Live): Track {
  const { motion: _motion, obsBox: _obs, ...rest } = t;
  return { ...rest, box: [...t.box] as Xyxy };
}

const boxes = (ts: Track[]): Xyxy[] => ts.map((t) => t.box);

export class IoUTracker extends BaseTracker {
  readonly name = "iou";
  protected associate(tracks: Live[], dets: Detection[]): Association {
    const tb = boxes(tracks);
    const db = dets.map(detBox);
    const iou = iouMatrix(tb, db);
    const cd = centerDistance(tb, db);
    const score = iou.map((row, i) =>
      row.map((v, j) =>
        v >= this.cfg.iouThreshold || cd[i][j] <= this.cfg.centerDistanceThreshold ? v + 0.1 * (1 - Math.min(cd[i][j], 1)) : -Infinity,
      ),
    );
    const a = greedy(score, -1e9, dets.length);
    return { matches: a.matches, newTrackCandidates: a.unmatchedCols };
  }
}

export class SortTracker extends BaseTracker {
  readonly name = "sort";
  protected newMotion(box: Xyxy): MotionModel {
    return new SortKalman(box);
  }
  protected associate(tracks: Live[], dets: Detection[]): Association {
    const iou = this.matchIou(tracks, dets.map(detBox));
    const a = hungarian(iou.map((r) => r.map((v) => 1 - v)), 1 - this.cfg.iouThreshold, dets.length);
    return { matches: a.matches, newTrackCandidates: a.unmatchedCols };
  }
}

export class ByteTracker extends BaseTracker {
  readonly name = "bytetrack";
  protected newMotion(box: Xyxy): MotionModel {
    return new XyahKalman(box);
  }
  protected minDetScore(): number {
    return this.cfg.lowDetThreshold;
  }
  protected associate(tracks: Live[], dets: Detection[]): Association {
    const cfg = this.cfg;
    const scores = dets.map((d) => d.confidence);
    const idx = (n: number) => [...Array(n).keys()];
    const high = idx(dets.length).filter((i) => scores[i] >= cfg.detThreshold);
    const low = idx(dets.length).filter((i) => scores[i] < cfg.detThreshold);
    const active = idx(tracks.length).filter((i) => tracks[i].state === "confirmed" || tracks[i].state === "lost");
    const tentative = idx(tracks.length).filter((i) => tracks[i].state === "tentative");
    const tb = boxes(tracks);
    const db = dets.map(detBox);
    const matches: [number, number][] = [];

    // Stage 1: high-score detections, IoU fused with score, raw-IoU gated.
    const iou1 = this.matchIou(active.map((i) => tracks[i]), high.map((j) => db[j]));
    const cost1 = iou1.map((row) => row.map((v, c) => (v >= cfg.iouThreshold ? 1 - v * scores[high[c]] : 1e6)));
    const a1 = hungarian(cost1, 1 - cfg.iouThreshold * cfg.detThreshold, high.length);
    for (const [r, c] of a1.matches) matches.push([active[r], high[c]]);
    const remActive = a1.unmatchedRows.map((r) => active[r]);
    let remHigh = a1.unmatchedCols.map((c) => high[c]);

    // Stage 2: low-score detections may only extend currently confirmed tracks.
    const confirmedLeft = remActive.filter((i) => tracks[i].state === "confirmed");
    if (confirmedLeft.length && low.length) {
      const iou2 = iouMatrix(confirmedLeft.map((i) => tb[i]), low.map((j) => db[j]));
      const a2 = hungarian(iou2.map((r) => r.map((v) => 1 - v)), 1 - cfg.lowIouThreshold, low.length);
      for (const [r, c] of a2.matches) matches.push([confirmedLeft[r], low[c]]);
    }

    // Stage 3: tentative tracks against leftover high-score detections.
    if (tentative.length && remHigh.length) {
      const iou3 = iouMatrix(tentative.map((i) => tb[i]), remHigh.map((j) => db[j]));
      const a3 = hungarian(iou3.map((r) => r.map((v) => 1 - v)), 1 - cfg.iouThreshold, remHigh.length);
      for (const [r, c] of a3.matches) matches.push([tentative[r], remHigh[c]]);
      remHigh = a3.unmatchedCols.map((c) => remHigh[c]);
    }
    return { matches, newTrackCandidates: remHigh };
  }
}

export type TrackerName = "iou" | "sort" | "bytetrack";

export function createTracker(name: TrackerName, cfg: Partial<TrackerConfig> = {}): BaseTracker {
  if (name === "iou") return new IoUTracker(cfg);
  if (name === "sort") return new SortTracker(cfg);
  return new ByteTracker(cfg);
}
