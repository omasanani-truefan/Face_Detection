// Contracts shared by every stage; mirror src/facetrack/types.py.
// Boxes are [x, y, w, h] in source-frame pixels; timestamps are milliseconds taken from
// the video's media time or the camera clock, never from a frame counter.

export type Box = [number, number, number, number];
export type Point = [number, number];

export interface Detection {
  bbox: Box;
  confidence: number;
  keypoints?: Point[];
}

export interface StageTimings {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  /** True when the runtime fuses stages (MediaPipe): the split is not observable. */
  fused: boolean;
}

export function toXyxy(b: Box): [number, number, number, number] {
  return [b[0], b[1], b[0] + b[2], b[1] + b[3]];
}
