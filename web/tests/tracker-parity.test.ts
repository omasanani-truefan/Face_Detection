// The TypeScript trackers must reproduce the Python reference frame by frame.
// Fixture: python scripts/make_tracker_fixture.py
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { createTracker, visible, type TrackerName } from "../src/tracking/tracker.ts";

interface Case {
  seed: number;
  config: Record<string, number>;
  frames: { t: number; dets: number[][] }[];
  expected: Record<string, { visible: number[][]; events: [string, number][] }[]>;
}

const cases: Case[] = JSON.parse(readFileSync(new URL("./fixtures/tracker_parity.json", import.meta.url), "utf8"));
const camel = (k: string) => k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

for (const c of cases) {
  for (const name of ["iou", "sort", "bytetrack", "sort+lastobs", "bytetrack+lastobs"]) {
    test(`parity seed=${c.seed} ${name}`, () => {
      const cfg = Object.fromEntries(Object.entries(c.config).map(([k, v]) => [camel(k), v]));
      const [base, variant] = name.split("+") as [TrackerName, string?];
      const trk = createTracker(base, { ...cfg, lostMatchLastObservation: variant === "lastobs" });
      c.frames.forEach((fr, i) => {
        const out = trk.update(fr.dets.map((d) => ({ bbox: [d[0], d[1], d[2], d[3]], confidence: d[4] })), fr.t, i);
        const exp = c.expected[name][i];
        const got = visible(out).map((t) => [t.trackId, ...t.box]);
        assert.equal(got.length, exp.visible.length, `frame ${i}: visible count`);
        got.forEach((g, k) => {
          assert.equal(g[0], exp.visible[k][0], `frame ${i}: track id`);
          for (let j = 1; j < 5; j++) assert.ok(Math.abs(g[j] - exp.visible[k][j]) < 1e-3, `frame ${i}: box ${g} vs ${exp.visible[k]}`);
        });
        const ev = out.events.filter((e) => e.event !== "track_updated").map((e) => [e.event, e.trackId]);
        assert.deepEqual(ev, exp.events, `frame ${i}: events`);
      });
    });
  }
}

test("re-entry after termination gets a new id; one miss never terminates", () => {
  const trk = createTracker("bytetrack", { minHits: 3, maxLostMs: 1000 });
  const face = { bbox: [100, 100, 60, 70] as [number, number, number, number], confidence: 0.9 };
  let t = 0;
  const step = (dets: (typeof face)[]) => trk.update(dets, (t += 33.3));
  for (let i = 0; i < 10; i++) step([face]);
  assert.equal(step([]).tracks[0].state, "lost");
  for (let i = 0; i < 5; i++) step([face]);
  for (let i = 0; i < 40; i++) step([]);
  let out = step([face]);
  for (let i = 0; i < 3; i++) out = step([face]);
  assert.deepEqual(visible(out).map((x) => x.trackId), [2]);
});
