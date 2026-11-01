# Recording and annotation protocol for real benchmark videos

The detection and tracking benchmarks currently use WIDER FACE (photos) and synthetic-motion
sequences built from WIDER faces (report §4). Both are stand-ins. This protocol replaces
them with **our own recorded, consented, annotated video**, without code changes:
recorded clips are imported into `data/sequences/` and evaluated by the same scripts.

## 1. Consent and privacy (before anything is recorded)

* Every person who appears gives **written, informed consent** to being recorded *for
  evaluating face detection and tracking*, stating where the footage is stored, who can
  access it and when it will be deleted. Bystanders who have not consented must not be in
  frame; if one is, discard the clip — do not blur it and keep it.
* Store raw footage only on access-controlled project storage, never in this repository or
  in chat tools. `data/` is ignored by git for this reason.
* `import_mot_annotations.py` requires a `--consent-ref` (an ID pointing at the consent
  record, never the record itself) so each clip stays traceable to its consent.
* Delete footage when the evaluation period agreed in the consent ends. Face crops and
  any future embeddings are biometric data; treat them under the applicable law (e.g. GDPR
  Art. 9, India DPDP Act) — get the data-protection officer's sign-off before Phase 3.

## 2. What to record (minimum set: ~30 clips of 10–30 s)

Record each scenario on **at least two devices**, and note for every clip: device and
which camera, resolution, frame rate setting, approximate subject distance, lighting.

| Scenario | Clips | Content |
|---|---|---|
| A. Close range | 6 | 1–3 people seated 0.5–1.5 m from a laptop webcam / phone front camera; frontal and ±30° yaw; talking, expressions |
| B. Far range | 6 | 2–5 people standing 3–6 m away; faces near the frame edges; 720p and 480p streams; phone rear camera |
| C. Dynamic | 8 | A seated, B enters, C enters (the brief's example); walking; two people crossing; head turned away for 1–3 s; a hand or object covering a face for ~1 s; one person leaving permanently; the same person returning after ≥ 5 s |
| D. Lighting | 6 | daylight window, indoor artificial, dim room, strong backlight (window behind subjects), side lighting |
| E. Devices | (covered above) | desktop webcam, phone front, phone rear, one low-end phone |

Split **by recording session**, not by clip: all clips from one session go to either
`dev` or `test`, so near-identical frames never straddle the split. Aim for 40% dev / 60% test.

## 3. Annotation

1. Pre-annotate with the current best pipeline:
   `facetrack-demo --video clip.mp4 --no-display --export-mot clip_pre.txt`
2. Import `clip_pre.txt` into **CVAT** as *MOT 1.1* and correct it: every visible face gets
   a tight box from eyebrows to chin, ears excluded; one identity per continuous
   appearance (a person who leaves and returns gets a **new** track ID, and — in a separate
   column or note — the same person ID); mark faces < 50% visible or turned past profile as
   *not visible* (they become ignore regions).
3. Export *MOT 1.1* and import:
   `python scripts/import_mot_annotations.py --video clip.mp4 --gt gt.txt --split test --tags close multi --device "Pixel 7 front" --distance-m 1.2 --consent-ref C-2026-014`
4. A second annotator reviews 10% of frames; disagreements above 10% IoU are re-labelled.

Detection evaluation on recorded clips uses the same GT (every annotated frame is an image
with face boxes); tracking evaluation uses the identities.

## 4. Re-running

```bash
python scripts/bench_tracking.py            # recorded clips are picked up from the manifest
python scripts/make_report_tables.py
```
Recorded clips are reported with `generator = recorded`, separately from synthetic ones.
