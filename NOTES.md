# Notes — video-analytics

Working log: what was decided, what broke, what was tried. Newest section last.

## 2026-10-25 — scope and data

**Question.** Not "can I draw a counting line on a video" (a library does that in ten lines),
but: how wrong is the count, does a net count hide errors that cancel, how much of the error is
identity switches, and what does each CPU speed-up cost in accuracy.

**Data choice.** Checked live before choosing:

| Candidate | Outcome |
|---|---|
| MOT17 / MOT20 (motchallenge.net) | **Chosen.** Annotated track ids, scriptable download with HTTP range support, CC BY-NC-SA 3.0. Only the needed zip members are pulled (about 3.6 GB of 11 GB). |
| UA-DETRAC (vehicles) | Original host gone; the Hugging Face mirror is tagged CC BY 4.0 by its uploader, which does not reflect the original terms. Not used. |
| BrnoCompSpeed (speeds) | Data host does not resolve (NXDOMAIN); access now by emailing the authors. Speed estimation is parked until then. |
| Zenodo "Vehicle Tracking Dataset" | CC BY 4.0 on the record, but the clips come from AI City Challenge footage and a TV video, so the rights are doubtful. Not used. |
| VS13 | No licence stated; speeds from cruise control, no road calibration. Not used. |

**Split.** Only MOT17's static-camera sequences (02, 04, 09) are usable for a fixed line: a line
fixed in the image moves in the world when the camera moves. They are the dev set. MOT20's four
train sequences (different places, much denser crowds) are the test set, scored once.

**Lines.** Mechanical: the horizontal and vertical midline of each frame, generated from
`seqinfo.ini` into `configs/lines.json` before any prediction existed. No line was placed by eye.

## Decisions

| Decision | Chose | Over | Why |
|---|---|---|---|
| Detector | RF-DETR (Apache-2.0), COCO weights, no fine-tuning | Ultralytics YOLO | AGPL-3.0; and fine-tuning on MOT would blur dev/test |
| Tracker | ByteTrack from `trackers` 2.6.1 (Apache-2.0) | `supervision.ByteTrack` | `supervision` 0.31 no longer ships it; it moved to `trackers` |
| Truth crossings | GT tracks through the same counting code, naive rule | a separate truth counter | the counting rule can't masquerade as accuracy; a band that hides a real back-and-forth is an error |
| Crossing match | identity-aware (the predicted track must be on the person who crossed) | time-only | in a crowd, a miss and someone else's double count land within a second of each other and cancel |
| Reference model | RF-DETR Small | Medium | Medium over 11,106 frames on a shared 6-core laptop CPU would take most of a day per config |
| Detection cache | score ≥ 0.10, thresholds applied on replay | caching at the final threshold | ByteTrack's second association needs the low-score boxes, and tuning never re-runs the detector |

## What broke

- `motmetrics` 1.4.0's `iou_matrix` calls `np.asfarray`, removed in numpy 2. The accumulator and
  `compute` still work, so the distance matrix is computed in `mot_eval._iou_distance` (1 − IoU,
  nan below the threshold): same output, no pin or patch.
- `trackers` declares the GUI build of OpenCV, which clashes with the headless wheel. A uv
  `override-dependencies` entry drops it; it imports fine against the headless build.
- `trackers` ByteTrack returns id −1 for the first observation of every track, and its ids start
  at 0. `run_bytetrack` drops the −1 rows and adds 1 to every id; each track therefore loses its
  first row (the library's `minimum_consecutive_frames = 2` costs one more).
- Plain `uv run` exact-syncs and removes the `vision` extra. Use `uv run --extra vision`.

## Speed numbers so far

Not quotable. The first timings ran while other projects held the CPU at 60–100 %: PyTorch and
ONNX swapped places between runs. That is why the plan has a quiet-machine gate.

## 2026-10-25 — equivalence check (dev frames, an H4 input)

ONNX fp32 vs PyTorch on 50 evenly spaced dev frames, at the cache threshold 0.10 that every
downstream step uses (`results/equivalence.csv`):

| Model | Frames with equal box count | Matched boxes | Min IoU | Mean IoU | p01 IoU |
|---|---|---|---|---|---|
| Nano | 50 / 50 | 2,787 | 0.983 | 0.99996 | 0.9993 |
| Small | 49 / 50 | 2,336 | 0.897 | 0.99981 | 0.9984 |

The plan says "matched boxes IoU ≥ 0.99". Read literally, that is every matched box, so the
equivalence clause of H4 **fails for both models**: Nano falls below it in one frame (MOT17-09
frame 170) and Small in two (MOT17-02 frames 223 and 400). The plan does not name the threshold
for this check. At 0.30 (`equivalence_thr0.30.csv`) Nano passes (min 0.9946) and Small still fails
(min 0.9805), so H4, which needs both models, fails under either reading. The rule is not being
relaxed after the fact; the percentile columns are descriptive only. The two backends agree on
the typical box to about four decimal places of IoU and disagree on a few boxes. The per-frame
file records no scores, so whether those are low-score boxes is not checked.

## 2026-10-25 — dev observation (tuning numbers, not results)

A Small-only pass over the dev grid (`results/scratch/tuneS`, not the committed tuning run) gave
dev f1 0.31 (naive) and 0.46 (hysteresis, band 32) at k = 1, τ = 0.6, on 94 true crossings.
Nearly all the error is extra crossings, and nearly none of it is identity switches
(`switch_share` 0.03–0.09). Two patterns, checked by hand:

- **Jitter on the line.** One MOT17-04 track crossed `h_mid` 57 times under the naive rule: a
  person standing on it. The band removes this (MOT17-04 `h_mid` extra crossings 62 → 2).
- **Anchor jumps under occlusion.** With the band on, most remaining extra crossings happen at a
  frame where the predicted box overlaps no annotated box at IoU ≥ 0.5 (34 of 45). The detector
  draws the visible part of a person; MOT ground truth draws the whole body, legs included, behind
  occluders. When the legs disappear the box's bottom edge jumps up by more than the band, and back
  when they reappear: a crossing nobody made. MOT17-09's `h_mid` has 0 true crossings and still 19
  predicted ones with a 32 px band. Predictions on non-pedestrian annotations (static people,
  reflections, distractors) account for a handful only.

So the bottom-centre of a detection box is not a foot position. That is the main threat to any
line counter built on box anchors, and it is not something a tracker setting fixes.

## 2026-10-25 — scope vs the approved slot brief (decided: amendment A2)

The slot brief approved on 2026-09-30 was read only after the build started. Where this repo
differs from it:

| Brief (approved) | Built here | Status |
|---|---|---|
| Vehicles (car, motorcycle, bus, truck), traffic-camera framing | Pedestrians (MOT person class) | divergence |
| Held-out data: owner-filmed fixed-camera clips + hand counts (decision 2a) | MOT17 static sequences (dev) → MOT20 (test), annotated track ids as truth | divergence |
| Pre-registered: abs count error ≤ 5 % per line on ≥ 2 of 3 clips (proposal) | H1–H4 in `EVAL_PLAN.md` (hidden error, hysteresis, int8, ONNX) | divergence |
| Wilson intervals; ID switches on a hand-labelled 60 s sub-clip | Cluster bootstrap; ID switches and error causes on every frame from annotated ids | stronger, different |
| Baselines: detection-only per-frame count; tracker defaults | Naive vs hysteresis rule; tracker defaults except the high-score threshold | baseline (1) missing |
| COCO RF-DETR Nano/Small, no #26 weights | Same | as briefed |
| `supervision` ByteTrack, else `trackers` (Apache-2.0) | `trackers` 2.6.1 (supervision 0.31 dropped it) | as briefed |
| Detect stage writes a cache, CPU replay for track/count | Same (`.npz` cache instead of CSV) | as briefed |
| T3: ONNX export + FPS harness | Built, with a quiet-machine gate | as briefed |
| T4–T6 (speed, signal phase, SPaT) | Not started; T4 blocked on BrnoCompSpeed access | later |

Why MOT was chosen, for the owner's decision: annotated identities make the question's second
half ("what share of the error is identity switches vs missed detections") measurable on every
crossing rather than on one hand-labelled minute; the test set is public, so anyone can rerun it;
and it needs no filming. What it gives up: the traffic framing that P-17/P-18 build on, and a
hand count on the owner's own camera. Before the decision, the MOT pipeline kept running as
built; nothing in it is tuned to the choice.

**Decision (owner, 2026-10-25):** keep MOT17 → MOT20 as the primary benchmark, superseding brief
decision 2a; recorded as `EVAL_PLAN.md` amendment A2. Own vehicle clips with a hand count become
a later, separate external check with its own pre-registration.

## 2026-10-25 — dev selection committed (frozen grid, before any test scoring)

`vidan eval tune` on all four configs (256 grid rows, 9 min) → `results/dev_grid.csv`,
`results/dev_selection.csv`. All 12 dev caches complete, 0 bad frames. Small's rows are identical
to the earlier scratch pass. Dev f1 at k = 1 (tuning numbers): S 0.31 naive / 0.46 hysteresis,
S8 0.26 / 0.39, N 0.21 / 0.34, N8 0.24 / 0.36.

Two things the grid shows, reported and not acted on (the grid is frozen):

- **Selections sit on the grid's edge.** The hysteresis arm picked band 32, the largest offered, in
  15 of 16 cells; τ = 0.6, the highest, won 13 of 16 naive cells and 7 of 16 hysteresis cells. The
  best setting may lie outside the grid; the
  test run uses what the rule picked.
- **Frame skip helps on dev.** f1 rises with k for most configs (S hysteresis 0.46 → 0.50 at
  k = 3). A likely reason (untested): fewer observations per second give fewer chances for an anchor to jitter across the
  line. Whether that holds on MOT20 is a test question.

## 2026-10-25 — the test run (MOT20, once, dev-selected settings)

Ran once at `69a21e6` after committing the sha256 of every cache (`results/cache_manifest.csv`);
raw output committed unedited (`33e3985`), claims (`5c1f246`).

**H1 not supported, H2 not supported (reversed).** Reference S, k = 1: hysteresis f1 0.057, naive
0.089; paired difference −0.032 [−0.052, −0.015]. Net-count WAPE 0.94 and event error rate 1.02,
so `eer − 1.5·wape` = −0.39 [−0.44, −0.33]: the count is not hiding error behind an exact total,
it is simply far too low.

**Why: the detector, not the tracker or the counter.** Almost every error is a missed crossing
(`fn_missed`), not a switch or a duplicate. Checked before believing it:

- **MOT20-05 has zero predicted crossings on both lines** (1,082 true). Small's highest person score
  over all 3,315 frames is **0.59**; the dev-selected τ is 0.60, so ByteTrack never activates a
  track. Not a bug: the cache has 112 detections per frame (score p50 0.18, p90 0.31) against 195
  annotated people per frame. On MOT17-04 (dev) the same model's p90 is 0.75.
- MOT20-03: 3.9 % track recall (MOTA 0.03). MOT20-02: 23 %. MOT20-01: 29 %.
- MOT20 frames are 1,173–1,920 px wide with crowds of small, overlapping people; the detector sees
  them at 512 × 512. Confidence falls with density, and a threshold chosen on sparser dev scenes
  fails silently: no error, just no output.

This is the brief's anticipated null result in its strongest form: on held-out crowded video the
counting error is dominated by missed detections, and no tracker or line rule can recover a
person who was never detected. Hysteresis makes it slightly worse, probably because it also needs a track to
stay on each side of the band, which short, fragmented tracks rarely do.

Not run, and not claimed: a lower τ, tiled or higher-resolution inference, or a crowd-trained
detector. Each is the obvious next experiment and each would need its own pre-registration.

## 2026-10-25 — fact-check of the write-up

An independent pass checked about 130 statements in README, INTERVIEW, RESUME_BULLET and NOTES
against `results/`. All tables, intervals and speeds were right. Fixed: the claim count (three of
four not supported, not two), MOT20 frame widths (1,173–1,920 px), the best cell's input size
(Nano, 384 px), the int8 speed-up stated for Small only (Nano 1.27×), the equivalence threshold
caveat, Small's mean IoU (0.9998), and "the threshold decides everything", now stated as a
confounded observation. Mechanisms not tested on the test set are now marked as likely or
untested.

## 2026-10-25 — local CI record (hosted Actions not used: no repo yet)

Fresh clone of `9a5e6d6`, fresh install (`UV_NO_CACHE=1`), the workflow's steps in order:
`uv sync --all-groups` → heavy packages (torch, onnxruntime, rfdetr) absent: pass →
`ruff check` + `ruff format --check`: clean (38 files) → `pytest -m "not integration" --cov=src`:
**174 passed, 3 skipped, 10 deselected, coverage 81 %** → `vidan --help` runs from the installed
entry point. Later commits change docs only. Full suite with the vision extra in the working copy:
172 passed earlier the same day (integration tests included). Differences from the hosted runner:
Windows 11 instead of Ubuntu, `--basetemp` in a scratch folder, `-p no:cacheprovider`. Hygiene:
0 matches for assistant attribution in history, no tracked tool files, secrets, weights, caches
or video; one author on every commit.

## 2026-11-01 — local CI record for `8779434` (hosted CI unavailable)

Hosted CI unavailable (Actions limit until ~2026-11-02); no repo was pushed yet. Fresh clone, checked
out at `8779434`, fresh install (`UV_NO_CACHE=1`), Windows 11, Python 3.12.13, uv 0.11.19. The
workflow's steps in order:

- `uv python install` → 3.12 already present
- `uv sync --all-groups` → pass (no `vision` extra)
- heavy packages (torch, onnxruntime, rfdetr) absent: pass
- `uv run ruff check .` → all checks passed; `uv run ruff format --check .` → 38 files already formatted
- `uv run pytest -m "not integration" --cov=src --cov-report=term-missing --basetemp=<scratch>\pt -p no:cacheprovider`
  → **174 passed, 3 skipped, 10 deselected, coverage 81 %** (45 s). Same numbers as the `9a5e6d6` record;
  the commits since then change docs only.
- `uv run vidan --help` runs from the installed entry point (extra check, not a workflow step).

Differences from the hosted runner: Windows 11 instead of ubuntu-latest, `--basetemp` in a scratch
folder and `-p no:cacheprovider` (sandbox blocks the default temp folder), `UV_NO_CACHE=1` instead of
the setup-uv cache. No step was skipped. Hygiene: 0 matches for assistant attribution in history, one
author on every commit, no tracked tool files, `.env`, weights, video or caches, and no `C:\Users\fever`
paths in tracked files.
