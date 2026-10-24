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
equivalence clause of H4 **fails for both models**: one Nano box and a handful of Small boxes
(two frames of MOT17-02) fall below it, all low-score boxes. The plan does not name the threshold
for this check. At 0.30 (`equivalence_thr0.30.csv`) Nano passes (min 0.9946) and Small still fails
(min 0.9805), so H4, which needs both models, fails under either reading. The rule is not being
relaxed after the fact; the percentile columns are descriptive only. The two backends agree on
the typical box to four decimal places and disagree on a few borderline boxes, which is what a
different summation order in fp32 does to a box sitting near a score threshold.
