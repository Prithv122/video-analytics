# Evaluation plan — frozen before any test-sequence run

**Frozen:** 2026-10-25, in the commit that adds this file. No detector has run on a test sequence
at that commit. Changes after it are dated amendments at the bottom, written before the runs
they govern.

## Questions

1. How far off is a line count from an off-the-shelf detector + ByteTrack, and how much error does
   a net count hide (a missed crossing and a double count cancel)?
2. Does a hysteresis band on the line beat the naive rule, and how much of the error is caused by
   identity switches near the line?
3. What does each CPU speed-up buy in frames per second, and what does it cost in counting accuracy?

## Data

| Split | Sequences | Frames | Use |
|---|---|---|---|
| dev | MOT17-02, MOT17-04, MOT17-09 (train, `-FRCNN` copy; static camera) | 2,175 at 30 fps | all tuning and every choice below |
| test | MOT20-01, MOT20-02, MOT20-03, MOT20-05 (train; static camera) | 8,931 at 25 fps | scored once per arm, after tuning |

MOT17's moving-camera sequences (05, 10, 11, 13) are excluded: a line fixed in the image is not
fixed in the world. Test sequences are different places, cameras and densities from dev; that
domain shift is part of what is measured, and the README says so. Licence for both: CC BY-NC-SA
3.0; no data is redistributed.

## Ground truth

- GT rows with `consider == 1` and `class == 1` (pedestrian), **any visibility**: a person crosses
  the line whether or not they are occluded at that moment.
- Truth crossings: GT tracks counted with the **naive** rule, anchor `bottom_center`.
- Lines (`configs/lines.json`, mechanical rule, fixed before any prediction): per sequence a
  horizontal line at `y = H/2` and a vertical line at `x = W/2`, each spanning the frame. Two
  directions per line → dev 12 cells, test 16 cells.

## Matching

- **Primary: identity-aware.** A predicted crossing matches a true one if the direction agrees,
  `|Δframe| ≤ 1 s` (30 frames dev, 25 test), and the predicted track's majority GT identity over that
  window (IoU ≥ 0.5 per-frame assignment) is the person who crossed. One-to-one, Hungarian.
- Secondary: time-only matching (same tolerance, identities ignored), reported beside it.

## Metrics (pooled over all cells of a split)

| Metric | Definition |
|---|---|
| `f1`, `precision`, `recall` | identity-aware event matching |
| `f1_time` | time-only matching |
| `wape` | Σ\|pred − true\| / Σ true over cells (net count error) |
| `eer` | event error rate = (FP + FN) / Σ true |
| `hidden` | `eer − wape`: error a net count does not show |
| attribution | FP: duplicate / spurious / unmatched; FN: tracked-not-counted / fragmented / missed |
| `switch_share` | (duplicate + fragmented) / (FP + FN) |
| MOT | MOTA, IDF1, ID switches per sequence (`motmetrics`, IoU 0.5), reported, not claimed |

**Intervals.** Cluster bootstrap over (sequence, 10-second block) of the test set, 2,000
resamples, seed 0, percentile 95 %. Differences between arms are paired (same resampled blocks).

## Arms

**Detector configs** (COCO-pretrained RF-DETR, never fine-tuned; person class only; ONNX Runtime
on CPU):

| Id | Model | Input | Precision |
|---|---|---|---|
| `S` | Small | 512 | fp32 — accuracy reference |
| `N` | Nano | 384 | fp32 |
| `S8` | Small | 512 | int8 (dynamic quantisation of weights) |
| `N8` | Nano | 384 | int8 |

RF-DETR Medium is left out: on this shared 6-core laptop CPU a Medium pass over 11,106 frames
would take most of a day per config, and the question is the cheap end of the budget. Decided
before any run.

Detections are cached once per (config, sequence) at score ≥ 0.10, so the tracker's low-score
second association has something to work with; every threshold below is applied on replay.

PyTorch eager runs only in the speed benchmark and in an equivalence check (ONNX fp32 vs PyTorch
detections on 50 dev frames).

**Frame skip** `k ∈ {1, 2, 3, 5}`: the detector and tracker see every k-th frame (ByteTrack frame
rate divided by k); crossings carry the original frame numbers. Replayed from the detection cache.

**Counting rule:** naive and hysteresis.

## Tuning (dev only)

For each (config, k): high-score threshold τ ∈ {0.3, 0.4, 0.5, 0.6}, used as both ByteTrack's
`track_activation_threshold` and its `high_conf_det_threshold` (cached detections from 0.10 to τ
only extend existing tracks); band ∈ {8, 16, 32} px for the hysteresis arm. ByteTrack
(`trackers` 2.6.1) otherwise at its library defaults: `lost_track_buffer` 30 frames at 30 fps
(scaled by the processed frame rate), `minimum_consecutive_frames` 2, `minimum_iou_threshold` 0.1.
Selection: highest pooled dev `f1`; ties → higher threshold, then smaller band.
The **naive arm** gets its own best threshold (band 0). Selected values are written to
`results/dev_selection.csv` and committed before the test run.

## Speed benchmark

**Quiet-machine gate.** This machine is shared. Before each repeat the harness samples
system-wide CPU use for 10 s; it starts only if the mean is below 15 % and records the reading.
A run forced past the gate is labelled `contended` and cannot evaluate H3 or H4.

100 frames of MOT17-04 (dev, 1920×1080), 10 warm-up frames, one process, all 12 logical cores
(`torch.set_num_threads(12)`, ORT `intra_op_num_threads = 12`), three repeats; the median of the
three runs' mean latencies is reported, with p50/p95 of the per-frame latencies. Detector FPS
includes preprocessing and decoding. Pipeline FPS at skip k = `k / (t_detect + t_track_count)`
per processed frame, with tracking and counting timed on the cached detections. The machine
fingerprint is written next to every timing.

## Pre-registered claims (test set)

- **H1 — a net count hides error.** Reference config `S`, k = 1, hysteresis arm:
  `eer ≥ 1.5 × wape`. Supported iff the bootstrap lower bound of `eer − 1.5·wape` is > 0.
- **H2 — hysteresis beats naive.** Config `S`, k = 1: `f1(hysteresis) − f1(naive) > 0`. Supported iff
  the paired lower bound is > 0.
- **H3 — int8 is a free speed-up.** Config `S8` vs `S`, k = 1: detector FPS ratio ≥ 1.3 **and** the
  paired upper bound of `f1(S) − f1(S8)` ≤ 0.03. Supported iff both hold.
- **H4 — ONNX Runtime beats eager PyTorch.** For Nano and Small fp32: ONNX FPS ≥ 1.5 × PyTorch FPS,
  with the equivalence check passing (matched boxes IoU ≥ 0.99, counts equal on ≥ 95 % of frames).
  Supported iff it holds for both models.

Each claim is evaluated mechanically by `vidan eval --claims` and reported whatever the outcome.

## Reported, not claimed

- The FPS–F1 frontier over all configs × k, and the cheapest point within 0.02 `f1` of `S`, k = 1.
- `switch_share` and the full attribution table per config.
- Per-sequence results (MOT20-03 and -05 are the densest), MOTA/IDF1/IDSW.
- Dev results, labelled as tuning numbers.

## What would not count

- Any test run of a config, k or threshold not selected on dev by the rule above.
- Moving a line, changing the tolerance or the IoU threshold after a test run.
- Timings from a different machine or thread count presented beside these ones.

## Amendments

### A1 — 2026-10-25, before any dev tuning or test scoring

Found by an independent code review; no dev grid or test number existed when this was written.
The only real result at this point is `results/equivalence.csv` (ONNX vs PyTorch boxes, which
none of these changes touch).

1. **Ground-truth boxes are clipped to the frame** (`[0, W] × [0, H]`) before counting, identity
   assignment and MOT metrics, as detections already are. Without it a GT foot anchor below the
   frame projects off the vertical line's segment and truth misses crossings that the clipped
   prediction counts (MOT20-05 `v_mid`: 359 raw vs 391 clipped truth crossings). GT rows whose
   clipped box has zero area are dropped.
2. **Per-frame IoU assignment thresholds before solving**: pairs below IoU 0.5 get a prohibitive
   cost instead of being dropped after the Hungarian step, which could trade a valid match for an
   invalid one in dense frames. Same rule `mot_eval` already used.
3. **Pipeline FPS at frame skip k** is computed from the k = 1 benchmark row, as the "Speed
   benchmark" section defines it: `k / (t_detect + t_track_count)`.
4. **H3 uses the hysteresis arm** of `S` and `S8` (the plan named an arm for H1 and H2 but not for
   H3).
5. **One test run.** `vidan eval test` refuses subset flags and refuses to run if test result files
   already exist; partial tuning runs must write outside `results/`. A cache whose frames differ
   from the sequence length is an error, and bad frames are reported per sequence.
6. MOTA / IDF1 here keep predictions matched to non-pedestrian GT (the MOTChallenge devkit removes
   them), so they are not comparable to leaderboard numbers. They were already "reported, not
   claimed".
