# video-analytics — H4

**Tier:** 3 · **Category:** H — Computer vision · **Wave:** 4

The project's own engineering guide and the contract every module is built against.

## What this is

Detect → track → count on fixed-camera video, scored against human-annotated ground truth on
held-out sequences, with the CPU inference budget measured rather than assumed. Three questions:

1. **Counting.** How far off is a line count from a detector + ByteTrack pipeline, and does a
   net count hide errors that cancel (a missed crossing plus a double count reads as "exact")?
2. **ID switches.** How much of the counting error is caused by identity switches near the line,
   and does a hysteresis band on the line remove it?
3. **Inference budget.** What does each CPU speed-up (ONNX Runtime, int8, lower input resolution,
   detecting every k-th frame) buy in frames per second, and what does it cost in counting accuracy?

The evaluation plan (`EVAL_PLAN.md`) is frozen before any detector runs on a test sequence.

## Stack

Python 3.12 · uv · `rfdetr` (Apache-2.0; sizes N/S/M only — XL/2XL are PML 1.0, excluded) ·
`trackers` (Apache-2.0; ByteTrack moved there from `supervision`) with `supervision` (MIT)
detection types · `onnxruntime` (MIT) on CPU · `motmetrics` (MIT) for MOTA /
IDF1 / ID switches · numpy · pytest · ruff. **No Ultralytics** (AGPL-3.0).

## Non-negotiables

- The core (`mot`, `geometry`, `counting`, `metrics`) is numpy-only: no torch, no onnxruntime, no
  supervision imports. It runs in CI without model weights or video.
- Ground-truth crossings come from the annotated GT tracks, through the **same** counting code
  as predictions, with the same line. Counting-rule differences can't masquerade as accuracy.
  Truth is always counted with the `naive` rule (every crossing of an annotated anchor); the rule
  under test applies to predictions only, so a band that suppresses a real back-and-forth is an error.
- Lines and zones are placed by a mechanical rule from frame size and GT only, written down before
  any prediction exists on that sequence. No line is moved after seeing a result.
- Every number in the README comes from a committed script and a committed results file.
- Detections are cached per (sequence, detector config); tracking and counting replay from the
  cache, so a counting experiment never re-runs the detector.

## Data contract

**MOT rows** (MOTChallenge text format, 1-based frames):
`frame, id, x, y, w, h, conf, class, visibility` — `x, y` top-left, pixels, float.
`read_mot(path, *, gt=False)` returns a `Tracks` value (column arrays, sorted by `(frame, id)`).
With `gt=True` it keeps only rows with `conf == 1` (MOT "consider" flag) and `class in {1}`
(pedestrian) unless `classes=` says otherwise; `min_visibility` filters on column 9.

`Tracks` (frozen dataclass of equal-length numpy arrays):

| Field | dtype | Notes |
|---|---|---|
| `frame` | int64 | 1-based |
| `track_id` | int64 | > 0 |
| `xywh` | float64 (n, 4) | top-left x, y, width, height |
| `score` | float64 | 1.0 for GT |

Helpers: `Tracks.anchor(kind)` → (n, 2) points: `"bottom_center"` (default, feet) or `"center"`;
`Tracks.by_id()` → dict id → row indices in frame order; `len(tracks)`.

**Line** = two points `(x1, y1), (x2, y2)`. The side of a point is the sign of the cross product
`(p2 - p1) × (pt - p1)`; positive side = "in". Signed distance = cross / |p2 - p1|.
A crossing only counts if the anchor's projection falls within the segment (with no extension).

**Zone** = polygon (list of points), point-in-polygon by ray casting, boundary counts as inside.

## Counting semantics

For each track, walk its anchors in frame order (gaps allowed). Two rules, same interface:

- **`naive`** (supervision `LineZone` behaviour): a crossing is any step where the anchor's side
  changes between consecutive observations of the same id, with both points' projections inside
  the segment. Every crossing counts, so an id that wobbles on the line counts repeatedly.
- **`hysteresis(band_px)`**: a track's state is `in`, `out` or unknown. It enters a state only when
  its signed distance is beyond `+band_px` (in) or `-band_px` (out). A crossing is a state change
  `out → in` or `in → out`; time inside the band does not change state.
- **Shared details.** Both rules are one state machine and `naive` is `band_px = 0`, so
  `hysteresis(0)` reproduces `naive` exactly (tested). A point exactly on the line has no side: it
  keeps the previous state. An observation whose projection falls outside the segment resets the
  state to unknown, so a crossing needs the state-setting observation and the completing
  observation to both project inside the segment; a path that goes around the end of the segment
  never counts. Gaps in a track are fine.

A crossing event is `Crossing(track_id, frame, direction)` where `direction` is `+1` (to the in
side) or `-1`, and `frame` is the frame of the observation that completed it.
`count_line(tracks, line, rule) -> LineCount(n_in, n_out, events)`.
`zone_occupancy(tracks, zone) -> dict[frame, int]` (anchors inside the zone per frame).

## Metrics (`metrics.py`)

- **Net count error** per line and direction: `pred − gt`. Aggregates: MAE, and WAPE =
  Σ|pred − gt| / Σ gt.
- **Event matching**: predicted vs GT crossings of one line, same direction, matched one-to-one
  by Hungarian assignment on `|frame difference|` with a tolerance `tol_frames` (default 1 s of
  video). Gives TP / FP / FN, precision, recall, F1. Net count can be exact while F1 < 1: that gap
  is "cancelled error" = `min(FP, FN)` and is reported per line.
- **Switch-attributed error**: an FP whose track id was born within `near_px` of the line or whose
  GT-matched identity already has a TP crossing in that direction (needs the MOT matching from
  `motmetrics`) — defined precisely in `EVAL_PLAN.md`.

## Module map

| Module | Public API |
|---|---|
| `mot.py` | `Tracks`, `read_mot`, `write_mot`, `tracks_from_rows` |
| `geometry.py` | `Line`, `Zone`, `side`, `signed_distance`, `projects_onto`, `point_in_polygon` |
| `counting.py` | `Crossing`, `LineCount`, `Rule` (`naive()`, `hysteresis(band_px)`), `count_line`, `zone_occupancy` |
| `metrics.py` | `count_errors`, `match_events`, `EventMatch`, `wape`, `mae` |
| `layout.py` | per-sequence lines/zones from the placement rule; reads/writes `configs/lines.json` |
| `data.py` | dataset download + checksum + sequence index (`seqinfo.ini`), frame iterator |
| `detect.py` | `Detector` protocol, `TorchRFDETR`, `OnnxRFDETR`, detection cache (`.npz`) |
| `track.py` | ByteTrack over cached detections → `Tracks`; frame-skip `k` |
| `export.py` | ONNX export from rfdetr, int8 dynamic quantisation |
| `bench.py` | latency/FPS harness with warm-up and an environment fingerprint |
| `mot_eval.py` | `motmetrics` wrapper: MOTA, IDF1, IDSW per sequence |
| `evalrun.py` | the frozen grid from `EVAL_PLAN.md` → `results/*.csv` |
| `cli.py` | `vidan fetch | detect | track | count | bench | eval` |

## Acceptance criteria

- [x] Hand-computed fixture tests for both counting rules (clean crossing, wobble on the line,
      crossing outside the segment, gap in a track, `band_px = 0` equals naive)
- [x] Event matching tested on fixtures where net count is exact but F1 < 1
- [x] `EVAL_PLAN.md` frozen before any test-sequence run; claims evaluated mechanically
- [x] CPU benchmark: PyTorch vs ONNX fp32 vs ONNX int8 at each model's input size, same machine, same threads
- [ ] Runs from a clean clone using only the documented steps; tests, lint and CI green

## Conventions

- `uv` for dependencies; `ruff check` + `ruff format`; `pytest`.
- Tests that need weights, video or onnxruntime sessions are marked `integration` and deselected in CI.
- Local pytest needs `--basetemp` pointed outside `%TEMP%` on this machine (not in `pyproject.toml`).

## Project-specific notes

- Data lives in `data/raw/` (git-ignored). Detection caches in `data/cache/` (git-ignored).
- GPU is never used. Every timing is CPU, single process, with the thread count recorded.
