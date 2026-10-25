"""The frozen evaluation grid of `EVAL_PLAN.md`: tune on dev, test once, bootstrap, claims.

Pure numpy, csv and subprocess (git); tracking is injected through `Sources.track` so everything
here imports without the `vision` extra. Only the default tracker needs it.

Flow
----
`tune`      dev grid -> `dev_grid.csv`, `dev_selection.csv` (selection rule from the plan).
`test`      refuses unless `dev_selection.csv` is committed and unmodified; scores only the
            selected settings -> `test_cells.csv`, `test_events.csv`, `test_sequences.csv`,
            `test_mot.csv`.
`claims`    H1..H4 mechanically -> `claims.json` and a table.
`frontier`  F1 vs pipeline FPS per config x k, reported not claimed -> `frontier.csv`.

Cells and pooling
-----------------
A cell is (sequence, line, direction): 12 on dev, 16 on test. Counts are pooled over all cells of
a split. `f1`, `precision`, `recall` use identity-aware matching; `f1_time` ignores identities;
`wape` = sum over cells |pred - true| / sum true; `eer` = (fp + fn) / sum true;
`hidden` = eer - wape; `switch_share` = (duplicate + fragmented) / (fp + fn).

Bootstrap
---------
Cluster bootstrap over (sequence, 10-second block) with `block = (frame - 1) // round(10 * fps)`.
The population is every block of every test sequence (also blocks with no events). One resample
draws as many blocks as there are with replacement, giving integer weights `w_b`. Per resample:
tp, fp, fn and the true and predicted counts are the weighted sums over blocks; an event belongs
to the block of its own frame (a matched pair, to the block of the true crossing). WAPE is on net
counts per cell: sum over cells |sum_b w_b (pred_cb - true_cb)| / sum true, with the cell sums
taken over the resampled blocks, so a cell's misses and extras cancel inside the resample exactly
as they do in a net count. Paired differences use the same weights. Intervals are the 2.5th and
97.5th percentiles of 2,000 resamples (seed 0).

Result-file schemas read here (written by the benchmark code)
-------------------------------------------------------------
`bench.csv`: `config` (S, N, S8, N8), `backend` (onnx | torch), `detector_fps`, `contended`
(true/false), optionally `k`, `t_track_count`, `pipeline_fps`. PyTorch rows use the fp32 config id
(S or N).
`equivalence.csv`: `config` or `model` (N/S or nano/small), `min_matched_iou`,
`frac_frames_equal_count`, or a boolean `passed`.
"""

import contextlib
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vidan.attribution import FN_CATEGORIES, FP_CATEGORIES, attribute_errors, frame_assignment
from vidan.counting import LineCount, Rule, count_line
from vidan.geometry import Line
from vidan.metrics import match_events
from vidan.mot import Tracks

CONFIGS = ("S", "N", "S8", "N8")
KS = (1, 2, 3, 5)
TAUS = (0.3, 0.4, 0.5, 0.6)
HYSTERESIS_BANDS = (8, 16, 32)
BANDS = (0, *HYSTERESIS_BANDS)
DIRECTIONS = (1, -1)
BLOCK_SECONDS = 10
N_RESAMPLES = 2000
SEED = 0
CI_LEVEL = 95.0
H1_FACTOR = 1.5
H2_MARGIN = 0.0
H3_MIN_FPS_RATIO = 1.3
H3_MAX_F1_DROP = 0.03
H4_MIN_FPS_RATIO = 1.5
H4_MIN_IOU = 0.99
H4_MIN_FRAC_EQUAL = 0.95
FRONTIER_TOLERANCE = 0.02

RESULTS = Path("results")
DEV_GRID = "dev_grid.csv"
DEV_SELECTION = "dev_selection.csv"
TEST_CELLS = "test_cells.csv"
TEST_EVENTS = "test_events.csv"
TEST_SEQUENCES = "test_sequences.csv"
TEST_MOT = "test_mot.csv"
BENCH = "bench.csv"
EQUIVALENCE = "equivalence.csv"
CLAIMS = "claims.json"
FRONTIER = "frontier.csv"
SUMMARY = "test_summary.csv"
BY_SEQUENCE = "test_by_sequence.csv"
DETECTOR_SCORES = "test_detector_scores.csv"
SUMMARY_CI_METRICS = ("f1", "recall", "precision", "wape", "eer")

CELL_FIELDS = (
    "config seq line direction k tau band arm n_bad_frames n_true n_pred tp fp fn tp_time "
    + " ".join(f"fp_{c}" for c in FP_CATEGORIES)
    + " "
    + " ".join(f"fn_{c}" for c in FN_CATEGORIES)
).split()
EVENT_FIELDS = [
    "config",
    "seq",
    "line",
    "direction",
    "k",
    "tau",
    "band",
    "arm",
    "kind",
    "frame",
    "block",
    "outcome",
    "category",
]
_INT_COLS = {
    "n_bad_frames",
    "direction", "k", "band", "n_true", "n_pred", "tp", "fp", "fn", "tp_time", "frame", "block",
    *(f"fp_{c}" for c in FP_CATEGORIES),
    *(f"fn_{c}" for c in FN_CATEGORIES),
}  # fmt: skip
_FLOAT_COLS = {"tau", "f1", "precision", "recall", "f1_time", "wape", "eer", "hidden", "dev_f1"}


def _config_rank(config: str) -> tuple[int, str]:
    return (CONFIGS.index(config) if config in CONFIGS else len(CONFIGS), config)


def arm_of(band: float) -> str:
    return "naive" if band == 0 else "hysteresis"


def rule_of(band: float) -> Rule:
    return Rule.naive() if band == 0 else Rule.hysteresis(band)


# ----------------------------------------------------------------------------- sources + scoring


@dataclass
class Sources:
    """Everything the scorer reads, as callables so tests can supply synthetic data."""

    detections: Callable[[str, str], dict]  # (config, seq) -> {frame: Detections}
    gt: Callable[[str], Tracks]  # seq -> ground-truth tracks (consider == 1, pedestrian)
    fps: Callable[[str], float]
    n_frames: Callable[[str], int]
    size: Callable[[str], tuple[float, float]]  # seq -> (width, height); boxes are clipped to it
    lines: Callable[[str], dict[str, Line]]
    track: Callable[[dict, float, int, float], Tracks]  # (per_frame, fps, k, tau) -> Tracks
    n_bad_frames: Callable[[str, str], int | None] | None = None  # (config, seq) -> from cache meta


def check_frames(per_frame: dict, seq_length: int, label: str) -> None:
    """Raise unless the cache holds exactly the frames 1..seq_length."""
    have, want = set(per_frame), set(range(1, seq_length + 1))
    if have != want:
        missing, extra = sorted(want - have), sorted(have - want)
        raise ValueError(
            f"{label}: cache frames differ from 1..{seq_length} "
            f"({len(missing)} missing, first {missing[:3]}; {len(extra)} extra, first {extra[:3]})"
        )


def default_sources(
    cache_root: Path = Path("data/cache"),
    raw_root: Path = Path("data/raw"),
    layout_path: Path = Path("configs/lines.json"),
) -> Sources:
    """Caches under `<cache_root>/<config>/<seq>.npz`, data under `raw_root`, ByteTrack."""
    from vidan import data
    from vidan.detect import PERSON, load_cache
    from vidan.layout import read_layout
    from vidan.mot import read_mot

    layout: dict = {}
    infos: dict[str, Any] = {}
    bad: dict[tuple[str, str], int | None] = {}

    def info(seq: str):
        if seq not in infos:
            infos[seq] = data.read_seqinfo(data.seq_dir(seq, raw_root))
        return infos[seq]

    def detections(config: str, seq: str) -> dict:
        per_frame, meta = load_cache(Path(cache_root) / config / f"{seq}.npz")
        check_frames(per_frame, info(seq).seq_length, f"{config}/{seq}")
        bad[(config, seq)] = meta.get("n_bad_frames")
        return {
            f: (d if bool((d.class_id == PERSON).all()) else d.keep(d.class_id == PERSON))
            for f, d in per_frame.items()
        }

    def gt(seq: str) -> Tracks:
        return read_mot(data.seq_dir(seq, raw_root) / "gt" / "gt.txt", gt=True)

    def lines(seq: str) -> dict[str, Line]:
        if not layout:
            layout.update(read_layout(layout_path))
        return layout[seq]["lines"]

    def track(per_frame: dict, fps: float, k: int, tau: float) -> Tracks:
        from vidan.track import run_bytetrack

        return run_bytetrack(
            per_frame, fps, k, track_activation_threshold=tau, high_conf_det_threshold=tau
        )

    return Sources(
        detections=detections,
        gt=gt,
        fps=lambda s: float(info(s).frame_rate),
        n_frames=lambda s: int(info(s).seq_length),
        size=lambda s: (float(info(s).width), float(info(s).height)),
        lines=lines,
        track=track,
        n_bad_frames=lambda c, s: bad.get((c, s)),
    )


@dataclass
class CellScore:
    """Rows (one per line and direction) and event-level rows of one scored setting."""

    rows: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class Scorer:
    """Scores (config, seq, k, tau, band) settings; tracking is cached per (config, seq, k, tau).

    Only the most recent detection set and tracking result are kept (MOT20 tracks are large), so
    loop with the sequence outermost and the setting innermost. A different band re-counts but
    never re-tracks. Ground truth and its naive crossings are cached for the whole run.
    """

    def __init__(self, src: Sources, iou_thr: float = 0.5, anchor: str = "bottom_center"):
        self.src = src
        self.iou_thr = iou_thr
        self.anchor = anchor
        self._gt: dict[str, Tracks] = {}
        self._gt_counts: dict[tuple[str, str], LineCount] = {}
        self._det_key: tuple[str, str] | None = None
        self._det: dict = {}
        self._trk_key: tuple[str, str, int, float] | None = None
        self._trk: tuple[Tracks, dict] | None = None
        self._bad: dict[tuple[str, str], int | None] = {}
        self.track_seconds = 0.0
        self.n_tracked = 0

    def gt(self, seq: str) -> Tracks:
        if seq not in self._gt:
            self._gt[seq] = self.src.gt(seq).clip(*self.src.size(seq))
        return self._gt[seq]

    def gt_count(self, seq: str, line_name: str) -> LineCount:
        key = (seq, line_name)
        if key not in self._gt_counts:
            line = self.src.lines(seq)[line_name]
            self._gt_counts[key] = count_line(self.gt(seq), line, Rule.naive(), self.anchor)
        return self._gt_counts[key]

    def tracks(self, config: str, seq: str, k: int, tau: float) -> tuple[Tracks, dict]:
        """Tracks and the (frame, pred_id) -> gt_id assignment for one tracked setting."""
        key = (config, seq, k, tau)
        if self._trk_key != key:
            if self._det_key != (config, seq):
                self._det = self.src.detections(config, seq)
                self._det_key = (config, seq)
                if self.src.n_bad_frames is not None:
                    self._bad[(config, seq)] = self.src.n_bad_frames(config, seq)
            t0 = time.perf_counter()
            tracks = self.src.track(self._det, self.src.fps(seq), k, tau).clip(*self.src.size(seq))
            self.track_seconds += time.perf_counter() - t0
            self.n_tracked += 1
            self._trk = (tracks, frame_assignment(tracks, self.gt(seq), self.iou_thr))
            self._trk_key = key
        assert self._trk is not None
        return self._trk

    def score_cell(
        self, config: str, seq: str, k: int, tau: float, band: float, events: bool = False
    ) -> CellScore:
        """Score one setting on one sequence: a row per (line, direction); events optional."""
        tracks, assign = self.tracks(config, seq, k, tau)
        fps = self.src.fps(seq)
        tol = round(fps)
        block_len = round(BLOCK_SECONDS * fps)
        gt = self.gt(seq)
        out = CellScore()
        for line_name, line in sorted(self.src.lines(seq).items()):
            att = attribute_errors(
                tracks,
                gt,
                line,
                rule_of(band),
                tol,
                self.iou_thr,
                self.anchor,
                assignment=assign,
                gt_count=self.gt_count(seq, line_name),
            )
            matched_pred = {(p.track_id, p.frame, p.direction) for p, _ in att.match.pairs}
            matched_true = {(g.track_id, g.frame, g.direction) for _, g in att.match.pairs}
            fp_cat = {(e.track_id, e.frame, e.direction): c for e, c in att.fp_events}
            fn_cat = {(e.track_id, e.frame, e.direction): c for e, c in att.fn_events}
            for d in DIRECTIONS:
                preds = [e for e in att.pred_events if e.direction == d]
                trues = [e for e in att.gt_events if e.direction == d]
                tp = sum(p.direction == d for p, _ in att.match.pairs)
                row = {
                    "config": config, "seq": seq, "line": line_name, "direction": d, "k": k,
                    "tau": tau, "band": band, "arm": arm_of(band),
                    "n_bad_frames": self._bad.get((config, seq)),
                    "n_true": len(trues), "n_pred": len(preds), "tp": tp,
                    "fp": len(preds) - tp, "fn": len(trues) - tp,
                    "tp_time": match_events(preds, trues, tol).tp,
                }  # fmt: skip
                for c in FP_CATEGORIES:
                    row[f"fp_{c}"] = sum(
                        1 for e, k_ in att.fp_events if e.direction == d and k_ == c
                    )
                for c in FN_CATEGORIES:
                    row[f"fn_{c}"] = sum(
                        1 for e, k_ in att.fn_events if e.direction == d and k_ == c
                    )
                out.rows.append(row)
                if not events:
                    continue
                base = {
                    "config": config, "seq": seq, "line": line_name, "direction": d, "k": k,
                    "tau": tau, "band": band, "arm": arm_of(band),
                }  # fmt: skip
                for e in preds:
                    key = (e.track_id, e.frame, e.direction)
                    ok = key in matched_pred
                    out.events.append(
                        {**base, "kind": "pred", "frame": e.frame,
                         "block": (e.frame - 1) // block_len,
                         "outcome": "tp" if ok else "fp", "category": "" if ok else fp_cat[key]}
                    )  # fmt: skip
                for e in trues:
                    key = (e.track_id, e.frame, e.direction)
                    ok = key in matched_true
                    out.events.append(
                        {**base, "kind": "true", "frame": e.frame,
                         "block": (e.frame - 1) // block_len,
                         "outcome": "tp" if ok else "fn", "category": "" if ok else fn_cat[key]}
                    )  # fmt: skip
        return out

    def mot(self, config: str, seq: str, k: int, tau: float) -> dict:
        """MOTA, IDF1, ID switches ... of a tracked setting (GT limited to the processed frames)."""
        from vidan.mot_eval import evaluate

        tracks, _ = self.tracks(config, seq, k, tau)
        gt = self.gt(seq)
        if k > 1:
            keep = (gt.frame - 1) % k == 0
            gt = Tracks(gt.frame[keep], gt.track_id[keep], gt.xywh[keep], gt.score[keep])
        return evaluate(gt, tracks, self.iou_thr)


def score_cell(
    scorer: Scorer, config: str, seq: str, k: int, tau: float, band: float, events: bool = False
) -> CellScore:
    """Track (cached) -> count with the rule -> match -> attribute, per line and direction."""
    return scorer.score_cell(config, seq, k, tau, band, events)


# ------------------------------------------------------------------------------ pooled metrics


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    """2tp / (2tp + fp + fn); 1.0 when there is nothing to find and nothing predicted."""
    den = 2 * tp + fp + fn
    return 1.0 if den == 0 else 2 * tp / den


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


_SUM_KEYS = (
    "n_true",
    "n_pred",
    "tp",
    "fp",
    "fn",
    "tp_time",
    *(f"fp_{c}" for c in FP_CATEGORIES),
    *(f"fn_{c}" for c in FN_CATEGORIES),
)


def pool(rows: Iterable[dict]) -> dict[str, float]:
    """Pooled metrics over cell rows (each row is one cell: sequence, line, direction)."""
    rows = list(rows)
    s = {key: sum(r[key] for r in rows) for key in _SUM_KEYS}
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    empty = s["n_true"] == 0 and s["n_pred"] == 0
    abs_net = sum(abs(r["n_pred"] - r["n_true"]) for r in rows)
    wape = _div(abs_net, s["n_true"])
    eer = _div(fp + fn, s["n_true"])
    out: dict[str, float] = {
        "f1": f1_from_counts(tp, fp, fn),
        "precision": _div(tp, tp + fp) if tp + fp else (1.0 if empty else 0.0),
        "recall": _div(tp, tp + fn) if tp + fn else (1.0 if empty else 0.0),
        "f1_time": f1_from_counts(
            s["tp_time"], s["n_pred"] - s["tp_time"], s["n_true"] - s["tp_time"]
        ),
        "wape": wape,
        "eer": eer,
        "hidden": eer - wape,
        "switch_share": _div(s["fp_duplicate"] + s["fn_fragmented"], fp + fn),
    }
    out.update(s)
    return out


def grid_rows(cells: Iterable[dict]) -> list[dict]:
    """One pooled row per (config, k, tau, band) from per-cell rows."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in cells:
        groups[(r["config"], r["k"], r["tau"], r["band"])].append(r)
    out = []
    for (config, k, tau, band), rs in sorted(groups.items()):
        bad = {r["seq"]: r.get("n_bad_frames") for r in rs}  # one value per sequence
        known = all(v not in (None, "") for v in bad.values())
        out.append(
            {
                "config": config,
                "k": k,
                "tau": tau,
                "band": band,
                "n_bad_frames": sum(bad.values()) if known else "",
                **pool(rs),
            }
        )
    return out


def select(grid: Sequence[dict]) -> list[dict]:
    """The plan's selection rule, per (config, k).

    Highest pooled dev `f1`; ties go to the higher threshold, then the smaller band. The naive
    arm uses band 0 only; the hysteresis arm picks over its bands (8, 16, 32).
    """
    out = []
    for config, k in sorted({(g["config"], g["k"]) for g in grid}):
        here = [g for g in grid if g["config"] == config and g["k"] == k]
        for arm in ("naive", "hysteresis"):
            cands = [g for g in here if arm_of(g["band"]) == arm]
            if not cands:
                continue
            best = max(cands, key=lambda g: (round(g["f1"], 12), g["tau"], -g["band"]))
            out.append(
                {"arm": arm, "config": config, "k": k, "tau": best["tau"], "band": best["band"],
                 "dev_f1": best["f1"]}
            )  # fmt: skip
    return out


# ------------------------------------------------------------------------------------ csv + git


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields) if fields else list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def read_csv(path: Path) -> list[dict]:
    """Rows with known numeric columns parsed (ints, floats); everything else stays a string."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key, val in r.items():
            if val is None or val == "":
                continue
            if key in _INT_COLS:
                r[key] = int(float(val))
            elif key in _FLOAT_COLS or key.endswith(("_lo", "_hi")):
                # bench.csv's `precision` is fp32/int8, not the metric of the same name
                with contextlib.suppress(ValueError):
                    r[key] = float(val)
    return rows


class RepeatTestRun(RuntimeError):
    """Test result files exist already: a second look at the test set must be deliberate."""


class SelectionNotCommitted(RuntimeError):
    """The dev selection is missing, untracked or edited since the last commit."""


def selection_is_committed(
    path: Path, run: Callable[..., Any] = subprocess.run
) -> tuple[bool, str]:
    """True iff `path` is tracked, present in HEAD and its working copy equals HEAD.

    Paths are interpreted relative to the current directory (the repository checkout).
    """
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(Path.cwd())
        except ValueError:
            return False, f"{path} is outside the working directory"
    rel = p.as_posix()
    checks = (
        (["git", "ls-files", "--error-unmatch", "--", rel], "is not tracked by git"),
        (["git", "cat-file", "-e", f"HEAD:./{rel}"], "is not in the HEAD commit"),
        (["git", "diff", "--quiet", "HEAD", "--", rel], "differs from the HEAD commit"),
    )
    for cmd, why in checks:
        try:
            r = run(cmd, capture_output=True, text=True)
        except OSError as exc:
            return False, f"git could not be run: {exc}"
        if r.returncode != 0:
            return False, f"{rel} {why}"
    return True, ""


# ------------------------------------------------------------------------------------------ tune


def tune(
    src: Sources,
    out_dir: Path = RESULTS,
    split: str = "dev",
    configs: Sequence[str] = CONFIGS,
    ks: Sequence[int] = KS,
    taus: Sequence[float] = TAUS,
    bands: Sequence[int] = BANDS,
    seqs: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
    scorer: Scorer | None = None,
) -> tuple[list[dict], list[dict]]:
    """Score the dev grid and write `dev_grid.csv` and `dev_selection.csv`.

    One row per (config, k, tau, band) with band 0 as the naive arm. Tracking is done once per
    (config, sequence, k, tau) and shared by every band.
    """
    if seqs is None:
        from vidan.data import names_for_split

        seqs = names_for_split(split)
    scorer = scorer or Scorer(src)
    cells: list[dict] = []
    t_start = time.perf_counter()
    points = 0
    for config in configs:
        for seq in seqs:
            for k in ks:
                for tau in taus:
                    t0 = time.perf_counter()
                    for band in bands:
                        cells += scorer.score_cell(config, seq, k, tau, band).rows
                        points += 1
                    log(
                        f"{config} {seq} k={k} tau={tau}: {len(bands)} bands in "
                        f"{time.perf_counter() - t0:.1f} s"
                    )
    grid = grid_rows(cells)
    selection = select(grid)
    write_csv(Path(out_dir) / DEV_GRID, grid)
    write_csv(Path(out_dir) / DEV_SELECTION, selection)
    log(
        f"dev grid: {len(grid)} rows, {points} seq-setting points, {scorer.n_tracked} trackings "
        f"({scorer.track_seconds:.0f} s tracking), {time.perf_counter() - t_start:.0f} s total"
    )
    return grid, selection


# ------------------------------------------------------------------------------------------ test


def test(
    src: Sources,
    out_dir: Path = RESULTS,
    split: str = "test",
    configs: Sequence[str] | None = None,
    seqs: Sequence[str] | None = None,
    log: Callable[[str], None] = print,
    check_committed: Callable[[Path], tuple[bool, str]] = selection_is_committed,
    scorer: Scorer | None = None,
) -> list[dict]:
    """Score only the settings in the committed `dev_selection.csv` on the test split.

    Raises `SelectionNotCommitted` unless the selection file exists, is tracked and equals HEAD.
    """
    sel_path = Path(out_dir) / DEV_SELECTION
    ok, why = (
        (False, f"{sel_path} does not exist")
        if not sel_path.exists()
        else check_committed(sel_path)
    )
    if not ok:
        raise SelectionNotCommitted(
            f"refusing to run the test split: {why}. Commit {sel_path} first."
        )
    existing = sorted(Path(out_dir).glob("test_*.csv"))
    if existing:
        raise RepeatTestRun(
            "refusing to run the test split again: "
            + ", ".join(p.name for p in existing)
            + " already exist. A second test-set exposure must be deliberate: delete them by hand."
        )
    selection = read_csv(sel_path)
    if configs is not None:
        selection = [s for s in selection if s["config"] in set(configs)]
    if seqs is None:
        from vidan.data import names_for_split

        seqs = names_for_split(split)
    scorer = scorer or Scorer(src)
    cells: list[dict] = []
    events: list[dict] = []
    mot_rows: list[dict] = []
    order = sorted({(s["config"], s["k"], s["tau"]) for s in selection})
    for config in sorted({c for c, _, _ in order}, key=_config_rank):
        for seq in seqs:
            for _, k, tau in [o for o in order if o[0] == config]:
                bands = sorted(
                    {
                        s["band"]
                        for s in selection
                        if (s["config"], s["k"], s["tau"]) == (config, k, tau)
                    }
                )
                for band in bands:
                    res = scorer.score_cell(config, seq, k, tau, band, events=True)
                    cells += res.rows
                    events += res.events
                mot_rows.append(
                    {
                        "config": config,
                        "seq": seq,
                        "k": k,
                        "tau": tau,
                        **scorer.mot(config, seq, k, tau),
                    }
                )
                log(f"test {config} {seq} k={k} tau={tau} bands={bands}")
    out_dir = Path(out_dir)
    write_csv(out_dir / TEST_CELLS, cells, CELL_FIELDS)
    write_csv(out_dir / TEST_EVENTS, events, EVENT_FIELDS)
    write_csv(out_dir / TEST_MOT, mot_rows)
    write_csv(
        out_dir / TEST_SEQUENCES,
        [
            {
                "seq": s,
                "fps": src.fps(s),
                "n_frames": src.n_frames(s),
                "block_frames": round(BLOCK_SECONDS * src.fps(s)),
            }
            for s in seqs
        ],
    )
    return cells


# -------------------------------------------------------------------------------------- bootstrap


@dataclass
class BlockTable:
    """Per-block event counts of one arm (config, k, arm) over the block universe.

    `tp`, `fp`, `fn`: (B,) counts by block. `pred`, `true`: (C, B) counts per cell and block.
    """

    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    pred: np.ndarray
    true: np.ndarray

    @property
    def n_blocks(self) -> int:
        return len(self.tp)


def block_universe(sequences: Iterable[dict]) -> list[tuple[str, int]]:
    """All (sequence, block) pairs from rows with `seq`, `n_frames`, `block_frames`."""
    out = []
    for s in sequences:
        n_blocks = math.ceil(int(s["n_frames"]) / int(s["block_frames"]))
        out += [(s["seq"], b) for b in range(n_blocks)]
    return out


def build_table(
    events: Iterable[dict], universe: Sequence[tuple[str, int]], config: str, k: int, arm: str
) -> BlockTable:
    """Aggregate event rows of one (config, k, arm) into per-block and per-cell-block counts."""
    index = {b: i for i, b in enumerate(universe)}
    rows = [e for e in events if e["config"] == config and int(e["k"]) == k and e["arm"] == arm]
    cells = sorted({(e["seq"], e["line"], int(e["direction"])) for e in rows})
    cidx = {c: i for i, c in enumerate(cells)}
    n_b = len(universe)
    tp, fp, fn = (np.zeros(n_b) for _ in range(3))
    pred, true = np.zeros((len(cells), n_b)), np.zeros((len(cells), n_b))
    for e in rows:
        b = index[(e["seq"], int(e["block"]))]
        c = cidx[(e["seq"], e["line"], int(e["direction"]))]
        if e["kind"] == "pred":
            pred[c, b] += 1
            if e["outcome"] == "fp":
                fp[b] += 1
        else:
            true[c, b] += 1
            if e["outcome"] == "tp":
                tp[b] += 1
            else:
                fn[b] += 1
    return BlockTable(tp, fp, fn, pred, true)


def resample_weights(n_blocks: int, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> np.ndarray:
    """(R, B) integer counts: how often each block is drawn in each resample (B draws each)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(n_resamples, n_blocks))
    w = np.zeros((n_resamples, n_blocks))
    rows = np.repeat(np.arange(n_resamples), n_blocks)
    np.add.at(w, (rows, draws.ravel()), 1.0)
    return w


def pooled_metrics(table: BlockTable, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Pooled f1, precision, recall, wape, eer, hidden for each weight row (see module doc)."""
    w = np.atleast_2d(weights)
    tp, fp, fn = w @ table.tp, w @ table.fp, w @ table.fn
    n_true = w @ table.true.sum(axis=0)
    n_pred = w @ table.pred.sum(axis=0)
    net = w @ (table.pred - table.true).T  # (R, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        wape = np.where(n_true > 0, np.abs(net).sum(axis=1) / n_true, np.nan)
        eer = np.where(n_true > 0, (fp + fn) / n_true, np.nan)
        empty = (n_true == 0) & (n_pred == 0)
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.where(empty, 1.0, 0.0))
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.where(empty, 1.0, 0.0))
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 1.0)
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "wape": wape, "eer": eer, "hidden": eer - wape,
    }  # fmt: skip


def interval(x: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    """Percentile interval, ignoring nan resamples."""
    lo = (100 - level) / 2
    a, b = np.nanpercentile(x, [lo, 100 - lo])
    return float(a), float(b)


def lower_bound_positive(x: np.ndarray, level: float = CI_LEVEL) -> bool:
    """True iff the interval's lower bound is strictly above 0."""
    return interval(x, level)[0] > 0


# ------------------------------------------------------------------------------------- claims


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "t"}


def _num(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _maybe_read(path: Path) -> list[dict] | None:
    return read_csv(path) if Path(path).exists() else None


def _bench_fps(bench: Sequence[dict], config: str, backend: str) -> tuple[float | None, bool]:
    """Detector FPS of (config, backend) and whether any matching row is contended."""
    rows = [
        r for r in bench
        if r.get("config") == config and (r.get("backend") or "onnx").lower() == backend
    ]  # fmt: skip
    fps = [x for x in (_num(r.get("detector_fps")) for r in rows) if x is not None]
    return (fps[0] if fps else None), any(_truthy(r.get("contended")) for r in rows)


def decide_h3_speed(fps_s8: float, fps_s: float) -> bool:
    return fps_s8 / fps_s >= H3_MIN_FPS_RATIO


def decide_h4_speed(fps_onnx: float, fps_torch: float) -> bool:
    return fps_onnx / fps_torch >= H4_MIN_FPS_RATIO


def equivalence_passes(row: dict) -> bool:
    if "passed" in row and row["passed"] not in (None, ""):
        return _truthy(row["passed"])
    iou, frac = _num(row.get("min_matched_iou")), _num(row.get("frac_frames_equal_count"))
    return iou is not None and frac is not None and iou >= H4_MIN_IOU and frac >= H4_MIN_FRAC_EQUAL


def _model_id(row: dict) -> str:
    v = str(row.get("config") or row.get("model") or "").strip().lower()
    return {"n": "N", "nano": "N", "s": "S", "small": "S"}.get(v, v.upper())


def _verdict(ok: bool | None) -> str:
    return "not evaluable" if ok is None else ("supported" if ok else "not supported")


def claims(
    out_dir: Path = RESULTS,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    log: Callable[[str], None] = print,
) -> dict:
    """Evaluate H1..H4 from the result files; write `claims.json` and print a table."""
    out_dir = Path(out_dir)
    events = _maybe_read(out_dir / TEST_EVENTS)
    seqs = _maybe_read(out_dir / TEST_SEQUENCES)
    bench = _maybe_read(out_dir / BENCH)
    equiv = _maybe_read(out_dir / EQUIVALENCE)
    out: dict[str, dict] = {}

    stat: dict[str, dict] = {}
    if events and seqs:
        universe = block_universe(seqs)
        w = resample_weights(len(universe), n_resamples, seed)
        tables = {
            key: build_table(events, universe, *key)
            for key in {(e["config"], int(e["k"]), e["arm"]) for e in events}
        }
        stat = {key: pooled_metrics(t, w) for key, t in tables.items()}
        point = {
            key: {m: float(v[0]) for m, v in pooled_metrics(t, np.ones((1, t.n_blocks))).items()}
            for key, t in tables.items()
        }
    ref_h, ref_n, ref_s8 = ("S", 1, "hysteresis"), ("S", 1, "naive"), ("S8", 1, "hysteresis")

    # H1: eer >= 1.5 * wape for S, k=1, hysteresis
    if ref_h in stat:
        d = stat[ref_h]["eer"] - H1_FACTOR * stat[ref_h]["wape"]
        lo, hi = interval(d)
        out["H1"] = {
            "claim": "a net count hides error: eer >= 1.5 x wape (S, k=1, hysteresis)",
            "status": _verdict(lower_bound_positive(d)),
            "estimate": point[ref_h]["eer"] - H1_FACTOR * point[ref_h]["wape"],
            "ci95": [lo, hi],
            "detail": {"eer": point[ref_h]["eer"], "wape": point[ref_h]["wape"]},
        }
    else:
        out["H1"] = _not_evaluable(
            "a net count hides error", "no test events for S, k=1, hysteresis"
        )

    # H2: f1(hysteresis) - f1(naive) > 0, paired
    if ref_h in stat and ref_n in stat:
        d = stat[ref_h]["f1"] - stat[ref_n]["f1"]
        lo, hi = interval(d)
        out["H2"] = {
            "claim": "hysteresis beats naive: f1 difference > 0 (S, k=1, paired)",
            "status": _verdict(lower_bound_positive(d - H2_MARGIN)),
            "estimate": point[ref_h]["f1"] - point[ref_n]["f1"],
            "ci95": [lo, hi],
            "detail": {"f1_hysteresis": point[ref_h]["f1"], "f1_naive": point[ref_n]["f1"]},
        }
    else:
        out["H2"] = _not_evaluable(
            "hysteresis beats naive", "no test events for S, k=1 (both arms)"
        )

    # H3: int8 FPS ratio >= 1.3 and upper bound of f1(S) - f1(S8) <= 0.03
    claim3 = (
        "int8 is a free speed-up: S8/S detector FPS >= 1.3 and f1(S) - f1(S8) <= 0.03 (upper bound)"
    )
    fps_s, c1 = _bench_fps(bench or [], "S", "onnx")
    fps_s8, c2 = _bench_fps(bench or [], "S8", "onnx")
    if c1 or c2:
        out["H3"] = _not_evaluable(
            claim3, "timings were taken contended (past the quiet-machine gate)"
        )
    elif fps_s is None or fps_s8 is None:
        out["H3"] = _not_evaluable(claim3, f"{BENCH} missing or has no S / S8 ONNX detector_fps")
    elif ref_h not in stat or ref_s8 not in stat:
        out["H3"] = _not_evaluable(claim3, "no test events for S and S8 at k=1, hysteresis")
    else:
        d = stat[ref_h]["f1"] - stat[ref_s8]["f1"]
        lo, hi = interval(d)
        speed_ok = decide_h3_speed(fps_s8, fps_s)
        out["H3"] = {
            "claim": claim3,
            "status": _verdict(speed_ok and hi <= H3_MAX_F1_DROP),
            "estimate": fps_s8 / fps_s,
            "ci95": [lo, hi],
            "detail": {
                "fps_ratio": fps_s8 / fps_s, "speed_ok": speed_ok,
                "f1_drop_upper_bound": hi, "accuracy_ok": hi <= H3_MAX_F1_DROP,
            },
        }  # fmt: skip

    # H4: ONNX FPS >= 1.5 x PyTorch for Nano and Small fp32, with equivalence passing
    claim4 = "ONNX Runtime >= 1.5 x eager PyTorch FPS for Nano and Small, equivalence passing"
    per_model: dict[str, dict] = {}
    reason = None
    for model in ("N", "S"):
        f_onnx, ca = _bench_fps(bench or [], model, "onnx")
        f_torch, cb = _bench_fps(bench or [], model, "torch")
        eq = [r for r in (equiv or []) if _model_id(r) == model]
        if ca or cb:
            reason = "timings were taken contended (past the quiet-machine gate)"
        elif f_onnx is None or f_torch is None:
            reason = f"{BENCH} missing or has no ONNX and PyTorch detector_fps for {model}"
        elif not eq:
            reason = f"{EQUIVALENCE} missing or has no row for {model}"
        else:
            per_model[model] = {
                "fps_ratio": f_onnx / f_torch,
                "speed_ok": decide_h4_speed(f_onnx, f_torch),
                "equivalence_ok": equivalence_passes(eq[0]),
            }
        if reason:
            break
    if reason:
        out["H4"] = _not_evaluable(claim4, reason)
    else:
        ok = all(m["speed_ok"] and m["equivalence_ok"] for m in per_model.values())
        out["H4"] = {"claim": claim4, "status": _verdict(ok), "detail": per_model}

    (out_dir / CLAIMS).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for line in format_claims(out):
        log(line)
    return out


def _not_evaluable(claim: str, why: str) -> dict:
    return {"claim": claim, "status": "not evaluable", "detail": why}


def format_claims(result: dict) -> list[str]:
    lines = [f"{'claim':<5} {'status':<14} detail"]
    for key in sorted(result):
        r = result[key]
        extra = ""
        if "estimate" in r and "ci95" in r:
            extra = (
                f"estimate {r['estimate']:.4f}, 95% CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]; "
            )
        detail = r.get("detail")
        extra += detail if isinstance(detail, str) else json.dumps(detail, default=float)
        lines.append(f"{key:<5} {r['status']:<14} {extra}")
        lines.append(f"      {r['claim']}")
    return lines


# ------------------------------------------------------------------------------------- frontier


def pipeline_fps(bench: Sequence[dict], config: str, k: int) -> float | None:
    """Pipeline FPS for (config, k) from the ONNX rows of `bench`.

    A row for exactly this k with a `pipeline_fps` value wins. Otherwise the plan's formula
    `k / (t_detect + t_track_count)` is applied to the k = 1 row (or a row without k), which is
    where the benchmark measures both times.
    """
    rows = [
        r for r in bench
        if r.get("config") == config and (r.get("backend") or "onnx").lower() == "onnx"
    ]  # fmt: skip

    def row_k(r: dict) -> int:
        return int(float(r["k"])) if r.get("k") not in (None, "") else 1

    for r in rows:
        direct = _num(r.get("pipeline_fps"))
        if row_k(r) == k and direct is not None:
            return direct
    for r in rows:
        det, trk = _num(r.get("detector_fps")), _num(r.get("t_track_count"))
        if row_k(r) == 1 and det and trk is not None:
            return k / (1.0 / det + trk)
    return None


def frontier(out_dir: Path = RESULTS, log: Callable[[str], None] = print) -> list[dict]:
    """F1 (hysteresis arm) vs pipeline FPS per config x k on test; write `frontier.csv`.

    Marks the cheapest point (highest FPS) whose f1 is within 0.02 of S at k=1. Reported, not
    claimed. Without bench data the f1 columns are still written and the FPS columns are empty.
    """
    out_dir = Path(out_dir)
    cells = read_csv(out_dir / TEST_CELLS)
    bench = _maybe_read(out_dir / BENCH) or []
    by_point: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in cells:
        if r["arm"] == "hysteresis":
            by_point[(r["config"], r["k"])].append(r)
    rows = [
        {"config": c, "k": k, "f1": pool(rs)["f1"], "pipeline_fps": pipeline_fps(bench, c, k)}
        for (c, k), rs in sorted(by_point.items())
    ]
    ref = next((r["f1"] for r in rows if r["config"] == "S" and r["k"] == 1), None)
    for r in rows:
        r["within_0.02_of_S1"] = ref is not None and r["f1"] >= ref - FRONTIER_TOLERANCE
        r["cheapest_within"] = False
    ok = [r for r in rows if r["within_0.02_of_S1"] and r["pipeline_fps"] is not None]
    if ok:
        max(ok, key=lambda r: r["pipeline_fps"])["cheapest_within"] = True
    write_csv(
        out_dir / FRONTIER,
        [{k: ("" if v is None else v) for k, v in r.items()} for r in rows],
        ["config", "k", "f1", "pipeline_fps", "within_0.02_of_S1", "cheapest_within"],
    )
    if not ok:
        log("frontier: no pipeline FPS available (bench.csv missing or incomplete); f1 only")
    for r in rows:
        mark = " <- cheapest within 0.02 of S,k=1" if r["cheapest_within"] else ""
        fps = "n/a" if r["pipeline_fps"] is None else f"{r['pipeline_fps']:.2f}"
        log(f"{r['config']:<3} k={r['k']}  f1={r['f1']:.4f}  pipeline_fps={fps}{mark}")
    return rows


# ---------------------------------------------------------------------------------------- summary

_POOLED = (
    "f1", "precision", "recall", "f1_time", "wape", "eer", "hidden", "switch_share",
    "n_true", "n_pred", "tp", "fp", "fn",
    *(f"fp_{c}" for c in FP_CATEGORIES),
    *(f"fn_{c}" for c in FN_CATEGORIES),
)  # fmt: skip
_MOT_COLS = (
    "mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations",
    "mostly_tracked", "mostly_lost", "precision", "recall", "num_frames", "num_objects",
    "num_predictions",
)  # fmt: skip


def _arm_key(r: dict) -> tuple[str, int, str]:
    return (r["config"], r["k"], r["arm"])


def summary_rows(
    cells: Sequence[dict],
    events: Sequence[dict],
    seqs: Sequence[dict],
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    """Pooled metrics with 95 % cluster-bootstrap intervals per (config, k, arm).

    Returns the rows and the per-key bootstrap metric arrays (shared weights, so differences
    between keys are paired).
    """  # fmt: skip
    universe = block_universe(seqs)
    w = resample_weights(len(universe), n_resamples, seed)
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in cells:
        by_key[_arm_key(r)].append(r)
    ev_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        ev_by_key[(e["config"], int(e["k"]), e["arm"])].append(e)
    rows, boot = [], {}
    for key in sorted(by_key, key=lambda t: (_config_rank(t[0]), t[1], t[2])):
        rs = by_key[key]
        pooled = pool(rs)
        row = {"config": key[0], "k": key[1], "arm": key[2],
               "tau": rs[0]["tau"], "band": rs[0]["band"]}  # fmt: skip
        row.update({m: pooled[m] for m in _POOLED})
        table = build_table(ev_by_key.get(key, []), universe, *key)
        boot[key] = pooled_metrics(table, w)
        for m in SUMMARY_CI_METRICS:
            row[f"{m}_lo"], row[f"{m}_hi"] = interval(boot[key][m])
        rows.append(row)
    return rows, boot


def by_sequence_rows(cells: Sequence[dict], mot: Sequence[dict]) -> list[dict]:
    """Pooled metrics per (config, k, arm, seq), with MOT metrics joined at the selected tau."""
    mot_by = {(m["config"], m["seq"], m["k"], m["tau"]): m for m in mot}
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in cells:
        groups[(*_arm_key(r), r["seq"])].append(r)
    rows = []
    for key in sorted(groups, key=lambda t: (_config_rank(t[0]), t[1], t[2], t[3])):
        rs = groups[key]
        pooled = pool(rs)
        row = {"config": key[0], "k": key[1], "arm": key[2], "seq": key[3],
               "tau": rs[0]["tau"], "band": rs[0]["band"]}  # fmt: skip
        row.update({m: pooled[m] for m in _POOLED})
        m = mot_by.get((key[0], key[3], key[1], rs[0]["tau"]), {})
        row.update({f"mot_{c}": m.get(c, "") for c in _MOT_COLS})
        rows.append(row)
    return rows


def detector_score_rows(
    src: Sources,
    configs: Sequence[str],
    seqs: Sequence[str],
    selected_tau: dict[str, float],
    split_of: Callable[[str], str],
    taus: Sequence[float] = TAUS,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Per (config, sequence): detection counts and person-score quantiles from the caches.

    A missing cache is skipped with a log line. `selected_tau` maps config -> the threshold
    chosen for k = 1, hysteresis. Scores here are the cached person detections (>= 0.10).
    """
    rows = []
    for config in configs:
        for seq in seqs:
            try:
                per_frame = src.detections(config, seq)
            except FileNotFoundError:
                log(f"scores: no cache for {config}/{seq}, skipped")
                continue
            n_frames = max(1, src.n_frames(seq))
            scores = (
                np.concatenate([d.score for d in per_frame.values()])
                if per_frame else np.empty(0)
            )  # fmt: skip
            q = np.percentile(scores, [50, 90, 99]) if scores.size else [float("nan")] * 3
            row = {
                "config": config, "seq": seq, "split": split_of(seq), "n_frames": n_frames,
                "mean_det_per_frame": scores.size / n_frames,
                "mean_gt_per_frame": len(src.gt(seq)) / n_frames,
                "score_p50": float(q[0]), "score_p90": float(q[1]), "score_p99": float(q[2]),
                "score_max": float(scores.max()) if scores.size else float("nan"),
            }  # fmt: skip
            for t in taus:
                row[f"det_per_frame_ge_{t}"] = int((scores >= t).sum()) / n_frames
            row["selected_tau_k1_hysteresis"] = selected_tau.get(config, "")
            rows.append(row)
    return rows


def h3_accuracy_half(boot: dict, point: dict, log: Callable[[str], None] = print) -> dict | None:
    """f1(S) - f1(S8) at k = 1, hysteresis arm, paired over the same resampled blocks."""
    a, b = ("S", 1, "hysteresis"), ("S8", 1, "hysteresis")
    if a not in boot or b not in boot:
        log("H3 accuracy half: S and S8 at k=1, hysteresis not both in the results")
        return None
    d = boot[a]["f1"] - boot[b]["f1"]
    lo, hi = interval(d)
    out = {
        "f1_S": point[a], "f1_S8": point[b], "difference": point[a] - point[b],
        "ci95": [lo, hi], "upper_bound_within_0.03": hi <= H3_MAX_F1_DROP,
    }  # fmt: skip
    log(
        f"H3 accuracy half (reported; the speed half needs bench.csv): f1(S) - f1(S8) = "
        f"{out['difference']:.4f}, paired 95% CI [{lo:.4f}, {hi:.4f}], upper bound <= "
        f"{H3_MAX_F1_DROP}: {'yes' if out['upper_bound_within_0.03'] else 'no'}"
    )
    return out


def summary(
    out_dir: Path = RESULTS,
    src: Sources | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    log: Callable[[str], None] = print,
) -> dict:
    """Reported-only tables from the committed `test_*.csv` files and the detection caches.

    Writes `test_summary.csv` (pooled metrics with bootstrap intervals), `test_by_sequence.csv`
    (per sequence, with MOT metrics at the selected tau) and `test_detector_scores.csv` (score
    distribution per config and sequence, dev and test). Prints the accuracy half of H3.
    Never changes `test_cells.csv`, `test_events.csv`, `test_sequences.csv` or `test_mot.csv`.
    """
    out_dir = Path(out_dir)
    cells = read_csv(out_dir / TEST_CELLS)
    events = read_csv(out_dir / TEST_EVENTS)
    seqs = read_csv(out_dir / TEST_SEQUENCES)
    mot = read_csv(out_dir / TEST_MOT)
    rows, boot = summary_rows(cells, events, seqs, n_resamples, seed)
    write_csv(out_dir / SUMMARY, rows)
    write_csv(out_dir / BY_SEQUENCE, by_sequence_rows(cells, mot))
    point = {(r["config"], r["k"], r["arm"]): r["f1"] for r in rows}
    h3 = h3_accuracy_half(boot, point, log)

    selected = {r["config"]: r["tau"] for r in cells if r["k"] == 1 and r["arm"] == "hysteresis"}
    if src is None:
        src = default_sources()
    from vidan.data import SEQUENCES

    score_rows = detector_score_rows(
        src,
        sorted({r["config"] for r in cells}, key=_config_rank),
        list(SEQUENCES),
        selected,
        lambda s: SEQUENCES[s].split,
        log=log,
    )
    write_csv(out_dir / DETECTOR_SCORES, score_rows)
    log(f"summary: {len(rows)} rows, {len(score_rows)} score rows -> {out_dir}")
    return {"summary": rows, "scores": score_rows, "h3_accuracy": h3}


# ------------------------------------------------------------------------------------------ cli


def run_cli(args: Any) -> int:
    """Handler for `vidan eval <stage>` (arguments are defined in `cli._add_eval`)."""
    out_dir = Path(args.results_dir)
    subset = any(getattr(args, f, None) for f in ("configs", "seqs", "ks", "taus"))
    if args.stage == "test" and subset:
        print(
            "error: `eval test` takes no --configs/--seqs/--ks/--taus: the test split is scored "
            "once, for every selected setting",
            file=sys.stderr,
        )
        return 2
    if args.stage == "tune" and subset and out_dir.resolve() == RESULTS.resolve():
        print(
            "error: a subset tune run must write elsewhere: pass --results-dir other than "
            f"{RESULTS}",
            file=sys.stderr,
        )
        return 2
    if args.stage == "claims":
        claims(out_dir)
        return 0
    if args.stage == "frontier":
        frontier(out_dir)
        return 0
    if args.stage == "summary":
        summary(out_dir, default_sources(args.cache_root, args.raw_root))
        return 0
    src = default_sources(args.cache_root, args.raw_root)
    if args.stage == "tune":
        tune(
            src,
            out_dir,
            configs=args.configs or CONFIGS,
            ks=args.ks or KS,
            taus=args.taus or TAUS,
            seqs=args.seqs,
        )
        return 0
    try:
        test(src, out_dir, configs=args.configs, seqs=args.seqs)
    except (SelectionNotCommitted, RepeatTestRun) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
