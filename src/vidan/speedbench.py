"""Speed benchmark behind a quiet-machine gate, and the ONNX-vs-PyTorch equivalence check.

Both follow `EVAL_PLAN.md`. The gate, the statistics and the equivalence maths are pure
functions (tested without the vision extra); the `run_*` drivers import torch, onnxruntime and
OpenCV lazily.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vidan.detect import PERSON, Detections

GATE_CPU_PCT = 15.0  # start a repeat only if system-wide CPU use averaged below this
GATE_SAMPLE_S = 10.0  # length of the idle sample before each repeat
GATE_RETRY_S = 30.0  # pause between a failed gate reading and the next one
BENCH_CONFIGS = ("torch-N", "torch-S", "onnx-N", "onnx-S", "onnx-N8", "onnx-S8")
BENCH_SEQUENCE = "MOT17-04"
TRACK_COUNT_TAU = 0.5  # one representative setting; timing does not depend on tuning
TRACK_COUNT_BAND_PX = 16.0

# Equivalence pass rule (EVAL_PLAN H4).
EQUIV_MIN_IOU = 0.99
EQUIV_MIN_COUNT_EQUAL = 0.95
EQUIV_THRESHOLDS = (0.10, 0.30)  # the cache threshold and the lowest tuning threshold


# --------------------------------------------------------------------------- gate


@dataclass(frozen=True)
class GateResult:
    """Outcome of a quiet-machine check: the last reading, how long it took, how many readings."""

    passed: bool
    cpu_pct: float
    waited_s: float
    attempts: int


def sample_cpu_percent(seconds: float = GATE_SAMPLE_S) -> float:
    """Mean system-wide CPU use over `seconds`, one reading per second (psutil).

    Taken while this process is idle, and it excludes nothing: every process on the machine counts.
    """
    import psutil

    psutil.cpu_percent(interval=None)  # prime the counter
    readings = [psutil.cpu_percent(interval=1.0) for _ in range(max(1, round(seconds)))]
    return float(statistics.fmean(readings))


def quiet_gate(
    sample: Callable[[], float],
    threshold: float = GATE_CPU_PCT,
    max_wait_s: float = 0.0,
    retry_s: float = GATE_RETRY_S,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> GateResult:
    """Take readings until one is below `threshold` percent, for at most `max_wait_s` seconds.

    The first reading is always taken. After a failed one, give up if the wait budget is spent,
    else pause `retry_s` and read again.
    """
    start = now()
    attempts = 0
    while True:
        reading = float(sample())
        attempts += 1
        if reading < threshold:
            return GateResult(True, reading, now() - start, attempts)
        if now() - start >= max_wait_s:
            return GateResult(False, reading, now() - start, attempts)
        sleep(retry_s)


# --------------------------------------------------------------------------- statistics


def repeat_stats(latencies_s: Sequence[float]) -> dict[str, float]:
    """Mean, p50 and p95 of one repeat's per-frame latencies, in milliseconds."""
    arr = np.asarray(latencies_s, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def summarise_repeats(repeats: Sequence[Sequence[float]]) -> dict[str, float]:
    """Median of the repeats' mean latencies; p50 / p95 of all per-frame latencies pooled."""
    means = [float(np.mean(r)) * 1000.0 for r in repeats]
    pooled = repeat_stats([x for r in repeats for x in r])
    mean_ms = float(statistics.median(means))
    return {"mean_ms": mean_ms, "p50_ms": pooled["p50_ms"], "p95_ms": pooled["p95_ms"]}


# --------------------------------------------------------------------------- equivalence maths


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of two (n, 4) / (m, 4) xyxy arrays -> (n, m)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    iy = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def match_ious(a: Detections, b: Detections) -> tuple[list[float], int, int]:
    """One-to-one Hungarian matching on IoU between two detection sets of one frame.

    Any pair with IoU > 0 may match, so a poor match shows up as a low IoU instead of vanishing
    into the unmatched counts. Returns (IoU of each matched pair, unmatched in a, unmatched in b).
    """
    from scipy.optimize import linear_sum_assignment

    m = iou_matrix(a.xyxy, b.xyxy)
    if m.size == 0:
        return [], len(a), len(b)
    rows, cols = linear_sum_assignment(m, maximize=True)
    ious = [float(m[r, c]) for r, c in zip(rows, cols, strict=True) if m[r, c] > 0]
    return ious, len(a) - len(ious), len(b) - len(ious)


def above_threshold(det: Detections, threshold: float) -> Detections:
    return det.keep(det.score > threshold)


def equivalence_summary(
    per_frame: Sequence[tuple[Detections, Detections]],
    min_iou: float = EQUIV_MIN_IOU,
    min_count_equal: float = EQUIV_MIN_COUNT_EQUAL,
) -> dict[str, Any]:
    """Aggregate (reference, candidate) detection pairs, one per frame, against the pass rule.

    Pass iff every matched box has IoU >= `min_iou` and the two detectors report the same number
    of boxes on at least `min_count_equal` of the frames.
    """
    ious: list[float] = []
    equal = 0
    unmatched_ref = unmatched_cand = 0
    for ref, cand in per_frame:
        equal += len(ref) == len(cand)
        got, ua, ub = match_ious(ref, cand)
        ious.extend(got)
        unmatched_ref += ua
        unmatched_cand += ub
    n = len(per_frame)
    arr = np.asarray(ious, dtype=np.float64)
    frac_equal = equal / n if n else 0.0
    iou_min = float(arr.min()) if arr.size else float("nan")
    return {
        "n_frames": n,
        "n_count_equal": equal,
        "frac_count_equal": frac_equal,
        "n_matched": int(arr.size),
        "n_unmatched_torch": unmatched_ref,
        "n_unmatched_onnx": unmatched_cand,
        "iou_mean": float(arr.mean()) if arr.size else float("nan"),
        "iou_min": iou_min,
        "iou_p01": float(np.percentile(arr, 1)) if arr.size else float("nan"),
        "frac_iou_ge_min": float((arr >= min_iou).mean()) if arr.size else float("nan"),
        "passed": bool(arr.size and iou_min >= min_iou and frac_equal >= min_count_equal),
    }


def evenly_spaced(n_total: int, n: int) -> list[int]:
    """`n` indices spread evenly over `range(n_total)`, first and last included, no repeats."""
    if n_total <= 0:
        return []
    if n >= n_total:
        return list(range(n_total))
    if n == 1:
        return [0]
    return [int(i) for i in np.round(np.linspace(0, n_total - 1, n))]


# --------------------------------------------------------------------------- tracking timing


def track_count_seconds(
    per_frame: dict[int, Detections],
    frame_rate: float,
    lines: Sequence[Any],
    tau: float = TRACK_COUNT_TAU,
    band_px: float = TRACK_COUNT_BAND_PX,
) -> tuple[float, int]:
    """Seconds to track (k = 1, ByteTrack, threshold `tau`) and count (naive and hysteresis, every
    line) one cached sequence, and the number of frames processed."""
    from vidan.counting import Rule, count_line
    from vidan.track import run_bytetrack

    t0 = time.perf_counter()
    tracks = run_bytetrack(
        per_frame,
        frame_rate,
        track_activation_threshold=tau,
        high_conf_det_threshold=tau,
    )
    for line in lines:
        count_line(tracks, line, Rule.naive())
        count_line(tracks, line, Rule.hysteresis(band_px))
    return time.perf_counter() - t0, len(per_frame)


# --------------------------------------------------------------------------- drivers

# `bench.csv` is read by `vidan eval claims|frontier`: one row per (backend, config).
BENCH_COLUMNS = [
    "config",
    "backend",
    "detector_fps",
    "contended",
    "k",
    "t_track_count",
    "pipeline_fps",
    "model",
    "input",
    "precision",
    "n_frames",
    "threads",
    "repeats",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "gate_cpu_pct_max",
    "gate_waited_s",
]
BENCH_REPEAT_COLUMNS = [
    "config",
    "backend",
    "repeat",
    "n_frames",
    "threads",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "fps",
    "gate_cpu_pct",
    "gate_waited_s",
    "contended",
]


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _bench_detector(name: str, threads: int):
    """Build the detector of one benchmark config name (`torch-N`, `onnx-S8`, ...)."""
    from vidan import runs
    from vidan.detect import OnnxRFDETR, TorchRFDETR

    backend, cid = name.split("-", 1)
    spec = runs.config_spec(cid)
    if backend == "torch":
        return TorchRFDETR(
            spec.size,
            spec.resolution,
            threshold=runs.CACHE_THRESHOLD,
            classes=(PERSON,),
            num_threads=threads,
        )
    return OnnxRFDETR(
        runs.model_path(cid),
        spec.resolution,
        threshold=runs.CACHE_THRESHOLD,
        classes=(PERSON,),
        intra_op_threads=threads,
        label=f"onnx-{cid}",
    )


def pipeline_fps(detector_fps: float, t_track_count: float | None, k: int = 1) -> float:
    """Frames per second of the whole pipeline at frame skip `k`: k / (t_detect + t_track_count)."""
    t = 1.0 / detector_fps + (t_track_count or 0.0)
    return k / t


def run_bench(args: argparse.Namespace) -> int:
    """`vidan bench`: detector latency per config, then tracking + counting time per frame."""
    from vidan import data, runs
    from vidan.bench import benchmark, env_fingerprint, set_threads
    from vidan.detect import load_cache
    from vidan.layout import read_layout

    configs = list(args.configs or BENCH_CONFIGS)
    unknown = [c for c in configs if c not in BENCH_CONFIGS]
    if unknown:
        print(f"unknown bench configs {unknown}; expected a subset of {BENCH_CONFIGS}")
        return 1
    max_wait_s = args.max_wait * 60.0
    repeat_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    track_count_s: dict[str, float] = {}
    latencies: dict[str, list[list[float]]] = {}
    gate_log: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    def gate(label: str) -> tuple[bool, GateResult]:
        """Run the gate; with --force take one reading, never wait, and report contended."""
        if args.force:
            reading = sample_cpu_percent()
            res = GateResult(reading < GATE_CPU_PCT, reading, 0.0, 1)
            gate_log.append({"label": label, **res.__dict__, "forced": True})
            return True, res
        res = quiet_gate(sample_cpu_percent, GATE_CPU_PCT, max_wait_s)
        gate_log.append({"label": label, **res.__dict__, "forced": False})
        print(
            f"[gate] {label}: {res.cpu_pct:.1f}% over {GATE_SAMPLE_S:.0f} s, "
            f"{'pass' if res.passed else 'FAIL'} after {res.waited_s:.0f} s"
        )
        return res.passed, res

    def finish(code: int) -> int:
        out: Path = args.out
        rows = []
        for sm in summaries:
            t_tc = track_count_s.get(sm["config"])
            rows.append(
                {
                    **sm,
                    "k": 1,
                    "t_track_count": "" if t_tc is None else t_tc,
                    "pipeline_fps": pipeline_fps(sm["detector_fps"], t_tc),
                }
            )
        _write_csv(out, BENCH_COLUMNS, rows)
        _write_csv(out.with_name(out.stem + "_repeats.csv"), BENCH_REPEAT_COLUMNS, repeat_rows)
        fp = {
            "started": started,
            "argv": sys.argv,
            "force": bool(args.force),
            "gate": {
                "cpu_pct_below": GATE_CPU_PCT,
                "sample_s": GATE_SAMPLE_S,
                "max_wait_min": args.max_wait,
            },
            "gate_readings": gate_log,
            "track_count_seconds_per_frame": track_count_s,
            "environment": env_fingerprint(),
            "exit_code": code,
        }
        out.with_name(out.stem + "_fingerprint.json").write_text(
            json.dumps(fp, indent=2), encoding="utf-8"
        )
        out.with_name(out.stem + "_latencies.json").write_text(
            json.dumps(latencies), encoding="utf-8"
        )
        print(f"wrote {out} ({len(rows)} rows)")
        return code

    seq = data.seq_dir(BENCH_SEQUENCE)
    items = list(data.frames(seq))[: args.frames]
    frames = [data.load_frame(p) for _, p in items]
    set_threads(args.threads)

    for name in configs:
        backend, cid = name.split("-", 1)
        spec = runs.config_spec(cid)
        det = _bench_detector(name, args.threads)
        per_repeat: list[list[float]] = []
        mine: list[dict[str, Any]] = []
        for rep in range(1, args.repeats + 1):
            ok, res = gate(f"{name} repeat {rep}")
            if not ok:
                print(f"machine not quiet after {args.max_wait} min; stopping (exit 2)")
                return finish(2)
            result = benchmark(
                det, frames, warmup=args.warmup, repeats=len(frames), threads=args.threads
            )
            per_repeat.append(result["latencies"])
            stats = repeat_stats(result["latencies"])
            mine.append(
                {
                    "config": cid,
                    "backend": backend,
                    "repeat": rep,
                    "n_frames": len(frames),
                    "threads": args.threads,
                    **stats,
                    "fps": 1000.0 / stats["mean_ms"],
                    "gate_cpu_pct": round(res.cpu_pct, 2),
                    "gate_waited_s": round(res.waited_s, 1),
                    "contended": bool(args.force),
                }
            )
            print(f"{name} repeat {rep}: {stats['mean_ms']:.1f} ms/frame")
        repeat_rows.extend(mine)
        latencies[name] = per_repeat
        summ = summarise_repeats(per_repeat)
        summaries.append(
            {
                "config": cid,
                "backend": backend,
                "detector_fps": 1000.0 / summ["mean_ms"],
                "contended": bool(args.force),
                "model": spec.size,
                "input": spec.resolution,
                "precision": "int8" if spec.int8 else "fp32",
                "n_frames": len(frames),
                "threads": args.threads,
                "repeats": args.repeats,
                **summ,
                "gate_cpu_pct_max": max(r["gate_cpu_pct"] for r in mine),
                "gate_waited_s": round(sum(r["gate_waited_s"] for r in mine), 1),
            }
        )
        del det

    # tracking + counting per processed frame (seconds), on cached dev detections
    layout = read_layout()
    for cid in ("N", "S", "N8", "S8"):
        names = data.names_for_split("dev")
        paths = [runs.cache_path(cid, n) for n in names]
        if not all(p.exists() for p in paths):
            print(f"track_count {cid}: dev cache missing, skipped")
            continue
        loaded = {n: load_cache(p) for n, p in zip(names, paths, strict=True)}
        per_repeat_s: list[float] = []
        for rep in range(1, args.repeats + 1):
            ok, _ = gate(f"track_count {cid} repeat {rep}")
            if not ok:
                print(f"machine not quiet after {args.max_wait} min; stopping (exit 2)")
                return finish(2)
            secs, frames_done = 0.0, 0
            for n in names:
                per_frame, meta = loaded[n]
                s, f = track_count_seconds(
                    per_frame, float(meta["frame_rate"]), list(layout[n]["lines"].values())
                )
                secs += s
                frames_done += f
            per_repeat_s.append(secs / frames_done)
        track_count_s[cid] = float(statistics.median(per_repeat_s))
        print(f"track_count {cid}: {track_count_s[cid] * 1000:.3f} ms per processed frame")
    return finish(0)


# `equivalence.csv` is read by `vidan eval claims`: one row per config, at the cache threshold.
EQUIV_COLUMNS = [
    "config",
    "threshold",
    "passed",
    "min_matched_iou",
    "frac_frames_equal_count",
    "n_frames",
    "n_frames_equal_count",
    "n_matched",
    "n_unmatched_torch",
    "n_unmatched_onnx",
    "mean_matched_iou",
    "p01_matched_iou",
    "frac_matched_iou_ge_min",
]
EQUIV_FRAME_COLUMNS = [
    "config",
    "threshold",
    "sequence",
    "frame",
    "n_torch",
    "n_onnx",
    "count_equal",
    "n_matched",
    "min_matched_iou",
]
_EQUIV_RENAME = {
    "iou_min": "min_matched_iou",
    "frac_count_equal": "frac_frames_equal_count",
    "n_count_equal": "n_frames_equal_count",
    "iou_mean": "mean_matched_iou",
    "iou_p01": "p01_matched_iou",
    "frac_iou_ge_min": "frac_matched_iou_ge_min",
}


def equiv_row(config: str, threshold: float, summary: dict[str, Any]) -> dict[str, Any]:
    """One CSV row of an `equivalence_summary`, with the column names `vidan eval` expects."""
    return {
        "config": config,
        "threshold": threshold,
        **{_EQUIV_RENAME.get(k, k): v for k, v in summary.items()},
    }


def run_equiv(args: argparse.Namespace) -> int:
    """`vidan equiv`: PyTorch vs ONNX fp32 on evenly spaced dev frames, for each config.

    `equivalence.csv` holds the primary rows, at the cache threshold (0.10), which is what the
    pipeline consumes. The same comparison at threshold 0.30 goes to `equivalence_thr0.30.csv`.
    """
    from vidan import data, runs
    from vidan.detect import OnnxRFDETR, TorchRFDETR

    entries = []
    for name in data.names_for_split("dev"):
        entries.extend((name, no, p) for no, p in data.frames(data.seq_dir(name)))
    chosen = [entries[i] for i in evenly_spaced(len(entries), args.frames)]
    print(f"{len(chosen)} frames of {len(entries)} dev frames", flush=True)
    frames = [(s, n, data.load_frame(p)) for s, n, p in chosen]

    rows_by_thr: dict[float, list[dict[str, Any]]] = {t: [] for t in EQUIV_THRESHOLDS}
    frame_rows: list[dict[str, Any]] = []
    for cid in args.configs:
        spec = runs.config_spec(cid)
        tor = TorchRFDETR(
            spec.size,
            spec.resolution,
            threshold=min(EQUIV_THRESHOLDS),
            classes=(PERSON,),
            num_threads=args.threads,
        )
        onx = OnnxRFDETR(
            runs.model_path(cid),
            spec.resolution,
            threshold=min(EQUIV_THRESHOLDS),
            classes=(PERSON,),
            intra_op_threads=args.threads,
        )
        pairs = [(tor.detect(f), onx.detect(f)) for _, _, f in frames]
        for thr in EQUIV_THRESHOLDS:
            cut = [(above_threshold(a, thr), above_threshold(b, thr)) for a, b in pairs]
            summ = equivalence_summary(cut)
            rows_by_thr[thr].append(equiv_row(cid, thr, summ))
            for (s, n, _), (a, b) in zip(frames, cut, strict=True):
                got, _, _ = match_ious(a, b)
                frame_rows.append(
                    {
                        "config": cid,
                        "threshold": thr,
                        "sequence": s,
                        "frame": n,
                        "n_torch": len(a),
                        "n_onnx": len(b),
                        "count_equal": len(a) == len(b),
                        "n_matched": len(got),
                        "min_matched_iou": min(got) if got else "",
                    }
                )
            print(
                f"{cid} thr {thr}: counts equal {summ['n_count_equal']}/{summ['n_frames']}, "
                f"matched {summ['n_matched']} (unmatched torch {summ['n_unmatched_torch']}, "
                f"onnx {summ['n_unmatched_onnx']}), IoU min {summ['iou_min']:.5f} "
                f"mean {summ['iou_mean']:.6f}, pass={summ['passed']}",
                flush=True,
            )
    primary, secondary = min(EQUIV_THRESHOLDS), max(EQUIV_THRESHOLDS)
    _write_csv(args.out, EQUIV_COLUMNS, rows_by_thr[primary])
    _write_csv(
        args.out.with_name(f"{args.out.stem}_thr{secondary:.2f}.csv"),
        EQUIV_COLUMNS,
        rows_by_thr[secondary],
    )
    _write_csv(args.out.with_name(args.out.stem + "_frames.csv"), EQUIV_FRAME_COLUMNS, frame_rows)
    print(f"wrote {args.out}")
    return 0
