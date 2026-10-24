"""Detector configs of the evaluation plan, model export and the per-sequence detection cache.

Configs (`EVAL_PLAN.md`): S = Small 512 fp32, N = Nano 384 fp32, S8 / N8 = int8 dynamic
quantisation of those. Models live in `models/<config>.onnx` (git-ignored), detections in
`data/cache/<config>/<sequence>.npz`.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vidan.detect import PERSON, Detections, Detector, load_cache, save_cache

MODELS_DIR = Path("models")
CACHE_ROOT = Path("data/cache")
CACHE_THRESHOLD = 0.10  # low on purpose: ByteTrack's second association uses low scores
PROGRESS_EVERY = 100


@dataclass(frozen=True)
class ConfigSpec:
    """One detector configuration: model size, input resolution, weight precision."""

    id: str
    size: str
    resolution: int
    int8: bool

    @property
    def base_id(self) -> str:
        """The fp32 config this one is derived from (itself for fp32)."""
        return self.id.removesuffix("8")


CONFIGS: dict[str, ConfigSpec] = {
    "S": ConfigSpec("S", "small", 512, False),
    "N": ConfigSpec("N", "nano", 384, False),
    "S8": ConfigSpec("S8", "small", 512, True),
    "N8": ConfigSpec("N8", "nano", 384, True),
}


def config_spec(config: str) -> ConfigSpec:
    try:
        return CONFIGS[config]
    except KeyError:
        raise ValueError(f"unknown config {config!r}; expected one of {sorted(CONFIGS)}") from None


def model_path(config: str, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"{config_spec(config).id}.onnx"


def cache_path(config: str, sequence: str, cache_root: Path = CACHE_ROOT) -> Path:
    return Path(cache_root) / config_spec(config).id / f"{sequence}.npz"


# --------------------------------------------------------------------------- export


def export_config(config: str, models_dir: Path = MODELS_DIR) -> Path:
    """Write `models/<config>.onnx`; int8 configs quantise the fp32 file (exported if absent)."""
    from vidan.export import export_onnx, quantize_int8

    spec = config_spec(config)
    out = model_path(config, models_dir)
    if spec.int8:
        base = model_path(spec.base_id, models_dir)
        if not base.exists():
            export_config(spec.base_id, models_dir)
        return quantize_int8(base, out)
    scratch = Path(models_dir) / f"_export_{spec.id}"
    exported = export_onnx(spec.size, spec.resolution, scratch)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), out)
    shutil.rmtree(scratch, ignore_errors=True)
    return out


# --------------------------------------------------------------------------- detection cache


def cache_is_complete(path: Path, n_frames: int) -> bool:
    """True if `path` holds a readable cache with exactly `n_frames` frames."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        per_frame, meta = load_cache(path)
    except Exception:  # truncated or foreign file: redo it
        return False
    return len(per_frame) == n_frames == meta.get("n_frames")


def format_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def detect_frames(
    detector: Detector,
    items: Sequence[tuple[int, Any]],
    loader: Callable[[Any], np.ndarray],
    log: Callable[[str], None] = print,
    label: str = "",
    progress_every: int = PROGRESS_EVERY,
) -> tuple[dict[int, Detections], dict[str, Any]]:
    """Detect over `(frame_no, handle)` items; `loader(handle)` returns the BGR frame.

    A frame that fails to load or detect is logged, stored as empty `Detections` and listed in
    the returned stats under `bad_frames`. `detect_seconds` sums only the detector calls.
    """
    per_frame: dict[int, Detections] = {}
    bad: list[dict[str, Any]] = []
    detect_s = 0.0
    t_start = time.perf_counter()
    total = len(items)
    for i, (number, handle) in enumerate(items, start=1):
        try:
            frame = loader(handle)
            t0 = time.perf_counter()
            det = detector.detect(frame)
            detect_s += time.perf_counter() - t0
        except Exception as exc:
            det = Detections.empty()
            bad.append({"frame": int(number), "error": f"{type(exc).__name__}: {exc}"})
            log(f"{label} frame {number}: {type(exc).__name__}: {exc}; stored as empty")
        per_frame[int(number)] = det
        if i % progress_every == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / i * (total - i)
            log(
                f"{label} {i}/{total} frames, {elapsed / i:.3f} s/frame, "
                f"elapsed {format_eta(elapsed)}, ETA {format_eta(eta)}"
            )
    seconds = time.perf_counter() - t_start
    stats = {
        "seconds_taken": seconds,
        "detect_seconds": detect_s,
        "sec_per_frame": seconds / max(total, 1),
        "detect_sec_per_frame": detect_s / max(total, 1),
        "bad_frames": bad,
    }
    return per_frame, stats


def detect_split(
    config: str,
    split: str,
    threads: int = 12,
    models_dir: Path = MODELS_DIR,
    data_root: Path | None = None,
    cache_root: Path = CACHE_ROOT,
    log: Callable[[str], None] = print,
    sequences: Iterable[str] | None = None,
) -> list[Path]:
    """Cache person detections (score >= 0.10) of one config over a split, one `.npz` per sequence.

    Sequences whose cache exists with the right frame count are skipped. No metric is computed.
    """
    from vidan import data
    from vidan.detect import OnnxRFDETR

    spec = config_spec(config)
    names = list(sequences) if sequences is not None else data.names_for_split(split)
    root = data.DEFAULT_ROOT if data_root is None else Path(data_root)
    detector: OnnxRFDETR | None = None
    written: list[Path] = []
    for name in names:
        seq = data.seq_dir(name, root)
        items = list(data.frames(seq))
        out = cache_path(config, name, cache_root)
        label = f"[{spec.id} {name}]"
        if cache_is_complete(out, len(items)):
            log(f"{label} cache complete ({len(items)} frames), skipping")
            continue
        if detector is None:
            detector = OnnxRFDETR(
                model_path(config, models_dir),
                resolution=spec.resolution,
                threshold=CACHE_THRESHOLD,
                classes=(PERSON,),
                intra_op_threads=threads,
                label=spec.id,
            )
        info = data.read_seqinfo(seq)
        log(f"{label} start: {len(items)} frames, {threads} threads")
        per_frame, stats = detect_frames(
            detector, items, lambda p: data.load_frame(p), log=log, label=label
        )
        meta = {
            **detector.meta(),
            "config": spec.id,
            "size": spec.size,
            "precision": "int8" if spec.int8 else "fp32",
            "sequence": name,
            "frame_rate": info.frame_rate,
            "image_width": info.width,
            "image_height": info.height,
            "n_bad_frames": len(stats["bad_frames"]),
            **stats,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".partial.npz")
        save_cache(tmp, per_frame, meta)
        tmp.replace(out)
        log(
            f"{label} done: {stats['sec_per_frame']:.3f} s/frame "
            f"({stats['detect_sec_per_frame']:.3f} detect), "
            f"{len(stats['bad_frames'])} bad frames -> {out}"
        )
        written.append(out)
    return written
