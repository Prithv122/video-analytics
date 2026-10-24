"""Latency and FPS harness with warm-up and an environment fingerprint."""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Sequence
from importlib import metadata
from typing import Any

import numpy as np

from vidan.detect import Detector


def latency_stats(latencies: Sequence[float]) -> dict[str, float]:
    """Mean, p50, p95 (seconds, linear interpolation) and fps = 1 / mean."""
    arr = np.asarray(latencies, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no latencies")
    mean = float(arr.mean())
    return {
        "mean": mean,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "fps": 1.0 / mean,
    }


def set_threads(n: int) -> int:
    """Set torch's intra-op thread count explicitly (if torch is installed); returns it."""
    try:
        import torch
    except ImportError:
        return n
    torch.set_num_threads(n)
    return torch.get_num_threads()


def benchmark(
    detector: Detector,
    frames: list[np.ndarray],
    warmup: int = 5,
    repeats: int = 20,
    threads: int | None = None,
) -> dict[str, Any]:
    """Time `detector.detect` over `frames` (cycled): `warmup` untimed calls, then `repeats` timed.

    Each timed call is one frame, measured with `time.perf_counter`, end to end (pre-processing,
    model, decoding). `threads`, if given, sets torch's thread count first; ONNX Runtime's is fixed
    when its session is built. Returns the per-frame latency list (seconds), mean / p50 / p95,
    fps = 1 / mean, and the detector's own `meta()` if it has one.
    """
    if not frames:
        raise ValueError("frames is empty")
    if threads is not None:
        set_threads(threads)
    for i in range(warmup):
        detector.detect(frames[i % len(frames)])
    latencies: list[float] = []
    n_det = 0
    for i in range(repeats):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        det = detector.detect(frame)
        latencies.append(time.perf_counter() - t0)
        n_det += len(det)
    meta = detector.meta() if hasattr(detector, "meta") else {}
    return {
        "detector": detector.name,
        "warmup": warmup,
        "repeats": repeats,
        "latencies": latencies,
        **latency_stats(latencies),
        "mean_detections": n_det / repeats,
        "detector_meta": meta,
    }


def _cpu_name() -> str:
    """CPU model string without shelling out: registry on Windows, /proc on Linux."""
    if platform.system() == "Windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def env_fingerprint() -> dict[str, Any]:
    """Where a timing was taken: CPU, cores, library versions, thread settings, OS."""
    info: dict[str, Any] = {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "onnxruntime": _version("onnxruntime"),
        "torch": _version("torch"),
        "rfdetr": _version("rfdetr"),
        "trackers": _version("trackers"),
        "supervision": _version("supervision"),
        "torch_threads": None,
        "torch_interop_threads": None,
        "env_threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
    except ImportError:
        pass
    return info
