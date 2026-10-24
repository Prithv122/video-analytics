"""Unit tests for frame selection and the benchmark harness (no vision extra needed)."""

import numpy as np
import pytest

from vidan.bench import benchmark, env_fingerprint, latency_stats
from vidan.detect import Detections
from vidan.track import select_frames


def test_select_frames_keeps_first_and_every_kth() -> None:
    frames = list(range(1, 11))
    assert select_frames(frames, 1) == frames
    assert select_frames(frames, 3) == [1, 4, 7, 10]
    assert select_frames(frames, 20) == [1]


def test_select_frames_sorts_and_respects_gaps() -> None:
    assert select_frames([9, 2, 5, 1, 4], 2) == [1, 5, 9]
    assert select_frames([], 2) == []
    with pytest.raises(ValueError):
        select_frames([1, 2], 0)


def test_latency_stats_by_hand() -> None:
    s = latency_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert s["mean"] == pytest.approx(0.3)
    assert s["p50"] == pytest.approx(0.3)
    assert s["p95"] == pytest.approx(0.48)  # linear interpolation: 0.4 + 0.8 * 0.1
    assert s["fps"] == pytest.approx(1 / 0.3)
    with pytest.raises(ValueError):
        latency_stats([])


class _Fake:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> Detections:
        self.calls += 1
        return Detections([[0, 0, 1, 1]], [0.9], [1])


def test_benchmark_counts_warmup_and_repeats() -> None:
    det = _Fake()
    frames = [np.zeros((4, 4, 3), np.uint8)]
    out = benchmark(det, frames, warmup=2, repeats=5)
    assert det.calls == 7
    assert len(out["latencies"]) == 5 and all(t >= 0 for t in out["latencies"])
    assert out["fps"] == pytest.approx(1 / out["mean"])
    assert out["mean_detections"] == 1.0 and out["detector"] == "fake"
    with pytest.raises(ValueError):
        benchmark(det, [], warmup=0, repeats=1)


def test_env_fingerprint_has_the_reported_fields() -> None:
    fp = env_fingerprint()
    for key in ("cpu", "logical_cores", "os", "onnxruntime", "torch", "rfdetr", "torch_threads"):
        assert key in fp
    assert fp["cpu"] and fp["logical_cores"] >= 1
