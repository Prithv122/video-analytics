"""Unit tests for the quiet-machine gate, benchmark statistics and the equivalence maths.

Nothing here needs torch, onnxruntime or OpenCV.
"""

import numpy as np
import pytest

from vidan.detect import Detections
from vidan.speedbench import (
    above_threshold,
    equivalence_summary,
    evenly_spaced,
    iou_matrix,
    match_ious,
    quiet_gate,
    repeat_stats,
    sample_cpu_percent,
    summarise_repeats,
)


class FakeClock:
    """A clock that only moves when `sleep` is called."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def _gate(readings: list[float], max_wait_s: float, retry_s: float = 30.0, threshold: float = 15.0):
    clock = FakeClock()
    it = iter(readings)
    res = quiet_gate(
        lambda: next(it),
        threshold=threshold,
        max_wait_s=max_wait_s,
        retry_s=retry_s,
        now=clock.now,
        sleep=clock.sleep,
    )
    return res, clock


def test_gate_passes_on_first_quiet_reading_without_waiting() -> None:
    res, clock = _gate([4.0], max_wait_s=600)
    assert res.passed and res.attempts == 1 and res.cpu_pct == 4.0
    assert clock.sleeps == []


def test_gate_threshold_is_strict() -> None:
    res, _ = _gate([15.0], max_wait_s=0)
    assert not res.passed
    res, _ = _gate([14.99], max_wait_s=0)
    assert res.passed


def test_gate_retries_until_quiet() -> None:
    res, clock = _gate([80.0, 40.0, 9.0], max_wait_s=600, retry_s=30)
    assert res.passed and res.attempts == 3 and res.cpu_pct == 9.0
    assert clock.sleeps == [30.0, 30.0] and res.waited_s == 60.0


def test_gate_gives_up_when_the_wait_budget_is_spent() -> None:
    res, clock = _gate([90.0] * 10, max_wait_s=90, retry_s=30)
    assert not res.passed and res.cpu_pct == 90.0
    assert res.attempts == 4 and res.waited_s == 90.0  # readings at t = 0, 30, 60, 90
    assert len(clock.sleeps) == 3


def test_gate_with_no_wait_budget_takes_exactly_one_reading() -> None:
    res, clock = _gate([50.0, 1.0], max_wait_s=0)
    assert not res.passed and res.attempts == 1 and clock.sleeps == []


def test_sample_cpu_percent_returns_a_percentage() -> None:
    pytest.importorskip("psutil")
    assert 0.0 <= sample_cpu_percent(seconds=1) <= 100.0


def test_repeat_stats_in_milliseconds() -> None:
    s = repeat_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    assert s["mean_ms"] == pytest.approx(300.0)
    assert s["p50_ms"] == pytest.approx(300.0)
    assert s["p95_ms"] == pytest.approx(480.0)


def test_summary_is_median_of_repeat_means_with_pooled_percentiles() -> None:
    repeats = [[0.1, 0.1], [0.2, 0.2], [0.9, 0.9]]  # means 100, 200, 900 ms
    s = summarise_repeats(repeats)
    assert s["mean_ms"] == pytest.approx(200.0)  # median, not the 400 ms average
    assert s["p50_ms"] == pytest.approx(200.0)
    assert s["p95_ms"] == pytest.approx(900.0)


def test_iou_matrix_by_hand() -> None:
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m[0], [1.0, 50 / 150, 0.0])
    assert iou_matrix(np.empty((0, 4)), b).shape == (0, 3)


def _det(boxes, scores=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.full(len(boxes), 0.9) if scores is None else scores
    return Detections(boxes, scores, np.ones(len(boxes), dtype=np.int64))


def test_match_ious_is_one_to_one_and_counts_unmatched() -> None:
    a = _det([[0, 0, 10, 10], [100, 100, 110, 110]])
    b = _det([[0, 0, 10, 10], [200, 200, 210, 210], [100, 100, 110, 112]])
    ious, ua, ub = match_ious(a, b)
    assert sorted(ious) == pytest.approx([100 / 120, 1.0])
    assert (ua, ub) == (0, 1)
    assert match_ious(_det([]), b) == ([], 0, 3)


def test_above_threshold_filters_on_score() -> None:
    d = _det([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], np.array([0.2, 0.5, 0.31]))
    assert above_threshold(d, 0.3).score.tolist() == pytest.approx([0.5, 0.31])


def test_equivalence_passes_when_boxes_and_counts_agree() -> None:
    frames = [
        (_det([[0, 0, 100, 100]]), _det([[0, 0, 100, 100.5]])),
        (
            _det([[10, 10, 50, 90], [200, 0, 260, 120]]),
            _det([[10, 10, 50, 90], [200, 0, 260, 120]]),
        ),
    ]
    s = equivalence_summary(frames)
    assert s["passed"] and s["n_frames"] == 2 and s["frac_count_equal"] == 1.0
    assert s["n_matched"] == 3 and s["iou_min"] >= 0.99


def test_equivalence_fails_on_one_poorly_matched_box() -> None:
    frames = [(_det([[0, 0, 100, 100]]), _det([[0, 0, 100, 95]]))]  # IoU 0.95
    s = equivalence_summary(frames)
    assert not s["passed"] and s["iou_min"] == pytest.approx(0.95)
    assert s["frac_iou_ge_min"] == 0.0


def test_equivalence_count_rule_is_95_percent_of_frames() -> None:
    same = (_det([[0, 0, 10, 10]]), _det([[0, 0, 10, 10]]))
    extra = (_det([[0, 0, 10, 10]]), _det([[0, 0, 10, 10], [50, 50, 60, 60]]))
    ok = equivalence_summary([same] * 19 + [extra])  # 19 / 20 = 95 %: passes
    assert ok["passed"] and ok["frac_count_equal"] == pytest.approx(0.95)
    assert ok["n_unmatched_onnx"] == 1
    bad = equivalence_summary([same] * 18 + [extra] * 2)  # 90 %
    assert not bad["passed"]


def test_equivalence_of_nothing_does_not_pass() -> None:
    s = equivalence_summary([(_det([]), _det([]))])
    assert s["n_matched"] == 0 and not s["passed"] and s["frac_count_equal"] == 1.0


def test_evenly_spaced_covers_first_and_last_without_repeats() -> None:
    idx = evenly_spaced(2175, 50)
    assert len(idx) == 50 and idx[0] == 0 and idx[-1] == 2174
    assert len(set(idx)) == 50 and idx == sorted(idx)
    assert evenly_spaced(10, 50) == list(range(10))
    assert evenly_spaced(10, 1) == [0] and evenly_spaced(0, 5) == []


def test_pipeline_fps_adds_tracking_time_per_processed_frame() -> None:
    from vidan.speedbench import pipeline_fps

    assert pipeline_fps(10.0, 0.1) == pytest.approx(5.0)
    assert pipeline_fps(10.0, 0.1, k=3) == pytest.approx(15.0)
    assert pipeline_fps(10.0, None) == pytest.approx(10.0)


def test_equivalence_csv_row_uses_the_column_names_vidan_eval_reads() -> None:
    from vidan.speedbench import EQUIV_COLUMNS, equiv_row

    frames = [(_det([[0, 0, 100, 100]]), _det([[0, 0, 100, 100]]))]
    row = equiv_row("N", 0.1, equivalence_summary(frames))
    assert row["config"] == "N" and row["passed"] is True
    assert row["min_matched_iou"] == pytest.approx(1.0)
    assert row["frac_frames_equal_count"] == 1.0
    assert set(EQUIV_COLUMNS) <= set(row)
