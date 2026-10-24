import math

import pytest

from vidan.counting import Crossing, LineCount
from vidan.metrics import count_errors, mae, match_events, wape


def ev(*items: tuple[int, int]) -> list[Crossing]:
    """(frame, direction) pairs -> crossings with distinct ids."""
    return [Crossing(k + 1, f, d) for k, (f, d) in enumerate(items)]


def test_count_errors_signed():
    pred = LineCount(5, 2, ())
    gt = LineCount(3, 4, ())
    assert count_errors(pred, gt) == {"in": 2, "out": -2, "total": 0}


def test_mae_and_wape():
    assert mae([3, 5], [4, 2]) == 2.0
    assert wape([3, 5], [4, 2]) == pytest.approx(4 / 6)
    assert math.isnan(wape([1, 2], [0, 0]))
    assert math.isnan(mae([], []))
    with pytest.raises(ValueError):
        mae([1], [1, 2])


def test_perfect_match():
    m = match_events(ev((10, 1), (50, -1)), ev((12, 1), (49, -1)), tol_frames=5)
    assert (m.tp, m.fp, m.fn, m.cancelled) == (2, 0, 0, 0)
    assert m.precision == m.recall == m.f1 == 1.0
    assert len(m.pairs) == 2


def test_net_count_exact_but_f1_below_one_with_cancelled_error():
    # GT has one crossing at frame 100; the prediction has a spurious one at frame 10 instead.
    pred, gt = ev((10, 1)), ev((100, 1))
    errors = count_errors(LineCount(1, 0, tuple(pred)), LineCount(1, 0, tuple(gt)))
    assert errors["total"] == 0
    m = match_events(pred, gt, tol_frames=25)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)
    assert m.cancelled == 1
    assert m.f1 == 0.0 and m.precision == 0.0 and m.recall == 0.0


def test_partial_f1():
    m = match_events(ev((10, 1), (20, 1), (900, 1)), ev((11, 1), (21, 1), (300, 1)), tol_frames=5)
    assert (m.tp, m.fp, m.fn, m.cancelled) == (2, 1, 1, 1)
    assert m.f1 == pytest.approx(2 / 3)


def test_tolerance_boundary_is_inclusive():
    assert match_events(ev((10, 1)), ev((15, 1)), tol_frames=5).tp == 1
    assert match_events(ev((10, 1)), ev((16, 1)), tol_frames=5).tp == 0
    assert match_events(ev((10, 1)), ev((10, 1)), tol_frames=0).tp == 1
    assert match_events(ev((10, 1)), ev((11, 1)), tol_frames=0).tp == 0


def test_directions_are_matched_separately():
    m = match_events(ev((10, 1)), ev((10, -1)), tol_frames=5)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_assignment_maximises_in_tolerance_matches():
    # pred 10, 14 / gt 12, 18 / tol 4. Pairing the closest first (14-12) strands 10 and 18;
    # the optimum is 10-12 and 14-18.
    m = match_events(ev((10, 1), (14, 1)), ev((12, 1), (18, 1)), tol_frames=4)
    assert m.tp == 2
    assert {(p.frame, g.frame) for p, g in m.pairs} == {(10, 12), (14, 18)}


def test_extra_prediction_near_one_gt_matches_only_once():
    m = match_events(ev((10, 1), (11, 1)), ev((10, 1)), tol_frames=5)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.pairs[0][0].frame == 10  # the closer one wins


def test_empty_inputs():
    both = match_events([], [], tol_frames=5)
    assert (both.tp, both.fp, both.fn, both.cancelled) == (0, 0, 0, 0)
    assert both.precision == both.recall == both.f1 == 1.0
    only_gt = match_events([], ev((5, 1)), tol_frames=5)
    assert (only_gt.fn, only_gt.precision, only_gt.recall, only_gt.f1) == (1, 0.0, 0.0, 0.0)
    only_pred = match_events(ev((5, 1)), [], tol_frames=5)
    assert (only_pred.fp, only_pred.precision, only_pred.recall, only_pred.f1) == (1, 0.0, 0.0, 0.0)
    assert only_pred.cancelled == 0


def test_negative_tolerance_rejected():
    with pytest.raises(ValueError):
        match_events([], [], tol_frames=-1)
