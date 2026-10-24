import numpy as np

from vidan.attribution import attribute_errors as attribute
from vidan.attribution import frame_assignment, iou_xywh
from vidan.counting import Crossing, Rule
from vidan.geometry import Line
from vidan.mot import Tracks, tracks_from_rows

LINE = Line((0, 0), (100, 0))  # in side is y > 0
NAIVE = Rule.naive()
TOL = 5


def boxes(tid: int, frames, yb: float, x: float = 45) -> list:
    """10x10 boxes whose bottom-center anchor sits at y = yb (x + 5 = 50 by default)."""
    return [(f, tid, x, yb - 10, 10, 10) for f in frames]


def tr(*groups) -> Tracks:
    return tracks_from_rows([row for g in groups for row in g])


def r(a: int, b: int) -> range:
    return range(a, b + 1)


def run(pred, gt, rule=NAIVE, **kw):
    return attribute(pred, gt, LINE, rule, TOL, **kw)


def person(tid=1, cross_at=6, end=11, lo=-5, hi=5):
    """GT person on the out side until cross_at - 1, then on the in side: one crossing."""
    return boxes(tid, r(1, cross_at - 1), lo) + boxes(tid, r(cross_at, end), hi)


def test_iou_xywh():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 5, 5]])
    iou = iou_xywh(a, b)
    assert iou.shape == (2, 3)
    assert iou[0].tolist() == [1.0, 50 / 150, 0.0]
    assert iou_xywh(np.empty((0, 4)), b).shape == (0, 3)
    assert iou_xywh(np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]]))[0, 0] == 0.0


def test_frame_assignment_picks_best_pairs_and_applies_threshold():
    gt = tr(boxes(1, [1], 10, 0), boxes(2, [1], 10, 100), boxes(3, [2], 10, 0))
    pred = tr(
        boxes(7, [1], 10, 2),  # overlaps gt 1
        boxes(8, [1], 10, 300),  # overlaps nothing
        boxes(9, [2], 10, 9),  # IoU with gt 3 below 0.5
    )
    assert frame_assignment(pred, gt) == {(1, 7): 1}
    assert frame_assignment(pred, gt, iou_thr=0.05) == {(1, 7): 1, (2, 9): 3}
    assert frame_assignment(Tracks.empty(), gt) == {}


def test_clean_case_has_no_errors():
    gt = tr(person(1))
    a = run(tr(person(7)), gt)
    assert a.match.tp == 1 and (a.match.fp, a.match.fn) == (0, 0)
    assert sum(a.fp.values()) == sum(a.fn.values()) == 0
    assert a.fp_events == [] and a.fn_events == []
    assert a.switch_attributed == 0


def test_duplicate_is_an_id_switch_double_count():
    # GT person hovers near the line: out (-1) until frame 8, in (+1) from 9, so one crossing at 9.
    gt = tr(boxes(1, r(1, 8), -1), boxes(1, r(9, 16), 1))
    # id 7 follows and crosses at 9; id 8 takes over (offset 2 px), dips out and crosses again at 11
    pred = tr(
        boxes(7, r(1, 8), -1),
        boxes(7, [9], 1),
        boxes(8, [10], -1),
        boxes(8, r(11, 16), 1),
    )
    a = run(pred, gt)
    assert (a.match.tp, a.match.fp, a.match.fn) == (1, 1, 0)
    assert a.fp_events == [(Crossing(8, 11, 1), "duplicate")]
    assert a.fp == {"duplicate": 1, "spurious": 0, "unmatched": 0}
    assert a.switch_attributed == 1
    assert a.match.cancelled == 0


def test_spurious_is_jitter_on_a_person_who_did_not_cross():
    gt = tr(boxes(1, r(1, 10), 1))
    pred = tr(boxes(7, r(1, 3), 1), boxes(7, [4], -1), boxes(7, r(5, 10), 1))
    a = run(pred, gt)
    assert [(e.frame, e.direction, k) for e, k in a.fp_events] == [
        (4, -1, "spurious"),
        (5, 1, "spurious"),
    ]
    assert a.fp["spurious"] == 2 and a.switch_attributed == 0
    # a band wider than the jitter removes both
    b = run(pred, gt, Rule.hysteresis(3))
    assert b.match.fp == 0 and sum(b.fp.values()) == 0


def test_unmatched_has_no_ground_truth_behind_it():
    pred = tr(boxes(9, r(1, 3), -5, 70), boxes(9, r(4, 6), 5, 70))
    a = run(pred, tr(boxes(1, r(1, 6), -5, 5)))  # the only GT person is far away, no crossing
    assert a.fp_events == [(Crossing(9, 4, 1), "unmatched")]
    assert a.fp["unmatched"] == 1
    assert run(pred, Tracks.empty()).fp["unmatched"] == 1


def test_tracked_not_counted_when_the_band_swallows_a_real_crossing():
    gt = tr(person(1))
    pred = tr(person(7))
    assert run(pred, gt).match.fn == 0
    a = run(pred, gt, Rule.hysteresis(10))  # +-5 px never leaves the band
    assert (a.match.tp, a.match.fp, a.match.fn) == (0, 0, 1)
    assert a.fn_events == [(Crossing(1, 6, 1), "tracked_not_counted")]
    assert a.fn["tracked_not_counted"] == 1
    # ground truth is still counted naively: a band on gt_rule would hide the miss
    hidden = run(pred, gt, Rule.hysteresis(10), gt_rule=Rule.hysteresis(10))
    assert hidden.match.fn == 0


def test_fragmented_when_the_identity_changes_at_the_line():
    gt = tr(person(1))
    pred = tr(boxes(7, r(1, 5), -5), boxes(8, r(6, 11), 5))
    a = run(pred, gt)
    assert a.fn_events == [(Crossing(1, 6, 1), "fragmented")]
    assert a.fp_events == []
    assert a.fn["fragmented"] == 1 and a.switch_attributed == 1


def test_missed_when_there_is_no_track_on_one_side():
    gt = tr(person(1))
    a = run(tr(boxes(7, r(1, 4), -5)), gt)  # track dies before the line
    assert a.fn_events == [(Crossing(1, 6, 1), "missed")]
    b = run(Tracks.empty(), gt)  # nothing detected at all
    assert b.fn == {"tracked_not_counted": 0, "fragmented": 0, "missed": 1}
    assert b.switch_attributed == 0


def test_categories_sum_to_match_counts_and_switch_property_adds_both_sides():
    # one fragmented FN (person 1) plus one duplicate FP (person 2) in the same sequence
    gt = tr(person(1), boxes(2, r(51, 58), -1, 200), boxes(2, r(59, 66), 1, 200))
    line = Line((0, 0), (300, 0))
    pred = tr(
        boxes(7, r(1, 5), -5),
        boxes(8, r(6, 11), 5),
        boxes(17, r(51, 58), -1, 200),
        boxes(17, [59], 1, 200),
        boxes(18, [60], -1, 200),
        boxes(18, r(61, 66), 1, 200),
    )
    a = attribute(pred, gt, line, NAIVE, TOL)
    assert sum(a.fp.values()) == a.match.fp == 1
    assert sum(a.fn.values()) == a.match.fn == 1
    assert (a.fp["duplicate"], a.fn["fragmented"]) == (1, 1)
    assert a.switch_attributed == 2
    assert a.match.cancelled == 1
