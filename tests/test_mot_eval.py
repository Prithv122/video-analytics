import math

import pytest

from vidan.mot import Tracks, tracks_from_rows
from vidan.mot_eval import evaluate


def walk(tid: int, frames, x0: float = 0.0) -> list:
    return [(f, tid, x0 + 2 * f, 10, 20, 40) for f in frames]


def test_perfect_tracking():
    gt = tracks_from_rows(walk(1, range(1, 11)) + walk(2, range(1, 11), 300))
    m = evaluate(gt, gt)
    assert m["mota"] == 1.0 and m["idf1"] == 1.0
    assert m["num_switches"] == 0 and m["num_fragmentations"] == 0
    assert m["precision"] == m["recall"] == 1.0
    assert m["motp"] == pytest.approx(0.0)
    assert (m["num_frames"], m["num_objects"], m["num_predictions"]) == (10, 20, 20)
    assert m["mostly_tracked"] == 2 and m["mostly_lost"] == 0


def test_one_id_switch():
    gt = tracks_from_rows(walk(1, range(1, 11)))
    pred = tracks_from_rows(walk(5, range(1, 6)) + walk(6, range(6, 11)))
    m = evaluate(gt, pred)
    assert m["num_switches"] == 1
    assert m["mota"] == pytest.approx(0.9)  # 1 - 1 switch / 10 objects
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["idf1"] == pytest.approx(0.5)


def test_frames_without_predictions_are_misses():
    gt = tracks_from_rows(walk(1, range(1, 11)))
    pred = tracks_from_rows(walk(1, range(1, 6)))  # nothing after frame 5
    m = evaluate(gt, pred)
    assert m["num_frames"] == 10 and m["num_objects"] == 10
    assert m["recall"] == pytest.approx(0.5)
    assert m["mota"] == pytest.approx(0.5)


def test_iou_threshold_and_false_positives():
    gt = tracks_from_rows([(1, 1, 0, 0, 10, 10)])
    near = tracks_from_rows([(1, 9, 5, 0, 10, 10)])  # IoU 1/3
    assert evaluate(gt, near, iou_thr=0.5)["recall"] == 0.0
    assert evaluate(gt, near, iou_thr=0.3)["recall"] == 1.0
    m = evaluate(gt, tracks_from_rows([(1, 9, 100, 100, 10, 10)]))
    assert m["num_predictions"] == 1 and m["precision"] == 0.0 and m["mota"] == -1.0


def test_empty_inputs():
    gt = tracks_from_rows(walk(1, range(1, 4)))
    m = evaluate(gt, Tracks.empty())
    assert m["recall"] == 0.0 and m["mostly_lost"] == 1 and m["num_objects"] == 3
    both = evaluate(Tracks.empty(), Tracks.empty())
    assert both["num_frames"] == 0 and math.isnan(both["mota"])
