import numpy as np
import pytest

from vidan.mot import Tracks, read_mot, tracks_from_rows, write_mot


def test_tracks_from_rows_sorts_by_frame_then_id():
    t = tracks_from_rows([(2, 1, 0, 0, 4, 4), (1, 2, 0, 0, 4, 4), (1, 1, 0, 0, 4, 4, 0.5)])
    assert t.frame.tolist() == [1, 1, 2]
    assert t.track_id.tolist() == [1, 2, 1]
    assert t.score.tolist() == [0.5, 1.0, 1.0]
    assert len(t) == 3
    assert t.frame.dtype == np.int64 and t.xywh.dtype == np.float64


def test_unequal_lengths_rejected():
    with pytest.raises(ValueError):
        Tracks(np.array([1, 2]), np.array([1]), np.zeros((2, 4)), np.ones(2))
    with pytest.raises(ValueError):
        Tracks(np.array([1]), np.array([1]), np.zeros((1, 3)), np.ones(1))


def test_anchor_and_by_id():
    t = tracks_from_rows([(1, 7, 10, 20, 4, 8), (2, 7, 12, 20, 4, 8), (1, 3, 0, 0, 2, 2)])
    # sorted order: (1,3), (1,7), (2,7)
    assert t.anchor().tolist() == [[1.0, 2.0], [12.0, 28.0], [14.0, 28.0]]
    assert t.anchor("center").tolist() == [[1.0, 1.0], [12.0, 24.0], [14.0, 24.0]]
    ids = t.by_id()
    assert list(ids) == [3, 7]
    assert ids[7].tolist() == [1, 2]
    with pytest.raises(ValueError):
        t.anchor("top")


def test_by_id_orders_unsorted_input_by_frame():
    t = Tracks(np.array([5, 1, 3]), np.array([1, 1, 1]), np.zeros((3, 4)), np.ones(3))
    assert t.by_id()[1].tolist() == [1, 2, 0]


def test_clip_to_the_frame_drops_empty_boxes_and_is_idempotent():
    t = tracks_from_rows(
        [
            (1, 1, -10, 20, 30, 40),  # sticks out on the left: x 0..20
            (1, 2, 90, 90, 30, 30),  # sticks out bottom right: 90..100
            (2, 3, 150, 10, 10, 10),  # fully outside
            (2, 4, 5, 110, 10, 10),  # fully below
            (2, 5, 10, 10, 20, 20),  # inside, untouched
        ]
    )
    c = t.clip(100, 100)
    assert c.track_id.tolist() == [1, 2, 5]
    assert c.xywh.tolist() == [[0, 20, 20, 40], [90, 90, 10, 10], [10, 10, 20, 20]]
    assert c.frame.tolist() == [1, 1, 2]
    again = c.clip(100, 100)
    assert np.array_equal(again.xywh, c.xywh)
    assert len(Tracks.empty().clip(10, 10)) == 0


def test_empty_cases(tmp_path):
    e = tracks_from_rows([])
    assert len(e) == 0 and e.by_id() == {} and e.anchor().shape == (0, 2)
    p = tmp_path / "empty.txt"
    p.write_text("")
    assert len(read_mot(p)) == 0
    assert len(read_mot(p, gt=True)) == 0
    write_mot(p, e)
    assert p.read_text() == ""


def test_write_format_and_round_trip(tmp_path):
    t = tracks_from_rows([(3, 2, 10.25, 20.5, 30.125, 40.0, 0.875), (1, 1, 1, 2, 3, 4, 0.5)])
    p = tmp_path / "pred.txt"
    write_mot(p, t)
    first = p.read_text().splitlines()[0]
    assert first == "1,1,1.0,2.0,3.0,4.0,0.5,-1,-1,-1"
    back = read_mot(p)
    assert np.array_equal(back.frame, t.frame)
    assert np.array_equal(back.track_id, t.track_id)
    assert np.array_equal(back.xywh, t.xywh)
    assert np.array_equal(back.score, t.score)


GT_ROWS = """\
1,1,10,10,5,5,1,1,1.0
1,2,20,20,5,5,0,1,1.0
1,3,30,30,5,5,1,2,1.0
1,4,40,40,5,5,1,1,0.2
2,1,11,10,5,5,1,1,0.9
"""


def test_gt_filters_consider_class_visibility(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text(GT_ROWS)
    t = read_mot(p, gt=True)
    assert t.track_id.tolist() == [1, 4, 1]  # id 2 not considered, id 3 wrong class
    assert t.score.tolist() == [1.0, 1.0, 1.0]
    vis = read_mot(p, gt=True, min_visibility=0.5)
    assert vis.track_id.tolist() == [1, 1]
    cls2 = read_mot(p, gt=True, classes=(1, 2))
    assert sorted(cls2.track_id.tolist()) == [1, 1, 3, 4]
    only2 = read_mot(p, gt=True, classes=(2,))
    assert only2.track_id.tolist() == [3]


def test_nine_column_file_without_gt_flag_and_single_row(tmp_path):
    p = tmp_path / "gt.txt"
    p.write_text("1,1,10,10,5,5,1,1,1.0\n")
    t = read_mot(p)  # unfiltered; score is column 7
    assert len(t) == 1 and t.score.tolist() == [1.0]
    with pytest.raises(ValueError):
        read_mot(p, min_visibility=0.5)


def test_gt_with_too_few_columns_rejected(tmp_path):
    p = tmp_path / "short.txt"
    p.write_text("1,1,1,2,3,4,0.5\n")
    with pytest.raises(ValueError):
        read_mot(p, gt=True)
