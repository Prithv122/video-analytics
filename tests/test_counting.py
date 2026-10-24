import numpy as np
import pytest

from vidan.counting import Crossing, LineCount, Rule, count_line, zone_occupancy
from vidan.geometry import Line, Zone, projects_onto, signed_distance
from vidan.mot import Tracks, tracks_from_rows

LINE = Line((0, 0), (100, 0))  # in side is y > 0
NAIVE = Rule.naive()


def make(*obs: tuple[int, int, float, float]) -> Tracks:
    """Tracks whose bottom-center anchors are the given (frame, id, x, y) points."""
    return tracks_from_rows([(f, i, x - 1, y - 2, 2, 2) for f, i, x, y in obs])


def path(tid: int, pts: list[tuple[float, float]], start: int = 1) -> list:
    return [(start + k, tid, x, y) for k, (x, y) in enumerate(pts)]


def test_clean_crossing_in():
    r = count_line(make(*path(1, [(50, -5), (50, 5)])), LINE, NAIVE)
    assert (r.n_in, r.n_out) == (1, 0)
    assert r.events == (Crossing(1, 2, 1),)


def test_clean_crossing_out():
    r = count_line(make(*path(1, [(50, 5), (50, 3), (50, -3)])), LINE, NAIVE)
    assert (r.n_in, r.n_out) == (0, 1)
    assert r.events == (Crossing(1, 3, -1),)


def test_no_crossing_when_staying_on_one_side():
    r = count_line(make(*path(1, [(50, 5), (50, 1), (50, 9)])), LINE, NAIVE)
    assert r == LineCount(0, 0, ())


def test_wobble_naive_counts_three_hysteresis_counts_one():
    wobble = make(*path(1, [(50, 20), (50, -5), (50, 5), (50, -5), (50, -20)]))
    naive = count_line(wobble, LINE, NAIVE)
    assert [(e.frame, e.direction) for e in naive.events] == [(2, -1), (3, 1), (4, -1)]
    assert (naive.n_in, naive.n_out) == (1, 2)
    hyst = count_line(wobble, LINE, Rule.hysteresis(10))
    assert hyst.events == (Crossing(1, 5, -1),)
    assert (hyst.n_in, hyst.n_out) == (0, 1)


def test_hysteresis_dip_into_the_band_and_back_counts_nothing():
    r = count_line(make(*path(1, [(50, 20), (50, -5), (50, 15)])), LINE, Rule.hysteresis(10))
    assert r.events == ()


def test_crossing_outside_the_segment_is_not_counted():
    for rule in (NAIVE, Rule.hysteresis(2)):
        assert count_line(make(*path(1, [(150, -5), (150, 5)])), LINE, rule).events == ()
        assert count_line(make(*path(1, [(-20, 5), (-20, -5)])), LINE, rule).events == ()


def test_segment_endpoints_are_inside():
    assert count_line(make(*path(1, [(0, -5), (100, 5)])), LINE, NAIVE).n_in == 1


def test_walking_around_the_segment_end_is_not_counted():
    # crosses the line's extension, never the segment: out-of-segment points reset the state
    around = make(*path(1, [(50, -5), (150, -5), (150, 5), (50, 5)]))
    assert count_line(around, LINE, NAIVE).events == ()
    assert count_line(around, LINE, Rule.hysteresis(3)).events == ()


def test_gap_in_a_track_still_counts_at_the_next_observation():
    t = make((1, 1, 50, -5), (10, 1, 50, 5))
    assert count_line(t, LINE, NAIVE).events == (Crossing(1, 10, 1),)


def test_point_exactly_on_the_line_keeps_the_previous_side():
    stay = make(*path(1, [(50, -5), (50, 0), (50, -5)]))
    assert count_line(stay, LINE, NAIVE).events == ()
    cross = make(*path(1, [(50, -5), (50, 0), (50, 5)]))
    assert count_line(cross, LINE, NAIVE).events == (Crossing(1, 3, 1),)


def test_two_ids_counted_independently_and_events_sorted():
    t = make(*path(1, [(10, -5), (10, 5)]), *path(2, [(90, 5), (90, -5)]))
    r = count_line(t, LINE, NAIVE)
    assert (r.n_in, r.n_out, r.net) == (1, 1, 0)
    assert r.events == (Crossing(1, 2, 1), Crossing(2, 2, -1))


def test_id_switch_after_the_line_loses_the_crossing():
    # A stops before the line, B appears after it: nobody crosses
    t = make(*path(1, [(50, -9), (50, -6), (50, -3)]), *path(2, [(50, 3), (50, 6)], start=4))
    assert count_line(t, LINE, NAIVE).events == ()
    assert count_line(t, LINE, Rule.hysteresis(2)).events == ()


def test_id_switch_before_the_line_still_counts_once():
    # A ends before the line; B starts just before it and crosses
    t = make(*path(1, [(50, -9), (50, -6)]), *path(2, [(50, -4), (50, -1), (50, 4)], start=3))
    assert count_line(t, LINE, NAIVE).events == (Crossing(2, 5, 1),)


def test_anchor_choice_changes_the_result():
    # 8 px tall boxes: the bottom edge crosses the line, the centre does not
    t = tracks_from_rows([(1, 1, 49, -10, 2, 8), (2, 1, 49, -6, 2, 8)])
    assert count_line(t, LINE, NAIVE).n_in == 1  # bottom: -2 -> +2
    assert count_line(t, LINE, NAIVE, anchor="center").n_in == 0  # centre: -6 -> -2


def test_empty_tracks_and_rule_validation():
    assert count_line(Tracks.empty(), LINE, NAIVE) == LineCount(0, 0, ())
    with pytest.raises(ValueError):
        Rule.hysteresis(-1)
    assert Rule.naive() == Rule.hysteresis(0)
    assert Rule.naive().name == "naive"
    assert Rule.hysteresis(10).name == "hysteresis(10)"


def reference_naive(tracks: Tracks, line: Line) -> list[tuple[int, int, int]]:
    """Independent restatement: a step between consecutive observations of an id counts when
    both endpoints project inside the segment and their (non-zero) sides differ."""
    pts = tracks.anchor()
    out = []
    for tid in sorted(set(tracks.track_id.tolist())):
        rows = sorted(
            (int(tracks.frame[i]), i) for i in range(len(tracks)) if tracks.track_id[i] == tid
        )
        prev_side = None
        for frame, i in rows:
            if not bool(projects_onto(line, pts[i])):
                prev_side = None
                continue
            d = float(signed_distance(line, pts[i]))
            if d == 0:
                continue
            s = 1 if d > 0 else -1
            if prev_side is not None and s != prev_side:
                out.append((frame, tid, s))
            prev_side = s
    return sorted(out)


def test_band_zero_equals_naive_and_matches_reference_on_random_tracks():
    rng = np.random.default_rng(0)
    line = Line((0, 0), (100, 0))
    total = 0
    for _ in range(25):
        n = int(rng.integers(20, 60))
        # integer anchors on a small grid so on-the-line and segment-end cases occur
        rows = {}
        for _ in range(n):
            frame, tid = int(rng.integers(1, 30)), int(rng.integers(1, 5))
            x, y = float(rng.integers(-10, 111)), float(rng.integers(-3, 4))
            rows[(frame, tid)] = (frame, tid, x - 1, y - 2, 2, 2)
        tracks = tracks_from_rows(list(rows.values()))
        a = count_line(tracks, line, Rule.naive())
        b = count_line(tracks, line, Rule.hysteresis(0))
        assert a.events == b.events
        got = sorted((e.frame, e.track_id, e.direction) for e in a.events)
        assert got == reference_naive(tracks, line)
        total += len(got)
    assert total > 50  # the fixtures are not vacuous


def test_zone_occupancy_per_frame():
    zone = Zone([(0, 0), (10, 0), (10, 10), (0, 10)])
    t = make(
        (1, 1, 5, 5),  # inside
        (1, 2, 20, 5),  # outside
        (2, 1, 10, 10),  # corner counts as inside
        (2, 2, 5, 0),  # edge counts as inside
        (3, 1, 11, 5),
        (3, 2, 30, 30),
    )
    assert zone_occupancy(t, zone) == {1: 1, 2: 2, 3: 0}
    assert zone_occupancy(Tracks.empty(), zone) == {}
