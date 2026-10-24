import numpy as np
import pytest

from vidan.geometry import (
    Line,
    Zone,
    point_in_polygon,
    projection_t,
    projects_onto,
    side,
    signed_distance,
)

H = Line((0, 0), (100, 0))  # y > 0 is the "in" side


def test_zero_length_line_rejected():
    with pytest.raises(ValueError):
        Line((5, 5), (5, 5))


def test_side_and_signed_distance():
    pts = np.array([[50, 10], [50, -4], [50, 0]])
    assert side(H, pts).tolist() == [1, -1, 0]
    assert signed_distance(H, pts).tolist() == [10.0, -4.0, 0.0]
    assert side(H, (3, 2)) == 1


def test_signed_distance_is_in_pixels_for_diagonal_line():
    diag = Line((0, 0), (3, 4))  # length 5
    # cross = 3*y - 4*x ; point (0, 5) -> 15 -> distance 3
    assert signed_distance(diag, (0, 5)) == pytest.approx(3.0)
    assert signed_distance(diag, (3, 4)) == pytest.approx(0.0)
    assert signed_distance(Line((100, 0), (0, 0)), (50, 10)) == -10.0  # reversing flips sides


def test_projection_parameter_and_segment_test():
    pts = np.array([[0, 5], [100, 5], [50, 5], [-1, 5], [101, 5]])
    assert projection_t(H, pts).tolist() == [0.0, 1.0, 0.5, -0.01, 1.01]
    assert projects_onto(H, pts).tolist() == [True, True, True, False, False]
    assert bool(projects_onto(H, (200, 1))) is False


def test_point_in_polygon_inside_outside_boundary():
    sq = Zone([(0, 0), (10, 0), (10, 10), (0, 10)])
    pts = np.array([[5, 5], [11, 5], [-1, 5], [5, 11], [0, 5], [10, 10], [5, 0], [0, 0]])
    assert point_in_polygon(sq, pts).tolist() == [True, False, False, False, True, True, True, True]
    assert bool(point_in_polygon(sq, (5, 5))) is True


def test_point_in_concave_polygon():
    # "L" shape: the notch at the top right is outside
    ell = np.array([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])
    pts = np.array([[2, 8], [8, 2], [8, 8], [4, 7], [5, 5]])
    assert point_in_polygon(ell, pts).tolist() == [True, True, False, True, False]


def test_zone_validation_and_empty_points():
    with pytest.raises(ValueError):
        Zone([(0, 0), (1, 1)])
    sq = Zone([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert point_in_polygon(sq, np.empty((0, 2))).shape == (0,)
