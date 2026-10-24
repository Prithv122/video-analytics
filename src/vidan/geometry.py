"""Lines, polygons and the point tests behind counting (numpy only).

Image coordinates: x right, y down. For a `Line` from p1 to p2 the side of a point is the sign
of the cross product `(p2 - p1) x (pt - p1)`; positive is the "in" side.
"""

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


def _pt(p) -> tuple[float, float]:
    x, y = p
    return float(x), float(y)


@dataclass(frozen=True)
class Line:
    """Directed segment from `p1` to `p2`."""

    p1: tuple[float, float]
    p2: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "p1", _pt(self.p1))
        object.__setattr__(self, "p2", _pt(self.p2))
        if self.p1 == self.p2:
            raise ValueError("Line endpoints must differ (zero-length line)")

    @property
    def length(self) -> float:
        return float(np.hypot(self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]))


@dataclass(frozen=True)
class Zone:
    """Simple polygon given by its vertices in order (closed implicitly)."""

    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        pts = tuple(_pt(p) for p in self.points)
        if len(pts) < 3:
            raise ValueError("Zone needs at least 3 vertices")
        object.__setattr__(self, "points", pts)


def _cross(line: Line, pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    return dx * (p[..., 1] - line.p1[1]) - dy * (p[..., 0] - line.p1[0])


def side(line: Line, pts):
    """Sign (+1, 0, -1) of the cross product; works on one point or an (n, 2) array."""
    return np.sign(_cross(line, pts)).astype(np.int64)


def signed_distance(line: Line, pts):
    """Cross product divided by the line length: perpendicular pixels, positive on the in side."""
    return _cross(line, pts) / line.length


def projection_t(line: Line, pts):
    """Projection parameter t of each point on the line (0 at p1, 1 at p2, unclamped)."""
    p = np.asarray(pts, dtype=np.float64)
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    return ((p[..., 0] - line.p1[0]) * dx + (p[..., 1] - line.p1[1]) * dy) / (dx * dx + dy * dy)


def projects_onto(line: Line, pts):
    """True where the point's projection lies on the segment itself (t in [0, 1], no extension)."""
    t = projection_t(line, pts)
    return (t >= 0.0) & (t <= 1.0)


def point_in_polygon(polygon: Zone | np.ndarray, pts):
    """Ray-casting point-in-polygon; points on the boundary count as inside.

    `polygon` is a `Zone` or an (m, 2) array of vertices; `pts` is one point or an (n, 2) array.
    """
    verts = np.asarray(polygon.points if isinstance(polygon, Zone) else polygon, dtype=np.float64)
    p = np.asarray(pts, dtype=np.float64)
    x, y = p[..., 0], p[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    on_edge = np.zeros(x.shape, dtype=bool)
    nxt = np.roll(verts, -1, axis=0)
    for (ax, ay), (bx, by) in zip(verts, nxt, strict=True):
        ex, ey = bx - ax, by - ay
        cross = ex * (y - ay) - ey * (x - ax)
        dot = (x - ax) * ex + (y - ay) * ey
        elen2 = ex * ex + ey * ey
        tol = _EPS * max(1.0, float(np.sqrt(elen2)))
        on_edge |= (np.abs(cross) <= tol) & (dot >= -tol) & (dot <= elen2 + tol)
        straddles = (ay > y) != (by > y)
        if ey != 0:
            x_cross = ax + (y - ay) * ex / ey
            inside ^= straddles & (x < x_cross)
    return inside | on_edge
