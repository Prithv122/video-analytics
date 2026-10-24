"""Line-crossing counts and zone occupancy from `Tracks` (numpy only)."""

from dataclasses import dataclass

import numpy as np

from vidan.geometry import Line, Zone, point_in_polygon, projects_onto, signed_distance
from vidan.mot import Tracks


@dataclass(frozen=True)
class Crossing:
    """One counted crossing. `direction` is +1 (to the in side) or -1; `frame` completed it."""

    track_id: int
    frame: int
    direction: int


@dataclass(frozen=True)
class LineCount:
    n_in: int
    n_out: int
    events: tuple[Crossing, ...]

    @property
    def net(self) -> int:
        return self.n_in - self.n_out


@dataclass(frozen=True)
class Rule:
    """Crossing rule: a per-track state machine with a dead band of `band_px` around the line.

    A track's state is in (+1), out (-1) or unknown. An observation sets the state to in when its
    signed distance is `> band_px` and to out when it is `< -band_px`; inside the band the state
    is left alone (so a point exactly on the line has no side and keeps the previous one). A
    crossing is a change from one known state to the other. An observation whose projection falls
    outside the segment resets the state to unknown, so a crossing needs the state-setting
    observation and the completing observation both to project inside the segment, with only
    in-segment or in-band observations between them. `naive()` is `band_px = 0`.
    """

    band_px: float = 0.0

    def __post_init__(self) -> None:
        if not self.band_px >= 0:
            raise ValueError(f"band_px must be >= 0, got {self.band_px}")

    @classmethod
    def naive(cls) -> "Rule":
        return cls(0.0)

    @classmethod
    def hysteresis(cls, band_px: float) -> "Rule":
        return cls(float(band_px))

    @property
    def name(self) -> str:
        return "naive" if self.band_px == 0 else f"hysteresis({self.band_px:g})"


def count_line(
    tracks: Tracks, line: Line, rule: Rule | None = None, anchor: str = "bottom_center"
) -> LineCount:
    """Count crossings of `line` by each track's anchor under `rule` (default naive).

    Gaps in a track are fine: consecutive observations of the same id are compared whatever the
    frame distance between them. Events are sorted by (frame, track_id).
    """
    rule = rule or Rule.naive()
    events: list[Crossing] = []
    if len(tracks):
        pts = tracks.anchor(anchor)
        dist = signed_distance(line, pts)
        inside = projects_onto(line, pts)
        band = rule.band_px
        state_of = np.where(dist > band, 1, np.where(dist < -band, -1, 0))
        for tid, idx in tracks.by_id().items():
            state = 0
            for i in idx:
                if not inside[i]:
                    state = 0
                    continue
                new = int(state_of[i])
                if new == 0:
                    continue
                if state and new != state:
                    events.append(Crossing(tid, int(tracks.frame[i]), new))
                state = new
    events.sort(key=lambda e: (e.frame, e.track_id))
    n_in = sum(e.direction > 0 for e in events)
    return LineCount(n_in, len(events) - n_in, tuple(events))


def zone_occupancy(tracks: Tracks, zone: Zone, anchor: str = "bottom_center") -> dict[int, int]:
    """Number of anchors inside `zone` for every frame that has any observation."""
    if len(tracks) == 0:
        return {}
    inside = point_in_polygon(zone, tracks.anchor(anchor))
    frames, inverse = np.unique(tracks.frame, return_inverse=True)
    counts = np.bincount(inverse, weights=inside, minlength=len(frames)).astype(int)
    return {int(f): int(c) for f, c in zip(frames, counts, strict=True)}
