"""Count-error and event-matching metrics (numpy + scipy only)."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from vidan.counting import Crossing, LineCount


def count_errors(pred: LineCount, gt: LineCount) -> dict[str, int]:
    """Signed errors `pred - gt` for in, out and total (in + out) crossings."""
    return {
        "in": pred.n_in - gt.n_in,
        "out": pred.n_out - gt.n_out,
        "total": (pred.n_in + pred.n_out) - (gt.n_in + gt.n_out),
    }


def mae(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Mean absolute error of paired counts; nan for empty input."""
    p, g = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")
    return float(np.mean(np.abs(p - g))) if p.size else float("nan")


def wape(pred: Sequence[float], gt: Sequence[float]) -> float:
    """Weighted absolute percentage error: sum|pred - gt| / sum(gt); nan when sum(gt) is 0."""
    p, g = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")
    denom = float(g.sum())
    return float(np.abs(p - g).sum() / denom) if denom != 0 else float("nan")


@dataclass(frozen=True)
class EventMatch:
    """Result of matching predicted crossings to ground-truth crossings.

    `cancelled = min(fp, fn)`: errors a net count cannot see (a miss offset by a false alarm).
    `pairs` holds the matched `(pred, gt)` crossings.
    """

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    cancelled: int
    pairs: tuple[tuple[Crossing, Crossing], ...] = ()


def _ratio(num: int, den: int, both_empty: bool) -> float:
    if den:
        return num / den
    return 1.0 if both_empty else 0.0


def match_events(
    pred_events: Sequence[Crossing], gt_events: Sequence[Crossing], tol_frames: int
) -> EventMatch:
    """Match crossings one-to-one within `tol_frames`, separately per direction.

    Within a direction the Hungarian algorithm minimises total `|frame difference|`; a pair
    further apart than `tol_frames` is never a match (a difference equal to `tol_frames` is).
    Track ids are ignored, so a missed crossing by one person and a double count of another
    person close in time can match each other; `attribution.match_events_identity` forbids that.
    With no events on either side precision, recall and F1 are all 1.0;
    if only one side is empty the undefined ratio is 0.0.
    """
    if tol_frames < 0:
        raise ValueError("tol_frames must be >= 0")
    pairs: list[tuple[Crossing, Crossing]] = []
    for direction in (1, -1):
        p = [e for e in pred_events if e.direction == direction]
        g = [e for e in gt_events if e.direction == direction]
        if not p or not g:
            continue
        diff = np.abs(
            np.array([e.frame for e in p])[:, None] - np.array([e.frame for e in g])[None, :]
        )
        # Disallowed pairs cost more than any set of allowed ones, so the assignment first
        # maximises the number of in-tolerance matches and then minimises their total offset.
        big = min(len(p), len(g)) * tol_frames + 1
        cost = np.where(diff <= tol_frames, diff, big)
        rows, cols = linear_sum_assignment(cost)
        pairs.extend(
            (p[r], g[c]) for r, c in zip(rows, cols, strict=True) if diff[r, c] <= tol_frames
        )
    return event_match_from_pairs(pairs, len(pred_events), len(gt_events))


def event_match_from_pairs(
    pairs: Sequence[tuple[Crossing, Crossing]], n_pred: int, n_gt: int
) -> EventMatch:
    """Build an `EventMatch` from matched `(pred, gt)` pairs and the two event counts."""
    tp = len(pairs)
    fp = n_pred - tp
    fn = n_gt - tp
    both_empty = n_pred == 0 and n_gt == 0
    precision = _ratio(tp, tp + fp, both_empty)
    recall = _ratio(tp, tp + fn, both_empty)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EventMatch(tp, fp, fn, precision, recall, f1, min(fp, fn), tuple(pairs))
