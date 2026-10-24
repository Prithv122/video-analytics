"""Attribute counting errors to their cause (numpy + scipy only).

Predicted and ground-truth crossings are matched with `match_events`; every unmatched event is
then explained using the per-frame box assignment between predicted ids and ground-truth ids.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vidan.counting import Crossing, Rule, count_line
from vidan.geometry import Line
from vidan.metrics import EventMatch, match_events
from vidan.mot import Tracks

FP_CATEGORIES = ("duplicate", "spurious", "unmatched")
FN_CATEGORIES = ("tracked_not_counted", "fragmented", "missed")


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (n, 4) and (m, 4) top-left xywh boxes -> (n, m)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def frame_assignment(pred: Tracks, gt: Tracks, iou_thr: float = 0.5) -> dict[tuple[int, int], int]:
    """Map `(frame, pred_id)` -> `gt_id`: per frame, Hungarian on 1 - IoU, keeping IoU >= thr."""
    out: dict[tuple[int, int], int] = {}
    if len(pred) == 0 or len(gt) == 0:
        return out
    for f in np.intersect1d(pred.frame, gt.frame).tolist():
        pi, gi = np.flatnonzero(pred.frame == f), np.flatnonzero(gt.frame == f)
        iou = iou_xywh(pred.xywh[pi], gt.xywh[gi])
        rows, cols = linear_sum_assignment(1.0 - iou)
        for r, c in zip(rows, cols, strict=True):
            if iou[r, c] >= iou_thr:
                out[(int(f), int(pred.track_id[pi[r]]))] = int(gt.track_id[gi[c]])
    return out


@dataclass(frozen=True)
class Attribution:
    """Every unmatched crossing with its cause.

    `fp_events` / `fn_events` hold `(crossing, category)`; `fp` / `fn` count them per category.
    """

    match: EventMatch
    fp: dict[str, int]
    fn: dict[str, int]
    fp_events: list[tuple[Crossing, str]] = field(default_factory=list)
    fn_events: list[tuple[Crossing, str]] = field(default_factory=list)

    @property
    def switch_attributed(self) -> int:
        """Errors an identity switch explains: double counts plus fragmented identities."""
        return self.fp["duplicate"] + self.fn["fragmented"]


def attribute_errors(
    pred_tracks: Tracks,
    gt_tracks: Tracks,
    line: Line,
    rule: Rule,
    tol_frames: int,
    iou_thr: float = 0.5,
    anchor: str = "bottom_center",
    gt_rule: Rule | None = None,
) -> Attribution:
    """Count, match and classify the crossing errors of one line.

    Predictions are counted with `rule`; ground truth always with `gt_rule` (naive when None:
    every annotated crossing is real), so a band that suppresses a genuine back-and-forth shows
    up as a false negative.

    Unmatched predicted crossing (track p, frame f, direction d), with g the ground-truth id most
    often assigned to p within `tol_frames` of f (ties: smallest id):
      - `unmatched`: no such g, nothing in the ground truth is behind the track;
      - `duplicate`: g has a ground-truth crossing in direction d within the window, so the
        person was already counted by another id (the double count a switch produces);
      - `spurious`: otherwise, the person did not cross that way (jitter or a wobble).
    Unmatched ground-truth crossing (id g, frame f), with B the predicted ids assigned to g in
    [f - tol, f - 1] and A those in [f, f + tol]:
      - `tracked_not_counted`: B and A share an id, one track followed the person across;
      - `fragmented`: B and A are both non-empty but disjoint, the identity changed at the line;
      - `missed`: no track on one side, a detection gap.
    """
    pred_count = count_line(pred_tracks, line, rule, anchor)
    gt_count = count_line(gt_tracks, line, gt_rule or Rule.naive(), anchor)
    match = match_events(pred_count.events, gt_count.events, tol_frames)
    assign = frame_assignment(pred_tracks, gt_tracks, iou_thr)

    by_pred: dict[int, list[tuple[int, int]]] = {}
    by_gt: dict[int, list[tuple[int, int]]] = {}
    for (f, pid), gid in assign.items():
        by_pred.setdefault(pid, []).append((f, gid))
        by_gt.setdefault(gid, []).append((f, pid))
    gt_events: dict[int, list[Crossing]] = {}
    for e in gt_count.events:
        gt_events.setdefault(e.track_id, []).append(e)

    matched_pred = {_key(p) for p, _ in match.pairs}
    matched_gt = {_key(g) for _, g in match.pairs}

    fp_events: list[tuple[Crossing, str]] = []
    for e in pred_count.events:
        if _key(e) in matched_pred:
            continue
        near = Counter(g for f, g in by_pred.get(e.track_id, []) if abs(f - e.frame) <= tol_frames)
        if not near:
            fp_events.append((e, "unmatched"))
            continue
        g = min(near, key=lambda k: (-near[k], k))
        same_way = any(
            c.direction == e.direction and abs(c.frame - e.frame) <= tol_frames
            for c in gt_events.get(g, [])
        )
        fp_events.append((e, "duplicate" if same_way else "spurious"))

    fn_events: list[tuple[Crossing, str]] = []
    for e in gt_count.events:
        if _key(e) in matched_gt:
            continue
        obs = by_gt.get(e.track_id, [])
        before = {p for f, p in obs if e.frame - tol_frames <= f <= e.frame - 1}
        after = {p for f, p in obs if e.frame <= f <= e.frame + tol_frames}
        if before & after:
            cat = "tracked_not_counted"
        elif before and after:
            cat = "fragmented"
        else:
            cat = "missed"
        fn_events.append((e, cat))

    return Attribution(
        match=match,
        fp={c: sum(k == c for _, k in fp_events) for c in FP_CATEGORIES},
        fn={c: sum(k == c for _, k in fn_events) for c in FN_CATEGORIES},
        fp_events=fp_events,
        fn_events=fn_events,
    )


def _key(e: Crossing) -> tuple[int, int, int]:
    return (e.track_id, e.frame, e.direction)
