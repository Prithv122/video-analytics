"""Attribute counting errors to their cause (numpy + scipy only).

Predicted and ground-truth crossings are matched (by identity, or by time only); every
unmatched event is then explained using the per-frame box assignment between predicted ids
and ground-truth ids.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from vidan.counting import Crossing, LineCount, Rule, count_line
from vidan.geometry import Line
from vidan.metrics import EventMatch, event_match_from_pairs, match_events
from vidan.mot import Tracks

FP_CATEGORIES = ("duplicate", "spurious", "unmatched")
FN_CATEGORIES = ("tracked_not_counted", "fragmented", "missed")


def iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (n, 4) and (m, 4) top-left xywh boxes -> (n, m)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ax, ay, aw, ah = a.T
    bx, by, bw, bh = b.T
    iw = np.minimum((ax + aw)[:, None], bx + bw) - np.maximum(ax[:, None], bx)
    ih = np.minimum((ay + ah)[:, None], by + bh) - np.maximum(ay[:, None], by)
    inter = np.maximum(iw, 0.0) * np.maximum(ih, 0.0)
    union = (aw * ah)[:, None] + bw * bh - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _frame_groups(t: Tracks) -> dict[int, np.ndarray]:
    """Row indices of each frame, from one stable sort."""
    order = np.argsort(t.frame, kind="stable")
    frames, starts = np.unique(t.frame[order], return_index=True)
    ends = np.append(starts[1:], len(order))
    return {int(f): order[s:e] for f, s, e in zip(frames, starts, ends, strict=True)}


def frame_assignment(pred: Tracks, gt: Tracks, iou_thr: float = 0.5) -> dict[tuple[int, int], int]:
    """Map `(frame, pred_id)` -> `gt_id`: per frame, Hungarian on 1 - IoU, keeping IoU >= thr."""
    out: dict[tuple[int, int], int] = {}
    if len(pred) == 0 or len(gt) == 0:
        return out
    p_groups, g_groups = _frame_groups(pred), _frame_groups(gt)
    p_ids, g_ids = pred.track_id, gt.track_id
    for f in sorted(p_groups.keys() & g_groups.keys()):
        pi, gi = p_groups[f], g_groups[f]
        iou = iou_xywh(pred.xywh[pi], gt.xywh[gi])
        rows, cols = linear_sum_assignment(1.0 - iou)
        ok = iou[rows, cols] >= iou_thr
        out.update(
            zip(
                zip([f] * int(ok.sum()), p_ids[pi[rows[ok]]].tolist(), strict=True),
                g_ids[gi[cols[ok]]].tolist(),
                strict=True,
            )
        )
    return out


def _index_assignment(
    assign: dict[tuple[int, int], int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Per pred id -> [(frame, gt_id)] and per gt id -> [(frame, pred_id)]."""
    by_pred: dict[int, list[tuple[int, int]]] = {}
    by_gt: dict[int, list[tuple[int, int]]] = {}
    for (f, pid), gid in assign.items():
        by_pred.setdefault(pid, []).append((f, gid))
        by_gt.setdefault(gid, []).append((f, pid))
    return by_pred, by_gt


def _identity(obs: list[tuple[int, int]], frame: int, tol_frames: int) -> int | None:
    """GT id most often assigned within `tol_frames` of `frame` (ties: smallest id), or None."""
    near = Counter(g for f, g in obs if abs(f - frame) <= tol_frames)
    if not near:
        return None
    return min(near, key=lambda k: (-near[k], k))


def match_events_identity(
    pred_tracks: Tracks,
    gt_tracks: Tracks,
    pred_events: Sequence[Crossing],
    gt_events: Sequence[Crossing],
    tol_frames: int,
    iou_thr: float = 0.5,
    assignment: dict[tuple[int, int], int] | None = None,
) -> EventMatch:
    """Like `match_events`, but a pair must also be the same person.

    A predicted crossing (p, f, d) may match a GT crossing (g, f', d') only if d == d',
    |f - f'| <= tol_frames and the GT identity behind track p around f (majority vote of the
    per-frame box assignment over [f - tol, f + tol], ties to the smallest id) is g. Among the
    allowed pairs the match is one-to-one and minimises total |f - f'|. Pass `assignment` (from
    `frame_assignment`) to skip recomputing it.
    """
    if tol_frames < 0:
        raise ValueError("tol_frames must be >= 0")
    if assignment is None:
        assignment = frame_assignment(pred_tracks, gt_tracks, iou_thr)
    by_pred, _ = _index_assignment(assignment)
    who = [_identity(by_pred.get(e.track_id, []), e.frame, tol_frames) for e in pred_events]
    pairs: list[tuple[Crossing, Crossing]] = []
    for direction in (1, -1):
        pi = [i for i, e in enumerate(pred_events) if e.direction == direction]
        gj = [j for j, e in enumerate(gt_events) if e.direction == direction]
        if not pi or not gj:
            continue
        pf = np.array([pred_events[i].frame for i in pi])
        gf = np.array([gt_events[j].frame for j in gj])
        diff = np.abs(pf[:, None] - gf[None, :])
        same = np.array([[who[i] == gt_events[j].track_id for j in gj] for i in pi])
        allowed = same & (diff <= tol_frames)
        big = min(len(pi), len(gj)) * tol_frames + 1
        rows, cols = linear_sum_assignment(np.where(allowed, diff, big))
        pairs.extend(
            (pred_events[pi[r]], gt_events[gj[c]])
            for r, c in zip(rows, cols, strict=True)
            if allowed[r, c]
        )
    return event_match_from_pairs(pairs, len(pred_events), len(gt_events))


@dataclass(frozen=True)
class Attribution:
    """Every unmatched crossing with its cause.

    `fp_events` / `fn_events` hold `(crossing, category)`; `fp` / `fn` count them per category.
    `assignment` is the `(frame, pred_id) -> gt_id` map the analysis used, for reuse;
    `pred_events` / `gt_events` are the counted crossings that were matched.
    """

    match: EventMatch
    fp: dict[str, int]
    fn: dict[str, int]
    fp_events: list[tuple[Crossing, str]] = field(default_factory=list)
    fn_events: list[tuple[Crossing, str]] = field(default_factory=list)
    assignment: dict[tuple[int, int], int] = field(default_factory=dict)
    pred_events: tuple[Crossing, ...] = ()
    gt_events: tuple[Crossing, ...] = ()

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
    matching: str = "identity",
    assignment: dict[tuple[int, int], int] | None = None,
    gt_count: LineCount | None = None,
) -> Attribution:
    """Count, match and classify the crossing errors of one line.

    Predictions are counted with `rule`; ground truth always with `gt_rule` (naive when None:
    every annotated crossing is real), so a band that suppresses a genuine back-and-forth shows
    up as a false negative.

    `matching="identity"` (default) uses `match_events_identity`: crossings only match when they
    are the same person, so a miss of one person cannot cancel against a double count of
    another. `matching="time"` uses `match_events` (frame proximity and direction only).

    `assignment` (from `frame_assignment`) and `gt_count` (the ground truth's `LineCount` for this
    line, counted with `gt_rule`) may be passed in when a caller scores many predictions against
    the same truth; they are computed here otherwise.

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
    if gt_count is None:
        gt_count = count_line(gt_tracks, line, gt_rule or Rule.naive(), anchor)
    assign = (
        assignment if assignment is not None else frame_assignment(pred_tracks, gt_tracks, iou_thr)
    )
    if matching == "identity":
        match = match_events_identity(
            pred_tracks,
            gt_tracks,
            pred_count.events,
            gt_count.events,
            tol_frames,
            assignment=assign,
        )
    elif matching == "time":
        match = match_events(pred_count.events, gt_count.events, tol_frames)
    else:
        raise ValueError(f"matching must be 'identity' or 'time', got {matching!r}")
    by_pred, by_gt = _index_assignment(assign)
    gt_events: dict[int, list[Crossing]] = {}
    for e in gt_count.events:
        gt_events.setdefault(e.track_id, []).append(e)

    matched_pred = {_key(p) for p, _ in match.pairs}
    matched_gt = {_key(g) for _, g in match.pairs}

    fp_events: list[tuple[Crossing, str]] = []
    for e in pred_count.events:
        if _key(e) in matched_pred:
            continue
        g = _identity(by_pred.get(e.track_id, []), e.frame, tol_frames)
        if g is None:
            fp_events.append((e, "unmatched"))
            continue
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
        assignment=assign,
        pred_events=pred_count.events,
        gt_events=gt_count.events,
    )


def _key(e: Crossing) -> tuple[int, int, int]:
    return (e.track_id, e.frame, e.direction)
