"""MOT metrics (MOTA, IDF1, ID switches, ...) through `motmetrics`."""

import motmetrics as mm
import numpy as np

from vidan.attribution import iou_xywh
from vidan.mot import Tracks

_METRICS = (
    "mota",
    "motp",
    "idf1",
    "idp",
    "idr",
    "num_switches",
    "num_fragmentations",
    "mostly_tracked",
    "mostly_lost",
    "precision",
    "recall",
    "num_frames",
    "num_objects",
    "num_predictions",
)
_INT_METRICS = frozenset(
    {
        "num_switches",
        "num_fragmentations",
        "mostly_tracked",
        "mostly_lost",
        "num_frames",
        "num_objects",
        "num_predictions",
    }
)


def _by_frame(t: Tracks) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for f in np.unique(t.frame).tolist():
        out[f] = np.flatnonzero(t.frame == f)
    return out


def _iou_distance(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """1 - IoU with nan where IoU < iou_thr: what `mm.distances.iou_matrix(max_iou=1-thr)` gives.

    motmetrics 1.4.0's own `iou_matrix` calls `np.asfarray`, removed in NumPy 2, so it raises here.
    """
    dist = 1.0 - iou_xywh(gt_boxes, pred_boxes)
    dist[dist > 1.0 - iou_thr] = np.nan
    return dist


def evaluate(gt: Tracks, pred: Tracks, iou_thr: float = 0.5) -> dict[str, float]:
    """Score `pred` against `gt`; a pair matches when IoU >= `iou_thr`.

    Every frame that appears in either set is fed to the accumulator, so frames with ground
    truth but no predictions count as misses and frames with predictions only count as false
    positives. `motp` is motmetrics' mean distance over matches, i.e. mean (1 - IoU): lower is
    better. Predictions matched to non-pedestrian or distractor GT are not removed the way the
    MOTChallenge devkit does, so MOTA and IDF1 here are not comparable to leaderboard numbers.
    Ratios are nan when their denominator is zero (for example no ground truth at all).
    """
    acc = mm.MOTAccumulator(auto_id=False)
    g_rows, p_rows = _by_frame(gt), _by_frame(pred)
    empty = np.empty(0, dtype=np.int64)
    for f in sorted(g_rows.keys() | p_rows.keys()):
        gi, pi = g_rows.get(f, empty), p_rows.get(f, empty)
        if len(gi) and len(pi):
            dist = _iou_distance(gt.xywh[gi], pred.xywh[pi], iou_thr)
        else:
            dist = np.empty((len(gi), len(pi)))
        acc.update(gt.track_id[gi].tolist(), pred.track_id[pi].tolist(), dist, frameid=int(f))
    if not acc.mot_events.shape[0]:
        return {m: (0 if m in _INT_METRICS else float("nan")) for m in _METRICS}
    row = mm.metrics.create().compute(acc, metrics=list(_METRICS), name="seq").iloc[0]
    return {m: (int(row[m]) if m in _INT_METRICS else float(row[m])) for m in _METRICS}
