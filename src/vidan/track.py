"""ByteTrack over cached detections -> `Tracks`, with optional frame skipping.

supervision 0.31 no longer ships ByteTrack; it lives in the `trackers` package (Apache-2.0),
which takes and returns `supervision.Detections`. Parameter names below are that class's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vidan.detect import Detections
from vidan.mot import Tracks, tracks_from_rows

# `trackers==2.6.1` defaults for `ByteTrackTracker`, spelled out so a run records what it used.
BYTETRACK_DEFAULTS: dict[str, Any] = {
    "lost_track_buffer": 30,  # in 30 fps frames; scaled by frame_rate / 30
    "track_activation_threshold": 0.7,  # only detections at or above this can start a track
    "minimum_consecutive_frames": 2,
    "minimum_iou_threshold": 0.1,
    "high_conf_det_threshold": 0.6,  # first-stage / second-stage split
}


def select_frames(frames: list[int] | np.ndarray, frame_skip: int = 1) -> list[int]:
    """Frames that are fed to the tracker: those with `(frame - 1) % frame_skip == 0`."""
    if frame_skip < 1:
        raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
    return [int(f) for f in sorted(frames) if (int(f) - 1) % frame_skip == 0]


def run_bytetrack(
    per_frame: dict[int, Detections],
    frame_rate: float,
    frame_skip: int = 1,
    **bytetrack_params: Any,
) -> Tracks:
    """Track detections with ByteTrack. Deterministic: no randomness, ordered assignment.

    Only frames with `(frame - 1) % frame_skip == 0` are fed, in frame order; with skipping the
    tracker's `frame_rate` becomes `frame_rate / frame_skip` so its lost-track buffer still spans
    the same wall-clock time. Output rows keep the original frame numbers; track ids are
    1-based (the library counts from 0). Detections without a confirmed track (`tracker_id ==
    -1`) are dropped. In `trackers` 2.6 that includes the first observation of every track,
    whatever `minimum_consecutive_frames` is: the id is assigned on the first update.

    `bytetrack_params` override `BYTETRACK_DEFAULTS` (names of `trackers.ByteTrackTracker`).
    """
    import supervision as sv
    from trackers import ByteTrackTracker

    fed = select_frames(list(per_frame), frame_skip)
    params = {**BYTETRACK_DEFAULTS, **bytetrack_params}
    tracker = ByteTrackTracker(frame_rate=frame_rate / frame_skip, **params)
    rows: list[tuple[float, ...]] = []
    for frame in fed:
        det = per_frame[frame]
        sv_det = sv.Detections(
            xyxy=det.xyxy.astype(np.float64),
            confidence=det.score.astype(np.float64),
            class_id=det.class_id.astype(int),
        )
        out = tracker.update(sv_det)
        if len(out) == 0 or out.tracker_id is None:
            continue
        for box, score, tid in zip(out.xyxy, out.confidence, out.tracker_id, strict=True):
            if tid < 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            rows.append((frame, int(tid) + 1, x1, y1, x2 - x1, y2 - y1, float(score)))
    return tracks_from_rows(rows)
