"""MOTChallenge text I/O and the `Tracks` value type (numpy only)."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ANCHORS = ("bottom_center", "center")


@dataclass(frozen=True)
class Tracks:
    """Column arrays of equal length: one entry per (frame, track) observation."""

    frame: np.ndarray  # int64, 1-based
    track_id: np.ndarray  # int64
    xywh: np.ndarray  # float64 (n, 4): top-left x, y, width, height
    score: np.ndarray  # float64, 1.0 for ground truth

    def __post_init__(self) -> None:
        frame = np.asarray(self.frame, dtype=np.int64).reshape(-1)
        track_id = np.asarray(self.track_id, dtype=np.int64).reshape(-1)
        score = np.asarray(self.score, dtype=np.float64).reshape(-1)
        xywh = np.asarray(self.xywh, dtype=np.float64)
        if xywh.size == 0:
            xywh = xywh.reshape(0, 4)
        if xywh.ndim != 2 or xywh.shape[1] != 4:
            raise ValueError(f"xywh must have shape (n, 4), got {xywh.shape}")
        n = len(frame)
        if not (len(track_id) == len(score) == len(xywh) == n):
            raise ValueError(
                "Tracks columns must have equal length: "
                f"frame={n}, track_id={len(track_id)}, xywh={len(xywh)}, score={len(score)}"
            )
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "xywh", xywh)
        object.__setattr__(self, "score", score)

    def __len__(self) -> int:
        return len(self.frame)

    @classmethod
    def empty(cls) -> "Tracks":
        return cls(np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 4)), np.empty(0))

    def anchor(self, kind: str = "bottom_center") -> np.ndarray:
        """(n, 2) anchor points: `bottom_center` (feet) or `center`."""
        x, y, w, h = self.xywh.T
        if kind == "bottom_center":
            return np.column_stack([x + w / 2.0, y + h])
        if kind == "center":
            return np.column_stack([x + w / 2.0, y + h / 2.0])
        raise ValueError(f"unknown anchor {kind!r}; expected one of {_ANCHORS}")

    def by_id(self) -> dict[int, np.ndarray]:
        """Map track id -> row indices of that id, in frame order."""
        if len(self) == 0:
            return {}
        order = np.lexsort((self.frame, self.track_id))
        ids = self.track_id[order]
        cuts = np.flatnonzero(np.diff(ids)) + 1
        starts = np.concatenate([[0], cuts])
        ends = np.concatenate([cuts, [len(ids)]])
        return {int(ids[s]): order[s:e] for s, e in zip(starts, ends, strict=True)}


def tracks_from_rows(rows: Iterable[Sequence[float]]) -> Tracks:
    """Build `Tracks` from rows `(frame, id, x, y, w, h[, score])`, sorted by (frame, id).

    A missing score defaults to 1.0. No rows gives an empty `Tracks`.
    """
    data = [list(r) for r in rows]
    if not data:
        return Tracks.empty()
    if any(len(r) not in (6, 7) for r in data):
        raise ValueError(
            "each row must be (frame, id, x, y, w, h) or (frame, id, x, y, w, h, score)"
        )
    arr = np.array([r if len(r) == 7 else [*r, 1.0] for r in data], dtype=np.float64)
    return _from_array(arr[:, :6], arr[:, 6])


def _from_array(core: np.ndarray, score: np.ndarray) -> Tracks:
    frame = core[:, 0].astype(np.int64)
    track_id = core[:, 1].astype(np.int64)
    order = np.lexsort((track_id, frame))  # primary key frame, then id; stable
    return Tracks(frame[order], track_id[order], core[order, 2:6], np.asarray(score)[order])


def read_mot(
    path: str | Path,
    *,
    gt: bool = False,
    classes: Sequence[int] = (1,),
    min_visibility: float = 0.0,
) -> Tracks:
    """Read a MOTChallenge text file.

    Prediction files have 10 columns (`frame,id,x,y,w,h,conf,-1,-1,-1`); MOT17 ground truth
    has 9 (`frame,id,x,y,w,h,consider,class,visibility`). With `gt=True` only rows with
    `consider == 1`, `class in classes` and `visibility >= min_visibility` are kept and the
    score is 1.0. With `gt=False` every row is kept and the score is column 7 (1.0 if absent);
    `min_visibility` is then meaningless and must stay 0.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return Tracks.empty()
    arr = np.loadtxt(text.splitlines(), delimiter=",", ndmin=2, dtype=np.float64)
    ncols = arr.shape[1]
    if ncols < 6:
        raise ValueError(f"{path}: expected at least 6 columns, got {ncols}")
    if gt:
        if ncols < 9:
            raise ValueError(f"{path}: ground truth needs 9 columns, got {ncols}")
        keep = (
            (arr[:, 6] == 1)
            & np.isin(arr[:, 7], np.asarray(classes, dtype=np.float64))
            & (arr[:, 8] >= min_visibility)
        )
        arr = arr[keep]
        return _from_array(arr[:, :6], np.ones(len(arr)))
    if min_visibility > 0:
        raise ValueError("min_visibility only applies with gt=True")
    score = arr[:, 6] if ncols >= 7 else np.ones(len(arr))
    return _from_array(arr[:, :6], score)


def write_mot(path: str | Path, tracks: Tracks) -> None:
    """Write predictions as `frame,id,x,y,w,h,conf,-1,-1,-1` (round-trips through `read_mot`)."""
    lines = []
    for f, i, (x, y, w, h), s in zip(
        tracks.frame.tolist(),
        tracks.track_id.tolist(),
        tracks.xywh.tolist(),
        tracks.score.tolist(),
        strict=True,
    ):
        lines.append(f"{f},{i},{x!r},{y!r},{w!r},{h!r},{s!r},-1,-1,-1")
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
