"""Object detection: `Detections`, the `Detector` protocol, RF-DETR backends and the cache.

The pure numpy helpers (decoding, box maths, class filter, cache, pre-processing) import
neither torch nor onnxruntime, so they run in CI without the `vision` extra. The two
backends import their heavy dependencies lazily, inside the class.

Class ids are the COCO category ids RF-DETR reports (sparse, 1..90): person is 1.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

# COCO category ids as RF-DETR reports them (`rfdetr.assets.coco_classes.COCO_CLASSES`).
PERSON = 1
CAR = 3
MOTORCYCLE = 4
BUS = 6
TRUCK = 8
VEHICLES = (CAR, MOTORCYCLE, BUS, TRUCK)

# RF-DETR input normalisation (`RFDETR.means` / `RFDETR.stds`): ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default square input resolution per size (the variant's `ModelConfig.resolution`).
DEFAULT_RESOLUTION = {"nano": 384, "small": 512, "medium": 576}
# `ModelConfig.patch_size * num_windows`: resolution must be a multiple of this.
RESOLUTION_STEP = {"nano": 32, "small": 32, "medium": 32}
# `ModelConfig.num_select`: query/class pairs ranked before the threshold is applied.
NUM_SELECT = 300

CACHE_FORMAT = 1


@dataclass(frozen=True)
class Detections:
    """Detections of one frame: pixel `xyxy` boxes in the original frame, score, COCO class id."""

    xyxy: np.ndarray
    score: np.ndarray
    class_id: np.ndarray

    def __post_init__(self) -> None:
        xyxy = np.asarray(self.xyxy, dtype=np.float32)
        if xyxy.size == 0:
            xyxy = xyxy.reshape(0, 4)
        score = np.asarray(self.score, dtype=np.float32).reshape(-1)
        class_id = np.asarray(self.class_id, dtype=np.int64).reshape(-1)
        if xyxy.ndim != 2 or xyxy.shape[1] != 4:
            raise ValueError(f"xyxy must have shape (n, 4), got {xyxy.shape}")
        if not (len(xyxy) == len(score) == len(class_id)):
            raise ValueError("xyxy, score and class_id must have equal length")
        object.__setattr__(self, "xyxy", xyxy)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "class_id", class_id)

    def __len__(self) -> int:
        return len(self.score)

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64))

    def keep(self, mask: np.ndarray) -> Detections:
        """Rows selected by a boolean mask or an index array."""
        return Detections(self.xyxy[mask], self.score[mask], self.class_id[mask])


@runtime_checkable
class Detector(Protocol):
    """Anything that turns one BGR frame into `Detections`."""

    name: str

    def detect(self, frame_bgr: np.ndarray) -> Detections: ...


# --------------------------------------------------------------------------- numpy helpers


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function (logits clipped to +-88, like RF-DETR's decoder)."""
    x = np.clip(x, -88.0, 88.0)
    one = x.dtype.type(1)
    return one / (one + np.exp(-x))


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(n, 4) centre-x, centre-y, width, height -> top-left / bottom-right corners."""
    cx, cy, w, h = np.asarray(boxes).T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def rescale_boxes(xyxy_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale normalised xyxy to pixels of a `width` x `height` frame and clip to the frame."""
    scale = np.array([width, height, width, height], dtype=xyxy_norm.dtype)
    return np.clip(xyxy_norm * scale, 0.0, scale)


def topk_pairs(
    scores: np.ndarray, threshold: float, num_select: int = NUM_SELECT
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top `num_select` (query, class) pairs of a (Q, C) score grid, then `score > threshold`.

    RF-DETR scores every class with an independent sigmoid, so one query can yield several
    detections. Pairs are ranked by descending score, ties by ascending flat index, which is
    what `PostProcess._select_topk` does. Thresholding first and ranking the survivors gives
    the same set and order as ranking first, and sorts far fewer elements.

    Returns `(score, class_index, query_index)`, each 1-D, sorted by descending score.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape (Q, C), got {scores.shape}")
    flat = scores.reshape(-1)
    idx = np.flatnonzero(flat > threshold)
    order = np.lexsort((idx, -flat[idx]))[:num_select]
    idx = idx[order]
    n_classes = scores.shape[1]
    return flat[idx], idx % n_classes, idx // n_classes


def decode_outputs(
    boxes_cxcywh: np.ndarray,
    logits: np.ndarray,
    image_size: tuple[int, int],
    threshold: float,
    num_select: int = NUM_SELECT,
) -> Detections:
    """One image's raw RF-DETR outputs -> `Detections` in pixels of `image_size` = (width, height).

    `boxes_cxcywh` is (Q, 4) normalised, `logits` is (Q, C). Every class slot takes part in the
    ranking (the pretrained COCO head has no background slot to drop: slot index = COCO id).
    """
    score, cls, query = topk_pairs(sigmoid(logits), threshold, num_select)
    xyxy = rescale_boxes(cxcywh_to_xyxy(boxes_cxcywh[query]), *image_size)
    return Detections(xyxy, score, cls)


def filter_classes(det: Detections, classes: Iterable[int] | None) -> Detections:
    """Keep only the given COCO ids; `None` keeps every class."""
    if classes is None:
        return det
    return det.keep(np.isin(det.class_id, np.fromiter(classes, dtype=np.int64)))


def resize_bilinear(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of an (H, W, C) float image: half-pixel centres, no antialiasing.

    This is the convention of `torchvision.transforms.functional.resize(antialias=False)`
    that `RFDETR.predict` uses, and of `cv2.INTER_LINEAR`.
    """
    src_h, src_w = img.shape[:2]

    def axis(n_src: int, n_dst: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos = (np.arange(n_dst, dtype=np.float32) + 0.5) * (n_src / n_dst) - 0.5
        pos = np.clip(pos, 0.0, n_src - 1)
        lo = np.floor(pos).astype(np.intp)
        hi = np.minimum(lo + 1, n_src - 1)
        return lo, hi, (pos - lo).astype(np.float32)

    y0, y1, wy = axis(src_h, height)
    x0, x1, wx = axis(src_w, width)
    top = img[y0][:, x0] * (1 - wx)[None, :, None] + img[y0][:, x1] * wx[None, :, None]
    bot = img[y1][:, x0] * (1 - wx)[None, :, None] + img[y1][:, x1] * wx[None, :, None]
    return (top * (1 - wy)[:, None, None] + bot * wy[:, None, None]).astype(np.float32)


def preprocess_bgr(frame_bgr: np.ndarray, height: int, width: int | None = None) -> np.ndarray:
    """BGR uint8 frame -> (1, 3, H, W) float32 model input.

    RGB, scaled to [0, 1], resized (bilinear, half-pixel, no antialiasing) *as float*, then
    `(x - mean) / std` with the ImageNet statistics: the same order as `RFDETR.predict`.
    Uses OpenCV for the resize when it is installed, a numpy version otherwise.
    """
    width = height if width is None else width
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) * np.float32(1.0 / 255.0)
    try:
        import cv2
    except ImportError:
        small = resize_bilinear(rgb, height, width)
    else:
        small = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    chw = ((small - mean) / std).transpose(2, 0, 1)
    return np.ascontiguousarray(chw[None], dtype=np.float32)


# --------------------------------------------------------------------------- backends


def _check_resolution(size: str, resolution: int) -> None:
    step = RESOLUTION_STEP[size]
    if resolution <= 0 or resolution % step:
        raise ValueError(f"resolution must be a positive multiple of {step}, got {resolution}")


class TorchRFDETR:
    """RF-DETR through `RFDETR.predict` on the CPU (PyTorch).

    `resolution=None` keeps the variant's default (nano 384, small 512, medium 576). Weights
    download on first use to `~/.roboflow/models` (override with `RF_HOME`).
    """

    def __init__(
        self,
        size: str = "nano",
        resolution: int | None = None,
        threshold: float = 0.3,
        classes: Iterable[int] | None = (PERSON,),
        num_threads: int | None = None,
    ) -> None:
        if size not in DEFAULT_RESOLUTION:
            raise ValueError(f"size must be one of {sorted(DEFAULT_RESOLUTION)}, got {size!r}")
        import rfdetr
        import torch

        if num_threads is not None:
            torch.set_num_threads(num_threads)
        self.size = size
        self.resolution = DEFAULT_RESOLUTION[size] if resolution is None else resolution
        _check_resolution(size, self.resolution)
        self.threshold = threshold
        self.classes = None if classes is None else tuple(classes)
        self.name = f"rfdetr-{size}-torch-{self.resolution}"
        variant = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}[size]
        kwargs: dict[str, Any] = {"device": "cpu"}
        if resolution is not None and resolution != DEFAULT_RESOLUTION[size]:
            kwargs["resolution"] = resolution
        self._model = getattr(rfdetr, variant)(**kwargs)

    def detect(self, frame_bgr: np.ndarray) -> Detections:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        out = self._model.predict(rgb, threshold=self.threshold, include_source_image=False)
        if out.xyxy is None or len(out) == 0:
            return Detections.empty()
        det = Detections(out.xyxy, out.confidence, out.class_id)
        return filter_classes(det, self.classes)

    def meta(self) -> dict[str, Any]:
        import torch

        return {
            "detector": self.name,
            "size": self.size,
            "resolution": self.resolution,
            "backend": "torch",
            "threshold": self.threshold,
            "classes": None if self.classes is None else list(self.classes),
            "threads": torch.get_num_threads(),
        }


class OnnxRFDETR:
    """RF-DETR exported to ONNX, run on onnxruntime's CPU provider.

    Pre- and post-processing are numpy re-implementations of `RFDETR.predict` (see
    `preprocess_bgr` and `decode_outputs`). The model's input size is read from the graph, so
    `resolution` only has to agree with the export (it is checked).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        resolution: int | None = None,
        threshold: float = 0.3,
        classes: Iterable[int] | None = (PERSON,),
        intra_op_threads: int = 0,
        label: str | None = None,
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads  # 0 = onnxruntime's own default
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._sess = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self._sess.get_inputs()[0]
        self._input = inp.name
        _, _, height, width = inp.shape
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f"expected a static input shape, got {inp.shape}")
        if resolution is not None and (height, width) != (resolution, resolution):
            raise ValueError(f"model input is {height}x{width}, resolution={resolution} given")
        self.input_hw = (height, width)
        self.resolution = height
        outs = [o.name for o in self._sess.get_outputs()]
        self._box_idx, self._logit_idx = outs.index("dets"), outs.index("labels")
        self.threshold = threshold
        self.classes = None if classes is None else tuple(classes)
        self.intra_op_threads = intra_op_threads
        self.onnx_path = Path(onnx_path)
        self.name = label or f"rfdetr-onnx-{self.onnx_path.stem}-{self.resolution}"

    def detect(self, frame_bgr: np.ndarray) -> Detections:
        h, w = frame_bgr.shape[:2]
        x = preprocess_bgr(frame_bgr, *self.input_hw)
        out = self._sess.run(None, {self._input: x})
        det = decode_outputs(out[self._box_idx][0], out[self._logit_idx][0], (w, h), self.threshold)
        return filter_classes(det, self.classes)

    def meta(self) -> dict[str, Any]:
        return {
            "detector": self.name,
            "resolution": self.resolution,
            "backend": "onnxruntime",
            "model_file": self.onnx_path.name,
            "threshold": self.threshold,
            "classes": None if self.classes is None else list(self.classes),
            "threads": self.intra_op_threads,
        }


# --------------------------------------------------------------------------- cache


def library_versions() -> dict[str, str | None]:
    """Installed versions of the packages that decide detector output (None if absent)."""
    from importlib import metadata

    out: dict[str, str | None] = {}
    for pkg in ("rfdetr", "onnxruntime", "torch"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def save_cache(
    path: str | Path, per_frame: Mapping[int, Detections], meta: Mapping[str, Any]
) -> None:
    """Write per-frame detections plus a JSON meta block to a compressed `.npz`.

    Meta is free-form (detector name, size, resolution, backend, threshold, sequence, ...);
    library versions and the frame count are added if the caller left them out.
    """
    frames = np.array(sorted(per_frame), dtype=np.int64)
    counts = np.array([len(per_frame[int(f)]) for f in frames], dtype=np.int64)
    parts = [per_frame[int(f)] for f in frames]
    full = {"cache_format": CACHE_FORMAT, "n_frames": len(frames), "versions": library_versions()}
    full.update(meta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frames=frames,
        counts=counts,
        xyxy=np.concatenate([d.xyxy for d in parts]) if parts else np.empty((0, 4), np.float32),
        score=np.concatenate([d.score for d in parts]) if parts else np.empty(0, np.float32),
        class_id=np.concatenate([d.class_id for d in parts]) if parts else np.empty(0, np.int64),
        meta=np.array(json.dumps(full, sort_keys=True)),
    )


def load_cache(path: str | Path) -> tuple[dict[int, Detections], dict[str, Any]]:
    """Read a cache written by `save_cache`: `(per_frame, meta)`."""
    with np.load(Path(path), allow_pickle=False) as z:
        frames, counts = z["frames"], z["counts"]
        xyxy, score, class_id = z["xyxy"], z["score"], z["class_id"]
        meta = json.loads(str(z["meta"]))
    ends = np.cumsum(counts)
    starts = ends - counts
    per_frame = {
        int(f): Detections(xyxy[s:e], score[s:e], class_id[s:e])
        for f, s, e in zip(frames, starts, ends, strict=True)
    }
    return per_frame, meta


def detect_sequence(
    detector: Detector, frames: Iterable[tuple[int, np.ndarray]]
) -> tuple[dict[int, Detections], list[float]]:
    """Run a detector over `(frame_number, bgr)` pairs; returns detections and per-frame seconds."""
    per_frame: dict[int, Detections] = {}
    times: list[float] = []
    for number, bgr in frames:
        t0 = time.perf_counter()
        per_frame[int(number)] = detector.detect(bgr)
        times.append(time.perf_counter() - t0)
    return per_frame, times
