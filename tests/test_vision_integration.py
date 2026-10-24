"""Integration tests: need rfdetr weights, an exported ONNX graph and the `vision` extra."""

import urllib.request
from pathlib import Path

import numpy as np
import pytest

from vidan.detect import Detections, OnnxRFDETR, TorchRFDETR
from vidan.track import run_bytetrack

pytestmark = pytest.mark.integration

# The sample photo rfdetr's own docs use; cached under data/cache/ (git-ignored).
SAMPLE_URL = "https://media.roboflow.com/dog.jpg"
SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "cache" / "dog.jpg"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _matched_ious(a: Detections, b: Detections) -> list[float]:
    """Best same-class IoU of every box in `a` against the unused boxes of `b`."""
    used: set[int] = set()
    out = []
    for i in range(len(a)):
        best, best_j = 0.0, -1
        for j in range(len(b)):
            if j in used or a.class_id[i] != b.class_id[j]:
                continue
            v = _iou(a.xyxy[i], b.xyxy[j])
            if v > best:
                best, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
        out.append(best)
    return out


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
        except OSError as exc:
            pytest.skip(f"sample image unavailable offline: {exc}")
    img = cv2.imread(str(SAMPLE_PATH))
    assert img is not None
    return img


@pytest.fixture(scope="module")
def onnx_fp32(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("onnxruntime")
    from vidan.export import export_onnx

    return export_onnx("nano", None, tmp_path_factory.mktemp("onnx"))


def test_torch_and_onnx_fp32_agree_on_a_real_image(frame: np.ndarray, onnx_fp32: Path) -> None:
    torch_det = TorchRFDETR("nano", classes=None, threshold=0.3).detect(frame)
    onnx_det = OnnxRFDETR(onnx_fp32, classes=None, threshold=0.3).detect(frame)
    assert len(torch_det) >= 3  # dog, person, ... : a real scene, not an empty comparison
    assert abs(len(torch_det) - len(onnx_det)) <= 1
    ious = _matched_ious(torch_det, onnx_det)
    assert min(ious) > 0.98
    np.testing.assert_allclose(
        np.sort(torch_det.score)[::-1][:3], np.sort(onnx_det.score)[::-1][:3], atol=5e-3
    )


def test_class_filter_keeps_only_requested_ids(frame: np.ndarray, onnx_fp32: Path) -> None:
    everything = OnnxRFDETR(onnx_fp32, classes=None, threshold=0.3).detect(frame)
    persons = OnnxRFDETR(onnx_fp32, classes=(1,), threshold=0.3).detect(frame)
    assert set(persons.class_id.tolist()) == {1}
    assert len(persons) == int((everything.class_id == 1).sum()) >= 1


def test_int8_model_runs_and_finds_the_same_objects(
    frame: np.ndarray, onnx_fp32: Path, tmp_path: Path
) -> None:
    from vidan.export import quantize_int8

    int8 = quantize_int8(onnx_fp32, tmp_path / "nano-int8.onnx")
    assert int8.stat().st_size < onnx_fp32.stat().st_size
    ref = OnnxRFDETR(onnx_fp32, classes=None, threshold=0.3).detect(frame)
    got = OnnxRFDETR(int8, classes=None, threshold=0.3).detect(frame)
    # Quantisation may move scores across the threshold; the strongest detections must survive.
    top = ref.keep(ref.score > 0.5)
    assert min(_matched_ious(top, got)) > 0.9


def _box_frames(n: int, step: float = 5.0) -> dict[int, Detections]:
    per_frame = {}
    for f in range(1, n + 1):
        x = 100 + step * f
        per_frame[f] = Detections(
            [[x, 200, x + 40, 300], [800 - step * f, 200, 840 - step * f, 300]], [0.9, 0.8], [1, 1]
        )
    return per_frame


def test_bytetrack_gives_stable_ids_on_two_synthetic_walkers() -> None:
    tracks = run_bytetrack(_box_frames(6), frame_rate=30.0, minimum_consecutive_frames=1)
    # a track's first observation carries no id yet (trackers 2.6), so frames 2..6
    assert len(tracks) == 10
    assert set(tracks.track_id.tolist()) == {1, 2}
    for rows in tracks.by_id().values():
        assert tracks.frame[rows].tolist() == [2, 3, 4, 5, 6]


def test_bytetrack_two_frames_and_determinism() -> None:
    a = run_bytetrack(_box_frames(2), frame_rate=30.0, minimum_consecutive_frames=1)
    b = run_bytetrack(_box_frames(2), frame_rate=30.0, minimum_consecutive_frames=1)
    assert len(a) == 2 and sorted(a.track_id.tolist()) == [1, 2]  # both born in frame 1, ids in 2
    np.testing.assert_array_equal(a.track_id, b.track_id)
    np.testing.assert_array_equal(a.xywh, b.xywh)


def test_bytetrack_frame_skip_keeps_original_frame_numbers() -> None:
    tracks = run_bytetrack(
        _box_frames(10, step=2.0), frame_rate=30.0, frame_skip=3, minimum_consecutive_frames=1
    )
    assert sorted(set(tracks.frame.tolist())) == [4, 7, 10]  # fed: 1, 4, 7, 10; frame 1 has no ids
    assert len(set(tracks.track_id.tolist())) == 2


def test_bytetrack_handles_empty_frames() -> None:
    per_frame = {1: Detections.empty(), 2: Detections.empty()}
    assert len(run_bytetrack(per_frame, frame_rate=30.0)) == 0


def test_track_count_timing_runs_on_a_synthetic_sequence() -> None:
    from vidan.geometry import Line
    from vidan.speedbench import track_count_seconds

    secs, n = track_count_seconds(
        _box_frames(20),
        30.0,
        [Line((0.0, 250.0), (1920.0, 250.0)), Line((960.0, 0.0), (960.0, 1080.0))],
    )
    assert n == 20 and secs > 0
