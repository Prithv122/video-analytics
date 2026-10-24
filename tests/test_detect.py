"""Unit tests for the numpy parts of `vidan.detect` (no torch, no onnxruntime, no weights)."""

import numpy as np
import pytest

from vidan.detect import (
    CAR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PERSON,
    Detections,
    cxcywh_to_xyxy,
    decode_outputs,
    filter_classes,
    load_cache,
    preprocess_bgr,
    rescale_boxes,
    resize_bilinear,
    save_cache,
    sigmoid,
    topk_pairs,
)


def test_detections_coerces_dtypes_and_validates() -> None:
    d = Detections([[0, 0, 10, 20]], [0.5], [1])
    assert d.xyxy.dtype == np.float32 and d.score.dtype == np.float32
    assert d.class_id.dtype == np.int64 and len(d) == 1
    assert len(Detections.empty()) == 0 and Detections.empty().xyxy.shape == (0, 4)
    with pytest.raises(ValueError):
        Detections([[0, 0, 1, 1]], [0.5, 0.6], [1])
    with pytest.raises(ValueError):
        Detections([[0, 0, 1]], [0.5], [1])


def test_cxcywh_to_xyxy() -> None:
    out = cxcywh_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.9, 0.2, 0.2]]))
    np.testing.assert_allclose(out, [[0.4, 0.3, 0.6, 0.7], [0.0, 0.8, 0.2, 1.0]], atol=1e-7)


def test_rescale_boxes_scales_and_clips() -> None:
    boxes = np.array([[0.1, 0.2, 0.5, 0.6], [-0.1, 0.5, 1.2, 1.1]], dtype=np.float32)
    out = rescale_boxes(boxes, width=200, height=100)
    np.testing.assert_allclose(out[0], [20, 20, 100, 60], atol=1e-4)
    np.testing.assert_allclose(out[1], [0, 50, 200, 100], atol=1e-4)


def test_sigmoid_is_stable_for_extreme_logits() -> None:
    out = sigmoid(np.array([-1e4, 0.0, 1e4], dtype=np.float32))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)
    assert np.isfinite(out).all()


def test_topk_pairs_orders_thresholds_and_allows_one_query_twice() -> None:
    scores = np.array(
        [
            [0.9, 0.1, 0.6],  # query 0: two classes clear 0.5
            [0.2, 0.3, 0.4],  # query 1: none
            [0.1, 0.7, 0.5],  # query 2: one (0.5 is not > 0.5)
        ],
        dtype=np.float32,
    )
    score, cls, query = topk_pairs(scores, threshold=0.5)
    np.testing.assert_allclose(score, [0.9, 0.7, 0.6])
    assert cls.tolist() == [0, 1, 2]
    assert query.tolist() == [0, 2, 0]


def test_topk_pairs_caps_and_breaks_ties_by_index() -> None:
    scores = np.full((2, 3), 0.8, dtype=np.float32)
    _, cls, query = topk_pairs(scores, threshold=0.5, num_select=4)
    # equal scores: ascending flat index (query-major) -> (0,0) (0,1) (0,2) (1,0)
    assert list(zip(query.tolist(), cls.tolist(), strict=True)) == [(0, 0), (0, 1), (0, 2), (1, 0)]
    empty = topk_pairs(np.zeros((2, 3), np.float32), threshold=0.5)
    assert all(len(a) == 0 for a in empty)
    with pytest.raises(ValueError):
        topk_pairs(np.zeros(3), 0.5)


def test_decode_outputs_end_to_end_on_synthetic_logits() -> None:
    boxes = np.array(
        [[0.25, 0.5, 0.5, 1.0], [0.75, 0.5, 0.5, 0.5], [0.5, 0.5, 0.1, 0.1]], dtype=np.float32
    )
    logits = np.full((3, 5), -10.0, dtype=np.float32)
    logits[0, PERSON] = 4.0  # sigmoid ~ 0.982
    logits[1, CAR] = 0.0  # 0.5
    logits[1, PERSON] = -0.2  # ~0.45, below the 0.48 threshold used here
    det = decode_outputs(boxes, logits, image_size=(400, 200), threshold=0.48)
    assert det.class_id.tolist() == [PERSON, CAR]
    np.testing.assert_allclose(det.score, [0.982, 0.5], atol=1e-3)
    np.testing.assert_allclose(det.xyxy, [[0, 0, 200, 200], [200, 50, 400, 150]], atol=1e-3)


def test_filter_classes() -> None:
    d = Detections(np.zeros((3, 4)), [0.9, 0.8, 0.7], [PERSON, CAR, PERSON])
    assert filter_classes(d, (PERSON,)).score.tolist() == pytest.approx([0.9, 0.7])
    assert len(filter_classes(d, ())) == 0
    assert filter_classes(d, None) is d


def test_resize_bilinear_half_pixel_hand_computed() -> None:
    img = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)[:, :, None]
    w = np.array([0.0, 0.25, 0.75, 1.0])  # source positions after the half-pixel shift and clip
    expected = w[None, :] + 2 * w[:, None]
    out = resize_bilinear(img, 4, 4)
    assert out.shape == (4, 4, 1)
    np.testing.assert_allclose(out[:, :, 0], expected, atol=1e-6)


def test_resize_bilinear_identity_and_constant() -> None:
    rng = np.random.default_rng(0)
    img = rng.random((7, 9, 3), dtype=np.float32)
    np.testing.assert_allclose(resize_bilinear(img, 7, 9), img, atol=1e-6)
    const = np.full((30, 50, 3), 0.3, dtype=np.float32)
    np.testing.assert_allclose(resize_bilinear(const, 8, 8), 0.3, atol=1e-6)


def test_preprocess_bgr_channel_order_scale_and_normalisation() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)  # BGR
    x = preprocess_bgr(frame, 16)
    assert x.shape == (1, 3, 16, 16) and x.dtype == np.float32 and x.flags.c_contiguous
    for ch, value in enumerate((30, 20, 10)):  # network channels are R, G, B
        want = (value / 255 - IMAGENET_MEAN[ch]) / IMAGENET_STD[ch]
        np.testing.assert_allclose(x[0, ch], want, atol=1e-5)


def test_numpy_resize_agrees_with_opencv_inter_linear() -> None:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(1)
    rgb = rng.random((90, 160, 3), dtype=np.float32)
    ours = resize_bilinear(rgb, 24, 24)
    theirs = cv2.resize(rgb, (24, 24), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_allclose(ours, theirs, atol=1e-5)


def test_cache_round_trip_keeps_frames_order_and_empties(tmp_path) -> None:
    per_frame = {
        3: Detections([[1, 2, 3, 4], [5, 6, 7, 8]], [0.9, 0.4], [PERSON, CAR]),
        1: Detections.empty(),
        2: Detections([[10, 10, 20, 30]], [0.75], [PERSON]),
    }
    path = tmp_path / "sub" / "seq.npz"
    save_cache(path, per_frame, {"detector": "fake", "resolution": 384, "sequence": "S1"})
    got, meta = load_cache(path)
    assert list(got) == [1, 2, 3]
    assert len(got[1]) == 0
    for f in per_frame:
        np.testing.assert_array_equal(got[f].xyxy, per_frame[f].xyxy)
        np.testing.assert_array_equal(got[f].score, per_frame[f].score)
        np.testing.assert_array_equal(got[f].class_id, per_frame[f].class_id)
    assert meta["detector"] == "fake" and meta["resolution"] == 384 and meta["sequence"] == "S1"
    assert meta["n_frames"] == 3 and "rfdetr" in meta["versions"]


def test_cache_round_trip_empty_mapping(tmp_path) -> None:
    save_cache(tmp_path / "e.npz", {}, {"detector": "none"})
    got, meta = load_cache(tmp_path / "e.npz")
    assert got == {} and meta["n_frames"] == 0
