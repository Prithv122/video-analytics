"""Unit tests for configs, cache bookkeeping, the resilient detect loop and the new CLI parsers."""

import numpy as np
import pytest

from vidan import runs
from vidan.cli import build_parser
from vidan.detect import Detections, save_cache


def test_configs_match_the_evaluation_plan() -> None:
    assert {c: (s.size, s.resolution, s.int8) for c, s in runs.CONFIGS.items()} == {
        "S": ("small", 512, False),
        "N": ("nano", 384, False),
        "S8": ("small", 512, True),
        "N8": ("nano", 384, True),
    }
    assert runs.config_spec("S8").base_id == "S" and runs.config_spec("N").base_id == "N"
    with pytest.raises(ValueError):
        runs.config_spec("M")


def test_paths(tmp_path) -> None:
    assert runs.model_path("N8", tmp_path) == tmp_path / "N8.onnx"
    assert runs.cache_path("S", "MOT17-04", tmp_path) == tmp_path / "S" / "MOT17-04.npz"


def test_cache_is_complete_checks_existence_frame_count_and_integrity(tmp_path) -> None:
    path = tmp_path / "S" / "seq.npz"
    assert not runs.cache_is_complete(path, 3)
    per_frame = {f: Detections([[0, 0, 1, 1]], [0.5], [1]) for f in (1, 2, 3)}
    save_cache(path, per_frame, {"detector": "fake"})
    assert runs.cache_is_complete(path, 3)
    assert not runs.cache_is_complete(path, 4)
    path.write_bytes(path.read_bytes()[:100])  # truncated
    assert not runs.cache_is_complete(path, 3)


def test_format_eta() -> None:
    assert runs.format_eta(0) == "0:00:00"
    assert runs.format_eta(3725.9) == "1:02:05"
    assert runs.format_eta(-5) == "0:00:00"


class _Det:
    name = "fake"

    def detect(self, frame: np.ndarray) -> Detections:
        if frame[0, 0, 0] == 7:
            raise RuntimeError("model blew up")
        return Detections([[0, 0, 5, 5]], [0.9], [1])


def test_detect_frames_survives_bad_frames_and_records_them() -> None:
    def loader(handle: int) -> np.ndarray:
        if handle == 2:
            raise ValueError("could not decode")
        return np.full((2, 2, 3), handle, dtype=np.uint8)

    log: list[str] = []
    items = [(1, 1), (2, 2), (3, 7), (4, 4)]
    per_frame, stats = runs.detect_frames(
        _Det(), items, loader, log=log.append, label="[t]", progress_every=2
    )
    assert list(per_frame) == [1, 2, 3, 4]
    assert [len(per_frame[f]) for f in (1, 2, 3, 4)] == [1, 0, 0, 1]
    assert [b["frame"] for b in stats["bad_frames"]] == [2, 3]
    assert "ValueError" in stats["bad_frames"][0]["error"]
    assert "RuntimeError" in stats["bad_frames"][1]["error"]
    progress = [m for m in log if "s/frame" in m]
    assert len(progress) == 2 and "2/4" in progress[0] and "4/4" in progress[1]
    assert "ETA" in progress[0]
    assert stats["sec_per_frame"] >= 0 and stats["seconds_taken"] >= stats["detect_seconds"]


def test_cli_parsers_for_the_new_commands() -> None:
    p = build_parser()
    a = p.parse_args(["detect", "--config", "N8", "--split", "test", "--threads", "6"])
    assert (a.config, a.split, a.threads) == ("N8", "test", 6)
    assert p.parse_args(["detect", "--config", "S"]).threads == 12
    assert p.parse_args(["export", "--config", "S8"]).config == "S8"
    b = p.parse_args(["bench", "--max-wait", "5", "--force"])
    assert b.force and b.max_wait == 5.0 and (b.frames, b.warmup, b.repeats) == (100, 10, 3)
    assert p.parse_args(["equiv"]).frames == 50
    with pytest.raises(SystemExit):
        p.parse_args(["detect", "--config", "M"])
