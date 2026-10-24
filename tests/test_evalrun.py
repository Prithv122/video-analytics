import json
import math
from pathlib import Path

import numpy as np
import pytest

from vidan import evalrun as ev
from vidan.geometry import Line
from vidan.mot import Tracks, tracks_from_rows

# ------------------------------------------------------------------------- synthetic world

LINES = {"h_mid": Line((0, 50), (100, 50)), "v_mid": Line((50, 0), (50, 100))}  # in: y > 50


def person(tid: int, x: float, frames=range(1, 13)) -> list:
    """Anchor at (x + 5 ... ) walking down across the horizontal line between frames 5 and 6."""
    return [(f, tid, x - 5, (45 if f < 6 else 55) - 10, 10, 10) for f in frames]


GT = tracks_from_rows(person(1, 20))
PERFECT = tracks_from_rows(person(100, 20))
JITTER = tracks_from_rows(person(100, 20) + person(200, 80))  # plus an unmatched crossing
EMPTY = Tracks.empty()
BY_TAU = {0.3: JITTER, 0.4: PERFECT, 0.5: PERFECT, 0.6: EMPTY}


def fake_sources() -> ev.Sources:
    return ev.Sources(
        detections=lambda config, seq: {},
        gt=lambda seq: GT,
        fps=lambda seq: 10.0,
        n_frames=lambda seq: 12,
        size=lambda seq: (100.0, 100.0),
        lines=lambda seq: LINES,
        track=lambda per_frame, fps, k, tau: BY_TAU[tau],
        n_bad_frames=lambda config, seq: 2 if seq == "A" else 0,
    )


# ---------------------------------------------------------------------------------- scoring


def test_score_cell_perfect_and_banded():
    scorer = ev.Scorer(fake_sources())
    naive = scorer.score_cell("S", "A", 1, 0.4, 0, events=True)
    assert len(naive.rows) == 4  # 2 lines x 2 directions
    down = next(r for r in naive.rows if r["line"] == "h_mid" and r["direction"] == 1)
    assert (down["n_true"], down["n_pred"], down["tp"], down["fp"], down["fn"]) == (1, 1, 1, 0, 0)
    assert down["tp_time"] == 1 and down["arm"] == "naive"
    assert sum(r["tp"] for r in naive.rows) == 1
    kinds = sorted((e["kind"], e["outcome"]) for e in naive.events)
    assert kinds == [("pred", "tp"), ("true", "tp")]
    assert naive.events[0]["block"] == 0

    banded = scorer.score_cell("S", "A", 1, 0.4, 8)  # +-5 px never leaves the band
    down = next(r for r in banded.rows if r["line"] == "h_mid" and r["direction"] == 1)
    assert (down["tp"], down["fp"], down["fn"]) == (0, 0, 1)
    assert down["fn_tracked_not_counted"] == 1 and down["arm"] == "hysteresis"
    assert scorer.n_tracked == 1  # the band change did not re-track


def test_score_cell_attributes_unmatched_and_missed():
    scorer = ev.Scorer(fake_sources())
    jitter = scorer.score_cell("S", "A", 1, 0.3, 0, events=True)
    down = next(r for r in jitter.rows if r["line"] == "h_mid" and r["direction"] == 1)
    assert (down["tp"], down["fp"], down["fn"], down["fp_unmatched"]) == (1, 1, 0, 1)
    assert {(e["outcome"], e["category"]) for e in jitter.events} == {
        ("tp", ""),
        ("fp", "unmatched"),
    }
    empty = scorer.score_cell("S", "A", 1, 0.6, 0)
    down = next(r for r in empty.rows if r["line"] == "h_mid" and r["direction"] == 1)
    assert (down["tp"], down["fn"], down["fn_missed"]) == (0, 1, 1)


def test_pool_metrics():
    rows = [
        {**dict.fromkeys(ev._SUM_KEYS, 0), "n_true": 3, "n_pred": 3, "tp": 1, "fp": 2, "fn": 2,
         "tp_time": 3, "fp_duplicate": 1, "fn_fragmented": 1},
        {**dict.fromkeys(ev._SUM_KEYS, 0), "n_true": 1, "n_pred": 2, "tp": 1, "fp": 1, "fn": 0,
         "tp_time": 1, "fp_spurious": 1},
    ]  # fmt: skip
    m = ev.pool(rows)
    assert m["f1"] == pytest.approx(2 * 2 / (2 * 2 + 3 + 2))
    assert m["wape"] == pytest.approx((0 + 1) / 4)  # per-cell net errors
    assert m["eer"] == pytest.approx(5 / 4)
    assert m["hidden"] == pytest.approx(5 / 4 - 1 / 4)
    assert m["f1_time"] == pytest.approx(8 / 9)  # tp 4, fp 1, fn 0
    assert m["switch_share"] == pytest.approx(2 / 5)
    assert ev.pool([])["f1"] == 1.0


# -------------------------------------------------------------------------------- selection


def grid_row(config="S", k=1, tau=0.5, band=0, f1=0.5):
    return {"config": config, "k": k, "tau": tau, "band": band, "f1": f1}


def test_selection_ties_prefer_higher_tau_then_smaller_band():
    grid = [
        grid_row(tau=0.3, band=0, f1=0.7),
        grid_row(tau=0.5, band=0, f1=0.7),  # naive tie -> higher tau
        grid_row(tau=0.6, band=0, f1=0.69),
        grid_row(tau=0.4, band=8, f1=0.8),
        grid_row(tau=0.4, band=16, f1=0.8),  # same f1, bigger band loses
        grid_row(tau=0.3, band=8, f1=0.8),  # same f1, lower tau loses
        grid_row(tau=0.4, band=32, f1=0.79),
    ]
    sel = {s["arm"]: s for s in ev.select(grid)}
    assert (sel["naive"]["tau"], sel["naive"]["band"]) == (0.5, 0)
    assert (sel["hysteresis"]["tau"], sel["hysteresis"]["band"]) == (0.4, 8)
    assert sel["hysteresis"]["dev_f1"] == 0.8


def test_selection_arms_are_separate_and_per_config_k():
    grid = [
        grid_row(tau=0.5, band=0, f1=0.9),  # band 0 is never a hysteresis candidate
        grid_row(tau=0.5, band=8, f1=0.6),
        grid_row(config="N", k=2, tau=0.3, band=0, f1=0.2),
    ]
    sel = ev.select(grid)
    assert [(s["config"], s["k"], s["arm"]) for s in sel] == [
        ("N", 2, "naive"),
        ("S", 1, "naive"),
        ("S", 1, "hysteresis"),
    ]
    assert next(s for s in sel if s["arm"] == "hysteresis")["dev_f1"] == 0.6


def test_selection_ignores_float_noise_in_ties():
    grid = [grid_row(tau=0.3, f1=0.7), grid_row(tau=0.6, f1=0.7 + 1e-15)]
    # numerically equal f1 is a tie, so the higher threshold wins either way round
    assert ev.select(grid)[0]["tau"] == 0.6
    assert ev.select(grid[::-1])[0]["tau"] == 0.6


def test_tune_writes_grid_and_selection(tmp_path):
    logs: list[str] = []
    grid, sel = ev.tune(
        fake_sources(),
        tmp_path,
        configs=("S",),
        ks=(1,),
        taus=(0.3, 0.4, 0.6),
        bands=(0, 8),
        seqs=("A", "B"),
        log=logs.append,
    )
    assert len(grid) == 3 * 2  # tau x band, pooled over both sequences
    naive = {g["tau"]: g["f1"] for g in grid if g["band"] == 0}
    assert naive[0.4] == 1.0 and naive[0.6] == 0.0 and 0 < naive[0.3] < 1
    chosen = {s["arm"]: s for s in sel}
    assert chosen["naive"]["tau"] == 0.4
    assert chosen["hysteresis"]["band"] == 8  # all zero: higher tau, then (only) band 8
    assert chosen["hysteresis"]["tau"] == 0.6
    rows = ev.read_csv(tmp_path / ev.DEV_SELECTION)
    assert rows[0]["tau"] in (0.4, 0.6) and isinstance(rows[0]["k"], int)
    assert (tmp_path / ev.DEV_GRID).exists() and any("dev grid" in line for line in logs)


# ------------------------------------------------------------------------------ git guard


class Run:
    def __init__(self, codes):
        self.codes, self.cmds = list(codes), []

    def __call__(self, cmd, **kw):
        self.cmds.append(cmd)
        return type("R", (), {"returncode": self.codes[len(self.cmds) - 1]})()


def test_selection_is_committed_checks_tracked_in_head_and_unmodified():
    ok = Run([0, 0, 0])
    assert ev.selection_is_committed(Path("results/dev_selection.csv"), ok) == (True, "")
    assert ok.cmds[0][:3] == ["git", "ls-files", "--error-unmatch"]
    assert ok.cmds[1][:3] == ["git", "cat-file", "-e"]
    assert ok.cmds[2][:4] == ["git", "diff", "--quiet", "HEAD"]
    assert "results/dev_selection.csv" in ok.cmds[0][-1]
    for codes, word in (([1], "not tracked"), ([0, 1], "HEAD"), ([0, 0, 1], "differs")):
        run = Run(codes)
        good, why = ev.selection_is_committed(Path("results/dev_selection.csv"), run)
        assert not good and word in why
        assert len(run.cmds) == len(codes)  # stops at the first failing check

    def boom(cmd, **kw):
        raise FileNotFoundError("git")

    assert not ev.selection_is_committed(Path("x.csv"), boom)[0]


def make_selection(tmp_path: Path) -> None:
    ev.write_csv(
        tmp_path / ev.DEV_SELECTION,
        [
            {"arm": "naive", "config": "S", "k": 1, "tau": 0.4, "band": 0, "dev_f1": 1.0},
            {"arm": "hysteresis", "config": "S", "k": 1, "tau": 0.4, "band": 8, "dev_f1": 0.0},
        ],
    )


def test_test_refuses_without_a_committed_selection(tmp_path):
    with pytest.raises(ev.SelectionNotCommitted, match="does not exist"):
        ev.test(fake_sources(), tmp_path, seqs=("A",), log=lambda s: None)
    make_selection(tmp_path)
    with pytest.raises(ev.SelectionNotCommitted, match="differs"):
        ev.test(
            fake_sources(),
            tmp_path,
            seqs=("A",),
            check_committed=lambda p: (False, f"{p.name} differs from the HEAD commit"),
            log=lambda s: None,
        )
    assert not (tmp_path / ev.TEST_CELLS).exists()


def test_test_scores_only_selected_settings(tmp_path):
    make_selection(tmp_path)
    cells = ev.test(
        fake_sources(),
        tmp_path,
        seqs=("A",),
        check_committed=lambda p: (True, ""),
        log=lambda s: None,
    )
    assert {(c["config"], c["k"], c["tau"], c["band"]) for c in cells} == {
        ("S", 1, 0.4, 0),
        ("S", 1, 0.4, 8),
    }
    assert len(cells) == 8  # 2 settings x 2 lines x 2 directions
    events = ev.read_csv(tmp_path / ev.TEST_EVENTS)
    assert {e["arm"] for e in events} == {"naive", "hysteresis"}
    assert (tmp_path / ev.TEST_MOT).exists()
    seqs = ev.read_csv(tmp_path / ev.TEST_SEQUENCES)
    assert seqs[0]["block_frames"] == "100" and seqs[0]["n_frames"] == "12"


# ---------------------------------------------------------------------------------- bootstrap


def event_rows(blocks, **kw):
    """blocks: list of (block, kind, outcome) -> event dict rows for one cell."""
    base = {"config": "S", "k": 1, "arm": "hysteresis", "seq": "A", "line": "h", "direction": 1}
    base.update(kw)
    return [
        {**base, "kind": kind, "outcome": outcome, "block": b, "frame": 1, "category": ""}
        for b, kind, outcome in blocks
    ]


UNIVERSE2 = [("A", 0), ("A", 1)]
TINY = event_rows(
    [
        (0, "pred", "tp"), (0, "true", "tp"),  # block 0: one hit
        (1, "pred", "fp"), (1, "true", "fn"),  # block 1: a miss and an extra
    ]
)  # fmt: skip


def test_bootstrap_tiny_case_by_hand():
    table = ev.build_table(TINY, UNIVERSE2, "S", 1, "hysteresis")
    w = np.array([[1, 1], [2, 0], [0, 2]], dtype=float)
    m = ev.pooled_metrics(table, w)
    assert m["f1"].tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert m["precision"].tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert m["recall"].tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert m["eer"].tolist() == pytest.approx([1.0, 0.0, 2.0])
    assert m["wape"].tolist() == pytest.approx([0.0, 0.0, 0.0])  # extras cancel misses in the net
    assert m["hidden"].tolist() == pytest.approx([1.0, 0.0, 2.0])


def test_wape_is_computed_on_cell_net_counts_of_the_resample():
    # one cell: block 0 predicts 3 for 1 true, block 1 predicts 1 for 3 true
    rows = event_rows(
        [(0, "pred", "fp")] * 3
        + [(0, "true", "fn")]
        + [(1, "pred", "fp")]
        + [(1, "true", "fn")] * 3
    )
    table = ev.build_table(rows, UNIVERSE2, "S", 1, "hysteresis")
    w = np.array([[1, 1], [2, 0], [0, 2]], dtype=float)
    wape = ev.pooled_metrics(table, w)["wape"]
    assert wape.tolist() == pytest.approx([0.0, 2.0, 2 / 3])  # |4-4|/4, |6-2|/2, |2-6|/6


def test_wape_sums_absolute_net_errors_over_cells():
    rows = event_rows([(0, "pred", "fp"), (0, "pred", "fp")], direction=1) + event_rows(
        [(0, "true", "fn")] * 2 + [(0, "pred", "fp")], direction=-1
    )
    table = ev.build_table(rows, [("A", 0)], "S", 1, "hysteresis")
    out = ev.pooled_metrics(table, np.array([[1.0]]))
    # cell +1: pred 2, true 0 -> 2 ; cell -1: pred 1, true 2 -> 1 ; sum true = 2
    assert out["wape"][0] == pytest.approx(3 / 2)


def test_resampling_is_deterministic_and_well_formed():
    a = ev.resample_weights(7, 50, seed=0)
    b = ev.resample_weights(7, 50, seed=0)
    c = ev.resample_weights(7, 50, seed=1)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert a.shape == (50, 7) and (a.sum(axis=1) == 7).all()
    assert ev.resample_weights(3, ev.N_RESAMPLES).shape == (2000, 3)


def test_paired_differences_use_the_same_blocks():
    good = ev.build_table(
        event_rows([(0, "pred", "tp"), (0, "true", "tp"), (1, "pred", "tp"), (1, "true", "tp")]),
        UNIVERSE2, "S", 1, "hysteresis",
    )  # fmt: skip
    bad = ev.build_table(
        event_rows(
            [(0, "pred", "fp"), (0, "true", "fn"), (1, "pred", "fp"), (1, "true", "fn")],
            arm="naive",
        ),
        UNIVERSE2, "S", 1, "naive",
    )  # fmt: skip
    w = ev.resample_weights(2, 200, seed=0)
    diff = ev.pooled_metrics(good, w)["f1"] - ev.pooled_metrics(bad, w)["f1"]
    assert (diff == 1.0).all()  # each resample scores both arms on identical blocks


def test_interval_and_lower_bound_decision():
    assert ev.interval(np.arange(101.0)) == pytest.approx((2.5, 97.5))
    assert ev.lower_bound_positive(np.array([0.1, 0.2, 0.3]))
    assert not ev.lower_bound_positive(np.zeros(10))  # lower bound exactly 0 is not > 0
    assert not ev.lower_bound_positive(np.array([-0.1] + [0.5] * 9 + [0.5] * 100 + [-0.1] * 10))
    assert ev.interval(np.array([np.nan, 1.0, 3.0]))[0] == pytest.approx(1.05)


# -------------------------------------------------------------------------- claims decisions


def test_speed_and_equivalence_boundaries():
    assert ev.decide_h3_speed(13.0, 10.0) and not ev.decide_h3_speed(12.9, 10.0)
    assert ev.decide_h4_speed(15.0, 10.0) and not ev.decide_h4_speed(14.9, 10.0)
    assert ev.equivalence_passes({"min_matched_iou": "0.99", "frac_frames_equal_count": "0.95"})
    assert not ev.equivalence_passes({"min_matched_iou": "0.989", "frac_frames_equal_count": "1"})
    assert not ev.equivalence_passes({"min_matched_iou": "1", "frac_frames_equal_count": "0.94"})
    assert ev.equivalence_passes({"passed": "true"})
    assert not ev.equivalence_passes({"passed": "false", "min_matched_iou": "1"})
    assert not ev.equivalence_passes({})


def write_events(tmp_path, per_arm: dict[tuple[str, str], str], n_blocks=12):
    """per_arm maps (config, arm) -> 'perfect' | 'bad'; one cell, one event pair per block."""
    rows = []
    for (config, arm), kind in per_arm.items():
        for b in range(n_blocks):
            if kind == "perfect":
                spec = [(b, "pred", "tp"), (b, "true", "tp")]
            else:  # a miss and an extra: net count exact, f1 = 0
                spec = [(b, "pred", "fp"), (b, "true", "fn")]
            rows += event_rows(spec, config=config, arm=arm)
    ev.write_csv(tmp_path / ev.TEST_EVENTS, rows, ev.EVENT_FIELDS)
    ev.write_csv(
        tmp_path / ev.TEST_SEQUENCES,
        [{"seq": "A", "fps": 10.0, "n_frames": 100 * n_blocks, "block_frames": 100}],
    )


def run_claims(tmp_path):
    return ev.claims(tmp_path, n_resamples=200, log=lambda s: None)


def test_claims_not_evaluable_without_any_results(tmp_path):
    out = run_claims(tmp_path)
    assert {k: v["status"] for k, v in out.items()} == dict.fromkeys(
        ["H1", "H2", "H3", "H4"], "not evaluable"
    )
    assert json.loads((tmp_path / ev.CLAIMS).read_text())["H1"]["status"] == "not evaluable"


def test_h1_and_h2_follow_the_lower_bound(tmp_path):
    # hidden error (miss + extra) in every block: eer = 2, wape = 0 -> H1 supported; hysteresis
    # perfect vs naive all wrong -> H2 supported
    write_events(tmp_path, {("S", "hysteresis"): "perfect", ("S", "naive"): "bad"})
    out = run_claims(tmp_path)
    assert out["H1"]["status"] == "not supported"  # perfect arm: eer = wape = 0, bound is 0
    assert out["H2"]["status"] == "supported"
    assert out["H2"]["estimate"] == pytest.approx(1.0)
    write_events(tmp_path, {("S", "hysteresis"): "bad", ("S", "naive"): "bad"})
    out = run_claims(tmp_path)
    assert out["H1"]["status"] == "supported" and out["H1"]["estimate"] == pytest.approx(2.0)
    assert out["H2"]["status"] == "not supported"  # identical arms: difference is exactly 0
    assert out["H3"]["status"] == "not evaluable" and out["H4"]["status"] == "not evaluable"


def bench(rows):
    return [
        {"config": c, "backend": b, "detector_fps": f, "contended": str(con).lower()}
        for c, b, f, con in rows
    ]


def test_h3_needs_speed_and_accuracy_and_uncontended_timings(tmp_path):
    write_events(tmp_path, {("S", "hysteresis"): "perfect", ("S8", "hysteresis"): "perfect"})
    rows = bench([("S", "onnx", 10.0, False), ("S8", "onnx", 13.0, False)])
    ev.write_csv(tmp_path / ev.BENCH, rows)
    assert run_claims(tmp_path)["H3"]["status"] == "supported"

    ev.write_csv(
        tmp_path / ev.BENCH, bench([("S", "onnx", 10.0, False), ("S8", "onnx", 12.9, False)])
    )
    assert run_claims(tmp_path)["H3"]["status"] == "not supported"  # too slow a speed-up

    write_events(tmp_path, {("S", "hysteresis"): "perfect", ("S8", "hysteresis"): "bad"})
    ev.write_csv(tmp_path / ev.BENCH, rows)
    assert run_claims(tmp_path)["H3"]["status"] == "not supported"  # f1 drop of 1.0 > 0.03

    ev.write_csv(
        tmp_path / ev.BENCH, bench([("S", "onnx", 10.0, False), ("S8", "onnx", 13.0, True)])
    )
    contended = run_claims(tmp_path)["H3"]
    assert contended["status"] == "not evaluable" and "contended" in contended["detail"]


def test_h4_needs_both_models_speed_and_equivalence(tmp_path):
    fast = bench(
        [("N", "onnx", 30.0, False), ("N", "torch", 20.0, False),
         ("S", "onnx", 15.0, False), ("S", "torch", 10.0, False)]
    )  # fmt: skip
    ev.write_csv(tmp_path / ev.BENCH, fast)
    assert run_claims(tmp_path)["H4"]["status"] == "not evaluable"  # no equivalence file
    good = [
        {"config": "N", "min_matched_iou": 0.999, "frac_frames_equal_count": 0.99},
        {"config": "S", "min_matched_iou": 0.995, "frac_frames_equal_count": 0.96},
    ]
    ev.write_csv(tmp_path / ev.EQUIVALENCE, good)
    out = run_claims(tmp_path)["H4"]
    assert out["status"] == "supported" and out["detail"]["S"]["fps_ratio"] == pytest.approx(1.5)
    ev.write_csv(tmp_path / ev.EQUIVALENCE, [good[0], {**good[1], "frac_frames_equal_count": 0.9}])
    assert run_claims(tmp_path)["H4"]["status"] == "not supported"
    slow = bench(
        [("N", "onnx", 30.0, False), ("N", "torch", 20.0, False),
         ("S", "onnx", 14.0, False), ("S", "torch", 10.0, False)]
    )  # fmt: skip
    ev.write_csv(tmp_path / ev.BENCH, slow)
    ev.write_csv(tmp_path / ev.EQUIVALENCE, good)
    assert run_claims(tmp_path)["H4"]["status"] == "not supported"
    ev.write_csv(
        tmp_path / ev.BENCH, bench([("N", "onnx", 30.0, True), ("N", "torch", 20.0, False)])
    )
    assert run_claims(tmp_path)["H4"]["status"] == "not evaluable"


# ----------------------------------------------------------------------------------- frontier


def cell(config, k, arm, tp, fp, fn):
    row = dict.fromkeys(ev.CELL_FIELDS, 0)
    row.update(config=config, seq="A", line="h", direction=1, k=k, tau=0.5, arm=arm)
    row.update(band=8 if arm == "hysteresis" else 0, tp=tp, fp=fp, fn=fn)
    row.update(n_true=tp + fn, n_pred=tp + fp)
    return row


def test_frontier_marks_the_cheapest_point_within_tolerance(tmp_path):
    cells = [
        cell("S", 1, "hysteresis", 100, 0, 0),  # f1 1.0
        cell("N", 1, "hysteresis", 99, 1, 1),  # f1 .99 -> within 0.02
        cell("N", 3, "hysteresis", 90, 10, 10),  # f1 .9 -> outside
        cell("S", 1, "naive", 10, 0, 90),  # naive arm ignored
    ]
    ev.write_csv(tmp_path / ev.TEST_CELLS, cells, ev.CELL_FIELDS)
    rows = ev.frontier(tmp_path, log=lambda s: None)
    assert all(r["pipeline_fps"] is None for r in rows) and not any(
        r["cheapest_within"] for r in rows
    )

    ev.write_csv(
        tmp_path / ev.BENCH,
        [
            {"config": "S", "backend": "onnx", "k": 1, "detector_fps": 5.0, "pipeline_fps": 4.0},
            {"config": "N", "backend": "onnx", "k": 1, "detector_fps": 20.0, "t_track_count": 0.05},
        ],
    )
    rows = ev.frontier(tmp_path, log=lambda s: None)
    by = {(r["config"], r["k"]): r for r in rows}
    assert by[("N", 1)]["pipeline_fps"] == pytest.approx(1 / (1 / 20 + 0.05))  # 10 fps
    assert by[("N", 1)]["cheapest_within"] and not by[("S", 1)]["cheapest_within"]
    assert not by[("N", 3)]["within_0.02_of_S1"] and not by[("N", 3)]["cheapest_within"]
    text = (tmp_path / ev.FRONTIER).read_text()
    assert text.startswith("config,k,f1,pipeline_fps")


def test_csv_round_trip_parses_known_columns(tmp_path):
    ev.write_csv(tmp_path / "x.csv", [{"config": "S", "k": 2, "tau": 0.3, "tp": 5, "note": "a"}])
    (row,) = ev.read_csv(tmp_path / "x.csv")
    assert row == {"config": "S", "k": 2, "tau": 0.3, "tp": 5, "note": "a"}
    assert math.isnan(ev._div(1, 0))


# ------------------------------------------------------------------ review regressions


def test_pipeline_fps_at_k_comes_from_the_k1_row():
    bench = [  # what the benchmark writes: k = 1 rows only
        {"config": "S", "backend": "onnx", "k": 1, "detector_fps": 5.0, "t_track_count": 0.05},
        {"config": "S", "backend": "torch", "k": 1, "detector_fps": 1.0, "t_track_count": 0.05},
    ]
    assert ev.pipeline_fps(bench, "S", 1) == pytest.approx(1 / (0.2 + 0.05))
    assert ev.pipeline_fps(bench, "S", 3) == pytest.approx(3 / (0.2 + 0.05))
    assert ev.pipeline_fps(bench, "S", 5) == pytest.approx(5 / (0.2 + 0.05))
    assert ev.pipeline_fps(bench, "N", 3) is None
    assert ev.pipeline_fps([{"config": "S", "k": 1, "detector_fps": 5.0}], "S", 2) is None


def test_frontier_has_fps_for_every_k_from_a_k1_only_bench(tmp_path):
    cells = [cell("S", 1, "hysteresis", 100, 0, 0), cell("S", 3, "hysteresis", 100, 0, 0)]
    ev.write_csv(tmp_path / ev.TEST_CELLS, cells, ev.CELL_FIELDS)
    ev.write_csv(
        tmp_path / ev.BENCH,
        [{"config": "S", "backend": "onnx", "k": 1, "detector_fps": 5.0, "t_track_count": 0.05}],
    )
    by = {r["k"]: r for r in ev.frontier(tmp_path, log=lambda s: None)}
    assert by[3]["pipeline_fps"] == pytest.approx(3 / 0.25)
    assert by[3]["cheapest_within"] and not by[1]["cheapest_within"]


def clipped_world() -> ev.Sources:
    """A person walks right to left across x = 50 with the box bottom below the 100 px frame."""
    rows = [(f, 1, 50 + (6 - f) * 4 - 5, 60, 10, 50) for f in range(1, 12)]  # bottom at y = 110
    gt = tracks_from_rows(rows)
    src = fake_sources()
    src.gt = lambda seq: gt
    src.track = lambda per_frame, fps, k, tau: gt.clip(100, 100)  # predictions are clipped
    return src


def test_gt_is_clipped_so_truth_counts_what_the_clipped_prediction_counts():
    from vidan.counting import count_line

    src = clipped_world()
    v_mid = LINES["v_mid"]
    raw = src.gt("A")
    assert count_line(raw, v_mid).events == ()  # the foot anchor at y = 110 is off the segment
    assert len(count_line(raw.clip(100, 100), v_mid).events) == 1
    rows = ev.Scorer(src).score_cell("S", "A", 1, 0.4, 0).rows
    v = [r for r in rows if r["line"] == "v_mid"]
    assert sum(r["n_true"] for r in v) == 1 and sum(r["n_pred"] for r in v) == 1
    assert sum(r["tp"] for r in v) == 1 and sum(r["fn"] + r["fp"] for r in v) == 0


def test_check_frames_requires_exactly_1_to_seq_length():
    full = dict.fromkeys(range(1, 6))
    ev.check_frames(full, 5, "S/A")
    with pytest.raises(ValueError, match="1 missing"):
        ev.check_frames({f: None for f in (1, 2, 3, 5)}, 5, "S/A")
    with pytest.raises(ValueError, match="extra"):
        ev.check_frames({**full, 9: None}, 5, "S/A")


def test_n_bad_frames_reaches_the_cells_and_the_grid(tmp_path):
    grid, _ = ev.tune(
        fake_sources(), tmp_path, configs=("S",), ks=(1,), taus=(0.4,), bands=(0,),
        seqs=("A", "B"), log=lambda s: None,
    )  # fmt: skip
    assert grid[0]["n_bad_frames"] == 2  # per sequence once, not once per line and direction
    assert "n_bad_frames" in ev.CELL_FIELDS
    cells = ev.Scorer(fake_sources()).score_cell("S", "A", 1, 0.4, 0).rows
    assert {r["n_bad_frames"] for r in cells} == {2}


def cli_args(stage, results_dir, **kw):
    ns = dict(stage=stage, results_dir=results_dir, configs=None, seqs=None, ks=None, taus=None)
    ns.update(kw)
    return type("Args", (), ns)()


def test_cli_forbids_subset_flags_on_test_and_on_tune_into_results(tmp_path, capsys):
    for flag in ("configs", "seqs", "ks", "taus"):
        assert ev.run_cli(cli_args("test", tmp_path, **{flag: ["x"]})) == 2
    assert "once" in capsys.readouterr().err
    assert ev.run_cli(cli_args("tune", ev.RESULTS, configs=["S"])) == 2
    assert "elsewhere" in capsys.readouterr().err


def test_test_refuses_a_second_run_while_test_files_exist(tmp_path):
    make_selection(tmp_path)
    kw = dict(seqs=("A",), check_committed=lambda p: (True, ""), log=lambda s: None)
    ev.test(fake_sources(), tmp_path, **kw)
    with pytest.raises(ev.RepeatTestRun, match="delete them by hand"):
        ev.test(fake_sources(), tmp_path, **kw)
    for p in tmp_path.glob("test_*.csv"):
        p.unlink()
    ev.test(fake_sources(), tmp_path, **kw)  # allowed again once the files are gone
