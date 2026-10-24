import json
from pathlib import Path

import pytest

from vidan import cli, data, layout
from vidan.data import SeqInfo
from vidan.geometry import Line, Zone, signed_distance


def info(w, h, name="s"):
    return SeqInfo(name=name, frame_rate=25.0, seq_length=10, width=w, height=h)


def test_layout_rule_1920x1080():
    lay = layout.build_layout(info(1920, 1080))
    assert lay["lines"]["h_mid"] == Line((0, 540), (1920, 540))
    assert lay["lines"]["v_mid"] == Line((960, 0), (960, 1080))
    assert lay["zones"]["centre"] == Zone(((480, 270), (1440, 270), (1440, 810), (480, 810)))


def test_layout_rule_1654x1080():
    lay = layout.build_layout(info(1654, 1080))
    assert lay["lines"]["h_mid"] == Line((0, 540), (1654, 540))
    assert lay["lines"]["v_mid"] == Line((827, 0), (827, 1080))
    assert lay["zones"]["centre"] == Zone(
        ((413.5, 270), (1240.5, 270), (1240.5, 810), (413.5, 810))
    )


def test_layout_in_side_convention():
    lay = layout.build_layout(info(1920, 1080))
    # h_mid: "in" is below the line (larger y); v_mid: "in" is to the left (smaller x)
    assert signed_distance(lay["lines"]["h_mid"], (100, 600)) == pytest.approx(60)
    assert signed_distance(lay["lines"]["h_mid"], (100, 500)) == pytest.approx(-40)
    assert signed_distance(lay["lines"]["v_mid"], (900, 100)) == pytest.approx(60)
    assert signed_distance(lay["lines"]["v_mid"], (1000, 100)) == pytest.approx(-40)


def fake_manifest(path: Path) -> None:
    seqs = {
        "MOT17-02": {"width": 1920, "height": 1080, "fps": 30, "frame_count": 600},
        "MOT20-01": {"width": 1920, "height": 734, "fps": 25, "frame_count": 429},
        "MOT20-02": {"width": 1654, "height": 1080, "fps": 25, "frame_count": 2782},
    }
    path.write_text(json.dumps({"sequences": seqs}))


def test_write_and_read_layout_round_trip(tmp_path):
    manifest = tmp_path / "m.json"
    fake_manifest(manifest)
    out = tmp_path / "configs" / "lines.json"
    doc = layout.write_layout(out, manifest)
    assert set(doc["sequences"]) == {"MOT17-02", "MOT20-01", "MOT20-02"}

    back = layout.read_layout(out)
    assert back["MOT20-02"]["lines"]["v_mid"] == Line((827, 0), (827, 1080))
    assert back["MOT20-01"]["lines"]["h_mid"] == Line((0, 367), (1920, 367))
    assert back["MOT17-02"] == layout.build_layout(info(1920, 1080))
    first = out.read_bytes()  # stable bytes
    layout.write_layout(out, manifest)
    assert out.read_bytes() == first


def test_cli_layout_and_subcommand_structure(tmp_path, capsys):
    manifest = tmp_path / "m.json"
    fake_manifest(manifest)
    out = tmp_path / "lines.json"
    assert cli.main(["layout", "--manifest", str(manifest), "--out", str(out)]) == 0
    assert out.is_file()
    assert "3 sequences" in capsys.readouterr().out
    assert cli.main([]) == 0  # no subcommand: prints version and usage
    with pytest.raises(SystemExit):
        cli.main(["fetch", "--split", "bogus"])


@pytest.mark.integration
def test_committed_layout_matches_committed_manifest():
    if not Path("configs/lines.json").is_file() or not Path("data/manifest.json").is_file():
        pytest.skip("layout not generated")
    committed = layout.read_layout(Path("configs/lines.json"))
    manifest = data.read_manifest(Path("data/manifest.json"))
    for name, seq_info in layout.seqinfos_from_manifest(manifest).items():
        assert committed[name] == layout.build_layout(seq_info)
