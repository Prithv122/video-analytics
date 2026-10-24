"""Mechanical line and zone placement, decided from the frame size alone.

The rule is fixed here, before any detector has run on a sequence, and is never tuned afterwards:

- `h_mid`: horizontal line at y = H/2, from (0, H/2) to (W, H/2). Positive ("in") side is
  below the line in the image (larger y); crossing downwards counts as +1.
- `v_mid`: vertical line at x = W/2, from (W/2, 0) to (W/2, H). Positive side is to the left
  (smaller x); crossing leftwards counts as +1.
- `centre`: the rectangle spanning the middle 50% of width and height, i.e. corners
  (W/4, H/4), (3W/4, H/4), (3W/4, 3H/4), (W/4, 3H/4).

The sides follow from the cross-product convention in `vidan.geometry`.
"""

import json
from pathlib import Path

from vidan.data import MANIFEST_PATH, SeqInfo, read_manifest
from vidan.geometry import Line, Zone

LAYOUT_PATH = Path("configs/lines.json")

RULE = (
    "h_mid: y=H/2 from x=0 to x=W; v_mid: x=W/2 from y=0 to y=H; "
    "centre: rectangle over the middle 50% of width and height. "
    "Placed from frame size only, before any prediction exists."
)


def build_layout(seqinfo: SeqInfo) -> dict:
    """Lines and zones for one sequence: `{"lines": {name: Line}, "zones": {name: Zone}}`."""
    w, h = float(seqinfo.width), float(seqinfo.height)
    return {
        "lines": {
            "h_mid": Line((0.0, h / 2), (w, h / 2)),
            "v_mid": Line((w / 2, 0.0), (w / 2, h)),
        },
        "zones": {
            "centre": Zone(
                ((w / 4, h / 4), (3 * w / 4, h / 4), (3 * w / 4, 3 * h / 4), (w / 4, 3 * h / 4))
            ),
        },
    }


def _line_to_json(line: Line) -> dict:
    return {"p1": list(line.p1), "p2": list(line.p2)}


def _zone_to_json(zone: Zone) -> dict:
    return {"points": [list(p) for p in zone.points]}


def layout_to_json(seqinfos: dict[str, SeqInfo]) -> dict:
    """JSON-ready dict for several sequences (keys sorted on write)."""
    seqs = {}
    for name, info in seqinfos.items():
        lay = build_layout(info)
        seqs[name] = {
            "width": info.width,
            "height": info.height,
            "lines": {k: _line_to_json(v) for k, v in lay["lines"].items()},
            "zones": {k: _zone_to_json(v) for k, v in lay["zones"].items()},
        }
    return {"rule": RULE, "sequences": seqs}


def seqinfos_from_manifest(manifest: dict) -> dict[str, SeqInfo]:
    """Frame size and rate per sequence, as recorded in the manifest."""
    return {
        name: SeqInfo(
            name=name,
            frame_rate=float(e["fps"]),
            seq_length=int(e["frame_count"]),
            width=int(e["width"]),
            height=int(e["height"]),
        )
        for name, e in manifest["sequences"].items()
    }


def write_layout(path: Path = LAYOUT_PATH, manifest_path: Path = MANIFEST_PATH) -> dict:
    """Generate the layout for every manifest sequence and write it to `path`."""
    doc = layout_to_json(seqinfos_from_manifest(read_manifest(manifest_path)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return doc


def read_layout(path: Path = LAYOUT_PATH) -> dict[str, dict]:
    """Load `configs/lines.json` as `{seq: {"lines": {name: Line}, "zones": {name: Zone}}}`."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for seq, entry in doc["sequences"].items():
        out[seq] = {
            "lines": {k: Line(tuple(v["p1"]), tuple(v["p2"])) for k, v in entry["lines"].items()},
            "zones": {
                k: Zone(tuple(tuple(p) for p in v["points"])) for k, v in entry["zones"].items()
            },
        }
    return out
