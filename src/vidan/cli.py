"""Console entry point: `vidan <subcommand>`.

Each subcommand lives in an `_add_<name>(subparsers)` function that registers its parser and sets
`func` to a handler taking the parsed namespace and returning an exit code. To add a command,
write one such function and add it to `_REGISTRARS`.
"""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from vidan import __version__


def _add_fetch(subparsers) -> None:
    p = subparsers.add_parser("fetch", help="download the evaluation sequences (partial zip reads)")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.set_defaults(func=_run_fetch)


def _run_fetch(args: argparse.Namespace) -> int:
    from vidan import data

    names = data.names_for_split(args.split)
    t0 = time.monotonic()
    stats = data.fetch(names, args.root)
    total = sum(s["bytes"] for s in stats.values())
    print(
        f"fetched {len(names)} sequences, {total / data.MIB:,.0f} MiB written "
        f"in {time.monotonic() - t0:,.0f} s -> {args.root}"
    )
    return 0


def _add_verify(subparsers) -> None:
    p = subparsers.add_parser("verify", help="check data/raw against data/manifest.json")
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    p.set_defaults(func=_run_verify)


def _run_verify(args: argparse.Namespace) -> int:
    from vidan import data

    problems = data.verify(args.root, args.manifest)
    if problems:
        for line in problems:
            print(f"MISMATCH {line}", file=sys.stderr)
        return 1
    n = len(data.read_manifest(args.manifest)["sequences"])
    print(f"OK: {n} sequences match {args.manifest}")
    return 0


def _add_layout(subparsers) -> None:
    p = subparsers.add_parser("layout", help="write configs/lines.json from the manifest")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    p.add_argument("--out", type=Path, default=Path("configs/lines.json"))
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help="first (re)build the manifest from --root",
    )
    p.add_argument("--root", type=Path, default=Path("data/raw"))
    p.set_defaults(func=_run_layout)


def _run_layout(args: argparse.Namespace) -> int:
    from vidan import data, layout

    if args.write_manifest:
        data.write_manifest(args.manifest, args.root)
        print(f"wrote {args.manifest}")
    doc = layout.write_layout(args.out, args.manifest)
    print(f"wrote {args.out} ({len(doc['sequences'])} sequences)")
    return 0


def _add_export(subparsers) -> None:
    p = subparsers.add_parser("export", help="export a detector config to models/<config>.onnx")
    p.add_argument("--config", required=True, choices=["S", "N", "S8", "N8"])
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.set_defaults(func=_run_export)


def _run_export(args: argparse.Namespace) -> int:
    from vidan import runs

    t0 = time.monotonic()
    path = runs.export_config(args.config, args.models_dir)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.monotonic() - t0:,.0f} s")
    return 0


def _add_detect(subparsers) -> None:
    p = subparsers.add_parser(
        "detect", help="cache person detections (score >= 0.10) per sequence with ONNX Runtime"
    )
    p.add_argument("--config", required=True, choices=["S", "N", "S8", "N8"])
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    p.set_defaults(func=_run_detect)


def _run_detect(args: argparse.Namespace) -> int:
    from vidan import runs

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    runs.detect_split(
        args.config,
        args.split,
        threads=args.threads,
        models_dir=args.models_dir,
        cache_root=args.cache_root,
        log=log,
    )
    return 0


def _add_bench(subparsers) -> None:
    p = subparsers.add_parser(
        "bench", help="CPU speed benchmark behind a quiet-machine gate (EVAL_PLAN: Speed benchmark)"
    )
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--max-wait", type=float, default=30.0, help="minutes to wait for a quiet machine"
    )
    p.add_argument(
        "--force", action="store_true", help="run past the gate; rows get contended=true"
    )
    p.add_argument("--configs", nargs="+", help="subset of the benchmark configs")
    p.add_argument("--out", type=Path, default=Path("results/bench.csv"))
    p.set_defaults(func=_run_bench)


def _run_bench(args: argparse.Namespace) -> int:
    from vidan import speedbench

    return speedbench.run_bench(args)


def _add_equiv(subparsers) -> None:
    p = subparsers.add_parser(
        "equiv", help="ONNX fp32 vs PyTorch detections on 50 evenly spaced dev frames"
    )
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--configs", nargs="+", default=["N", "S"], choices=["N", "S"])
    p.add_argument("--out", type=Path, default=Path("results/equivalence.csv"))
    p.set_defaults(func=_run_equiv)


def _run_equiv(args: argparse.Namespace) -> int:
    from vidan import speedbench

    return speedbench.run_equiv(args)


def _add_eval(subparsers) -> None:
    p = subparsers.add_parser(
        "eval", help="the frozen evaluation: tune | test | claims | frontier | summary"
    )
    p.add_argument("stage", choices=["tune", "test", "claims", "frontier", "summary"])
    p.add_argument("--configs", nargs="+", help="restrict to these detector configs (S N S8 N8)")
    p.add_argument("--seqs", nargs="+", help="restrict to these sequences (smoke runs only)")
    p.add_argument("--ks", nargs="+", type=int, help="restrict the frame-skip grid (smoke runs)")
    p.add_argument("--taus", nargs="+", type=float, help="restrict the threshold grid (smoke runs)")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--cache-root", type=Path, default=Path("data/cache"))
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.set_defaults(func=_run_eval)


def _run_eval(args: argparse.Namespace) -> int:
    from vidan import evalrun

    return evalrun.run_cli(args)


# One entry per subcommand; other modules add their own `_add_<name>` here.
_REGISTRARS: list[Callable[[argparse._SubParsersAction], None]] = [
    _add_fetch,
    _add_verify,
    _add_layout,
    _add_export,
    _add_detect,
    _add_bench,
    _add_equiv,
    _add_eval,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vidan", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"vidan {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for register in _REGISTRARS:
        register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        print(f"vidan {__version__}")
        parser.print_usage()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
