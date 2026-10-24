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


# One entry per subcommand; other modules add their own `_add_<name>` here.
_REGISTRARS: list[Callable[[argparse._SubParsersAction], None]] = [
    _add_fetch,
    _add_verify,
    _add_layout,
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
