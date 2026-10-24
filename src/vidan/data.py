"""Dataset access: partial download of MOTChallenge zips, checksums, sequence index, frames.

The sequences used here live inside multi-gigabyte archives. Rather than download the archives,
`fetch` reads each zip's central directory with HTTP range requests and extracts only the members
it needs (`seqinfo.ini`, `gt/gt.txt`, `img1/*.jpg`) into `data/raw/<dataset>/train/<sequence>/`.

Licence: MOT17 and MOT20 are CC BY-NC-SA 3.0 and need attribution (see `ATTRIBUTION`). This
repository never redistributes the data; it ships only `data/manifest.json` (counts and
checksums) so a clean clone can confirm it fetched the same files.

Stdlib and numpy only. OpenCV is imported lazily by `load_frame`; nothing here needs torch or
onnxruntime.
"""

import configparser
import hashlib
import http.client
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_ROOT = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.json")

MIB = 1024 * 1024
USER_AGENT = "vidan-data/0.1 (+range-requests; partial zip extraction)"

LICENCE = "CC BY-NC-SA 3.0"
ATTRIBUTION = (
    "MOT17: Milan, Leal-Taixe, Reid, Roth, Schindler, 'MOT16: A Benchmark for Multi-Object "
    "Tracking', arXiv:1603.00831. MOT20: Dendorfer et al., 'MOT20: A benchmark for multi object "
    "tracking in crowded scenes', arXiv:2003.09003. Data from https://motchallenge.net, "
    "licensed CC BY-NC-SA 3.0; not redistributed by this repository."
)

DATASET_URLS: dict[str, dict[str, str]] = {
    "MOT17": {
        "images": "https://motchallenge.net/data/MOT17.zip",
        "labels": "https://motchallenge.net/data/MOT17Labels.zip",
    },
    "MOT20": {
        "images": "https://motchallenge.net/data/MOT20.zip",
        "labels": "https://motchallenge.net/data/MOT20Labels.zip",
    },
}


@dataclass(frozen=True)
class SeqSpec:
    """Where one sequence lives: its split, dataset and directory name inside the zip."""

    split: str  # "dev" | "test"
    dataset: str  # key of DATASET_URLS
    dir_name: str  # directory under <dataset>/train/ in the zip (and under root)


# Dev = static-camera MOT17 train sequences (FRCNN copy; the image files are the same across the
# DPM / FRCNN / SDP variants, only the detection files differ). Test = MOT20 train sequences.
SEQUENCES: dict[str, SeqSpec] = {
    "MOT17-02": SeqSpec("dev", "MOT17", "MOT17-02-FRCNN"),
    "MOT17-04": SeqSpec("dev", "MOT17", "MOT17-04-FRCNN"),
    "MOT17-09": SeqSpec("dev", "MOT17", "MOT17-09-FRCNN"),
    "MOT20-01": SeqSpec("test", "MOT20", "MOT20-01"),
    "MOT20-02": SeqSpec("test", "MOT20", "MOT20-02"),
    "MOT20-03": SeqSpec("test", "MOT20", "MOT20-03"),
    "MOT20-05": SeqSpec("test", "MOT20", "MOT20-05"),
}


def names_for_split(split: str) -> list[str]:
    """Sequence names in a split ("dev", "test" or "all"), in declaration order."""
    if split == "all":
        return list(SEQUENCES)
    if split not in {"dev", "test"}:
        raise ValueError(f"unknown split {split!r}; expected dev, test or all")
    return [n for n, s in SEQUENCES.items() if s.split == split]


def seq_dir(name: str, root: Path = DEFAULT_ROOT) -> Path:
    """Local directory of a sequence: `<root>/<dataset>/train/<dir_name>`."""
    spec = SEQUENCES[name]
    return Path(root) / spec.dataset / "train" / spec.dir_name


# --------------------------------------------------------------------------- range-request file


class RangeSource(Protocol):
    """Anything that can say how big a remote object is and return a byte range of it."""

    size: int

    def read_range(self, start: int, end: int) -> bytes:
        """Bytes `[start, end)`; must return exactly `end - start` bytes."""
        ...


class BytesSource:
    """In-memory `RangeSource`; used by tests and as a reference implementation."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.size = len(data)
        self.calls: list[tuple[int, int]] = []

    def read_range(self, start: int, end: int) -> bytes:
        self.calls.append((start, end))
        return self._data[start:end]


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, OSError | http.client.HTTPException | TimeoutError)


def with_retries[T](
    fn: Callable[[], T],
    *,
    retries: int = 5,
    backoff: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying transient network errors with exponential backoff."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= retries or not _retryable(exc):
                raise
            sleep(backoff * (2**attempt))
            attempt += 1


class UrlRangeSource:
    """`RangeSource` over HTTP range requests (urllib; follows redirects; retries)."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        retries: int = 5,
        backoff: float = 1.5,
        opener: Callable[..., http.client.HTTPResponse] = urllib.request.urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = url
        self._timeout = timeout
        self._retries = retries
        self._backoff = backoff
        self._opener = opener
        self._sleep = sleep
        self.size = with_retries(self._probe, retries=retries, backoff=backoff, sleep=sleep)

    def _request(self, start: int, end: int) -> tuple[http.client.HTTPResponse, bytes]:
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": USER_AGENT}
        )
        with self._opener(req, timeout=self._timeout) as resp:
            if resp.status != 206:
                raise RuntimeError(
                    f"{self.url}: server ignored the Range header (HTTP {resp.status})"
                )
            return resp, resp.read()

    def _probe(self) -> int:
        resp, _ = self._request(0, 1)
        m = re.fullmatch(r"bytes 0-0/(\d+)", resp.headers.get("Content-Range", ""))
        if not m:
            raise RuntimeError(f"{self.url}: no usable Content-Range header")
        return int(m.group(1))

    def read_range(self, start: int, end: int) -> bytes:
        def once() -> bytes:
            _, data = self._request(start, end)
            if len(data) != end - start:
                raise http.client.IncompleteRead(data, end - start - len(data))
            return data

        return with_retries(once, retries=self._retries, backoff=self._backoff, sleep=self._sleep)


class HTTPRangeFile(io.RawIOBase):
    """Read-only seekable file over a `RangeSource`, with block read-ahead and a small LRU.

    `zipfile.ZipFile` only needs `seek`, `tell` and `read`; reading a whole member sequentially
    costs one request per `block_size` bytes instead of one per few kilobytes.
    """

    def __init__(self, source: RangeSource, *, block_size: int = 4 * MIB, cache_blocks: int = 4):
        super().__init__()
        self._source = source
        self._size = source.size
        self._block = block_size
        self._max_blocks = cache_blocks
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0
        self.requests = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def _get_block(self, index: int) -> bytes:
        block = self._cache.get(index)
        if block is not None:
            self._cache.move_to_end(index)
            return block
        start = index * self._block
        end = min(start + self._block, self._size)
        block = self._source.read_range(start, end)
        self.requests += 1
        self.bytes_fetched += len(block)
        self._cache[index] = block
        while len(self._cache) > self._max_blocks:
            self._cache.popitem(last=False)
        return block

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        end = self._size if size is None or size < 0 else min(self._pos + size, self._size)
        parts: list[bytes] = []
        pos = self._pos
        while pos < end:
            index, offset = divmod(pos, self._block)
            block = self._get_block(index)
            chunk = block[offset : offset + (end - pos)]
            if not chunk:
                raise OSError("short block from range source")
            parts.append(chunk)
            pos += len(chunk)
        self._pos = pos
        return b"".join(parts)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


# --------------------------------------------------------------------------- extraction


def _default_progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _safe_dest(root: Path, member: str) -> Path:
    """Destination of a zip member under root; refuses absolute paths and `..` segments."""
    parts = member.split("/")
    if member.startswith("/") or ".." in parts or any(":" in p for p in parts):
        raise ValueError(f"unsafe zip member name {member!r}")
    return Path(root).joinpath(*parts)


def is_extracted(dest: Path, size: int) -> bool:
    """True when `dest` already exists with exactly the expected size (resume check)."""
    try:
        return dest.stat().st_size == size
    except OSError:
        return False


def extract_members(
    zf: zipfile.ZipFile,
    members: Sequence[zipfile.ZipInfo],
    root: Path,
    *,
    label: str = "",
    progress: Callable[[str], None] | None = _default_progress,
    min_interval: float = 2.0,
    member_retries: int = 3,
) -> tuple[int, int, int]:
    """Extract `members` under `root`, skipping those already present with the right size.

    Files are written to `<name>.part` then renamed, so an interrupted run never leaves a
    truncated file that looks complete. Returns `(extracted, skipped, bytes_written)`.
    """
    todo: list[tuple[zipfile.ZipInfo, Path]] = []
    skipped = 0
    for zi in members:
        if zi.is_dir():
            continue
        dest = _safe_dest(root, zi.filename)
        if is_extracted(dest, zi.file_size):
            skipped += 1
        else:
            todo.append((zi, dest))
    todo.sort(key=lambda t: t[0].header_offset)  # sequential in the archive = fewest requests

    extracted = 0
    written = 0
    total_bytes = sum(zi.file_size for zi, _ in todo)
    started = last = time.monotonic()
    for zi, dest in todo:
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(member_retries + 1):
            try:
                with zf.open(zi) as src, open(part, "wb") as out:
                    while chunk := src.read(1 * MIB):
                        out.write(chunk)
                break
            except (OSError, http.client.HTTPException, zipfile.BadZipFile):
                if attempt >= member_retries:
                    raise
                time.sleep(1.5 * (2**attempt))
        os.replace(part, dest)
        extracted += 1
        written += zi.file_size
        now = time.monotonic()
        if progress and (now - last >= min_interval or extracted == len(todo)):
            last = now
            rate = written / max(now - started, 1e-9) / MIB
            progress(
                f"  {label} {extracted}/{len(todo)} files  "
                f"{written / MIB:,.0f}/{total_bytes / MIB:,.0f} MiB  {rate:.1f} MiB/s"
            )
    return extracted, skipped, written


def _wanted(zf: zipfile.ZipFile, prefix: str) -> list[zipfile.ZipInfo]:
    keep = []
    for zi in zf.infolist():
        name = zi.filename
        if not name.startswith(prefix) or zi.is_dir():
            continue
        rel = name[len(prefix) :]
        if rel in {"seqinfo.ini", "gt/gt.txt"} or rel.startswith("img1/"):
            keep.append(zi)
    return keep


def fetch(
    names: Sequence[str],
    root: Path = DEFAULT_ROOT,
    *,
    progress: Callable[[str], None] | None = _default_progress,
    source_factory: Callable[[str], RangeSource] | None = None,
    block_size: int = 4 * MIB,
) -> dict[str, dict[str, int]]:
    """Download the named sequences into `root`, reading only the members they need.

    Per dataset, the main archive is opened once (central directory read via range requests);
    it holds `seqinfo.ini`, `gt/gt.txt` and `img1/` for the train sequences, so the separate
    labels archive is not needed. Already-extracted files with the right size are skipped, so
    re-running resumes. Returns per-sequence `{"extracted", "skipped", "bytes"}`.
    """
    unknown = [n for n in names if n not in SEQUENCES]
    if unknown:
        raise KeyError(f"unknown sequences: {unknown}; known: {list(SEQUENCES)}")
    make = source_factory or (lambda url: UrlRangeSource(url))
    root = Path(root)
    stats: dict[str, dict[str, int]] = {}

    by_dataset: dict[str, list[str]] = {}
    for n in names:
        by_dataset.setdefault(SEQUENCES[n].dataset, []).append(n)

    for dataset, seqs in by_dataset.items():
        urls = DATASET_URLS[dataset]
        if progress:
            progress(f"{dataset}: reading archive directory ({urls['images']})")
        rf = HTTPRangeFile(make(urls["images"]), block_size=block_size)
        try:
            with zipfile.ZipFile(rf) as zf:
                for n in seqs:
                    prefix = f"{dataset}/train/{SEQUENCES[n].dir_name}/"
                    members = _wanted(zf, prefix)
                    rels = {m.filename[len(prefix) :] for m in members}
                    if not {"seqinfo.ini", "gt/gt.txt"} <= rels or not any(
                        r.startswith("img1/") for r in rels
                    ):
                        raise RuntimeError(f"{n}: expected members not found under {prefix}")
                    if progress:
                        progress(f"{n}: {len(members)} members")
                    e, s, b = extract_members(zf, members, root, label=n, progress=progress)
                    stats[n] = {"extracted": e, "skipped": s, "bytes": b}
        finally:
            if progress:
                progress(
                    f"{dataset}: {rf.requests} range requests, "
                    f"{rf.bytes_fetched / MIB:,.0f} MiB transferred"
                )
    return stats


# --------------------------------------------------------------------------- sequence index


@dataclass(frozen=True)
class SeqInfo:
    """The fields of a MOTChallenge `seqinfo.ini`."""

    name: str
    frame_rate: float
    seq_length: int
    width: int
    height: int
    im_ext: str = ".jpg"
    im_dir: str = "img1"


def parse_seqinfo(text: str) -> SeqInfo:
    """Parse the text of a `seqinfo.ini` (case-insensitive keys, `[Sequence]` section)."""
    cp = configparser.ConfigParser()
    cp.read_string(text)
    if not cp.has_section("Sequence"):
        raise ValueError("seqinfo has no [Sequence] section")
    sec = cp["Sequence"]
    try:
        return SeqInfo(
            name=sec["name"].strip(),
            frame_rate=float(sec["frameRate"]),
            seq_length=int(sec["seqLength"]),
            width=int(sec["imWidth"]),
            height=int(sec["imHeight"]),
            im_ext=sec.get("imExt", ".jpg").strip(),
            im_dir=sec.get("imDir", "img1").strip(),
        )
    except KeyError as exc:
        raise ValueError(f"seqinfo is missing key {exc}") from exc


def read_seqinfo(path: Path) -> SeqInfo:
    """Read `seqinfo.ini`; `path` may be the file or the sequence directory."""
    p = Path(path)
    if p.is_dir():
        p = p / "seqinfo.ini"
    return parse_seqinfo(p.read_text(encoding="utf-8"))


def frames(seq: Path) -> Iterator[tuple[int, Path]]:
    """Yield `(frame_no, image_path)` for a sequence directory, in frame order (1-based)."""
    seq = Path(seq)
    img_dir = seq / "img1"
    found = []
    with os.scandir(img_dir) as it:
        for entry in it:
            stem, ext = os.path.splitext(entry.name)
            if ext.lower() in {".jpg", ".jpeg", ".png"} and stem.isdigit():
                found.append((int(stem), Path(entry.path)))
    found.sort()
    yield from found


def load_frame(path: Path) -> np.ndarray:
    """Decode an image to an `(h, w, 3)` uint8 BGR array. Needs OpenCV (imported lazily)."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "load_frame needs OpenCV; install it with `uv sync --extra vision`"
        ) from exc
    buf = np.fromfile(str(path), dtype=np.uint8)  # np.fromfile survives non-ASCII Windows paths
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image {path}")
    return img


# --------------------------------------------------------------------------- manifest


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 * MIB):
            h.update(chunk)
    return h.hexdigest()


def sequence_stats(seq: Path) -> dict:
    """Facts about one extracted sequence directory (everything the manifest records)."""
    seq = Path(seq)
    info = read_seqinfo(seq)
    gt = seq / "gt" / "gt.txt"
    listing = hashlib.sha256()
    n_images = 0
    for _, path in frames(seq):
        listing.update(f"{path.name}\t{path.stat().st_size}\n".encode())
        n_images += 1
    return {
        "frame_count": info.seq_length,
        "width": info.width,
        "height": info.height,
        "fps": info.frame_rate,
        "n_images": n_images,
        "gt_sha256": _sha256_file(gt),
        "images_sha256": listing.hexdigest(),
    }


def build_manifest(names: Sequence[str] | None = None, root: Path = DEFAULT_ROOT) -> dict:
    """Manifest dict for the extracted sequences (all of `SEQUENCES` by default)."""
    names = list(names) if names is not None else list(SEQUENCES)
    seqs = {}
    for n in names:
        spec = SEQUENCES[n]
        seqs[n] = {
            "split": spec.split,
            "dataset": spec.dataset,
            "dir": f"{spec.dataset}/train/{spec.dir_name}",
            **sequence_stats(seq_dir(n, root)),
        }
    return {
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "note": (
            "Checksums only; the data itself is not redistributed. images_sha256 is the "
            "SHA-256 of the sorted lines '<file name>\\t<size in bytes>\\n' of img1/."
        ),
        "sequences": seqs,
    }


def write_manifest(
    path: Path = MANIFEST_PATH, root: Path = DEFAULT_ROOT, names: Sequence[str] | None = None
) -> dict:
    """Compute the manifest from the data under `root` and write it (stable key order)."""
    manifest = build_manifest(names, root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(root: Path = DEFAULT_ROOT, manifest_path: Path = MANIFEST_PATH) -> list[str]:
    """Compare the data under `root` with the committed manifest; return problems (empty = OK)."""
    manifest = read_manifest(manifest_path)
    problems: list[str] = []
    for name, expected in manifest["sequences"].items():
        if name not in SEQUENCES:
            problems.append(f"{name}: not a known sequence")
            continue
        directory = seq_dir(name, root)
        if not (directory / "seqinfo.ini").is_file() or not (directory / "gt" / "gt.txt").is_file():
            problems.append(f"{name}: missing seqinfo.ini or gt/gt.txt under {directory}")
            continue
        try:
            actual = sequence_stats(directory)
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: unreadable ({exc})")
            continue
        for key, want in actual.items():
            if expected.get(key) != want:
                problems.append(f"{name}: {key} is {want!r}, manifest says {expected.get(key)!r}")
        if actual["n_images"] != actual["frame_count"]:
            problems.append(
                f"{name}: {actual['n_images']} images but seqinfo says {actual['frame_count']}"
            )
    return problems
