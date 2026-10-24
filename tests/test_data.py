import hashlib
import http.client
import io
import json
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest

from vidan import cli, data
from vidan.data import (
    BytesSource,
    HTTPRangeFile,
    UrlRangeSource,
    frames,
    parse_seqinfo,
    with_retries,
)

SEQINFO = """[Sequence]
name=MOT20-01
imDir=img1
frameRate=25
seqLength=3
imWidth=1920
imHeight=734
imExt=.jpg
"""


def make_zip(files: dict[str, bytes], compress: int = zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compress) as zf:
        for name, payload in files.items():
            zf.writestr(name, payload)
    return buf.getvalue()


def sequence_files(prefix: str = "MOT20/train/MOT20-01/", n: int = 3) -> dict[str, bytes]:
    files = {
        prefix + "seqinfo.ini": SEQINFO.replace("seqLength=3", f"seqLength={n}").encode(),
        prefix + "gt/gt.txt": b"1,1,10,10,5,5,1,1,1\n2,1,12,10,5,5,1,1,1\n",
        prefix + "det/det.txt": b"1,-1,0,0,1,1,0.5\n",  # must not be extracted
    }
    for i in range(1, n + 1):
        files[prefix + f"img1/{i:06d}.jpg"] = bytes([i]) * (1000 * i)
    return files


# --------------------------------------------------------------------------- seqinfo


def test_parse_seqinfo():
    info = parse_seqinfo(SEQINFO)
    assert info.name == "MOT20-01"
    assert info.frame_rate == 25.0
    assert info.seq_length == 3
    assert (info.width, info.height) == (1920, 734)
    assert info.im_ext == ".jpg"
    assert info.im_dir == "img1"


def test_parse_seqinfo_is_case_insensitive_and_reports_missing_keys():
    assert parse_seqinfo(SEQINFO.lower().replace("[sequence]", "[Sequence]")).width == 1920
    with pytest.raises(ValueError, match=r"imHeight|imheight"):
        parse_seqinfo(SEQINFO.replace("imHeight=734\n", ""))
    with pytest.raises(ValueError, match="Sequence"):
        parse_seqinfo("[Other]\nx=1\n")


def test_read_seqinfo_accepts_file_or_directory(tmp_path):
    (tmp_path / "seqinfo.ini").write_text(SEQINFO)
    assert data.read_seqinfo(tmp_path) == data.read_seqinfo(tmp_path / "seqinfo.ini")


# --------------------------------------------------------------------------- range file


def test_range_file_reads_zip_members_across_blocks():
    files = sequence_files(n=5)
    src = BytesSource(make_zip(files))
    f = HTTPRangeFile(src, block_size=512)
    with zipfile.ZipFile(f) as zf:
        assert set(zf.namelist()) == set(files)
        for name, payload in files.items():
            assert zf.read(name) == payload
    assert f.requests == len(src.calls) > 1
    assert all(end - start <= 512 for start, end in src.calls)


def test_range_file_seek_tell_and_eof():
    blob = bytes(range(256)) * 10
    f = HTTPRangeFile(BytesSource(blob), block_size=100)
    assert f.seek(250) == 250
    assert f.read(100) == blob[250:350]
    assert f.tell() == 350
    assert f.seek(-5, io.SEEK_END) == len(blob) - 5
    assert f.read() == blob[-5:]
    assert f.read(10) == b""
    f.seek(-10, io.SEEK_CUR)  # from EOF
    assert f.read(3) == blob[-10:-7]
    assert f.seek(0) == 0
    assert f.read() == blob
    with pytest.raises(ValueError):
        f.seek(-1)


def test_range_file_caches_blocks_and_bounds_memory():
    src = BytesSource(bytes(1000))
    f = HTTPRangeFile(src, block_size=100, cache_blocks=2)
    f.read(50)
    f.seek(0)
    f.read(50)
    assert len(src.calls) == 1  # second read served from the cached block
    f.seek(0)
    f.read(1000)
    assert len(f._cache) == 2


# --------------------------------------------------------------------------- HTTP layer


class FakeResponse:
    def __init__(self, status, headers, body):
        self.status = status
        self.headers = headers
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """Stands in for `urlopen`: serves ranges of `blob`, optionally failing the first calls."""

    def __init__(self, blob: bytes, failures: list[Exception] | None = None, honour_range=True):
        self.blob = blob
        self.failures = list(failures or [])
        self.honour_range = honour_range
        self.requests: list[str] = []

    def __call__(self, req, timeout=None):
        rng = req.get_header("Range")
        self.requests.append(rng)
        if self.failures:
            raise self.failures.pop(0)
        if not self.honour_range:
            return FakeResponse(200, {}, self.blob)
        start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
        body = self.blob[start : end + 1]
        headers = {"Content-Range": f"bytes {start}-{start + len(body) - 1}/{len(self.blob)}"}
        return FakeResponse(206, headers, body)


def no_sleep(_seconds):
    pass


def test_url_range_source_probe_and_read():
    server = FakeServer(b"abcdefghij")
    src = UrlRangeSource("http://x/f.zip", opener=server, sleep=no_sleep)
    assert src.size == 10
    assert src.read_range(2, 6) == b"cdef"
    assert server.requests[-1] == "bytes=2-5"


def test_url_range_source_retries_transient_errors():
    server = FakeServer(
        b"0123456789",
        failures=[
            urllib.error.URLError("reset"),
            urllib.error.HTTPError("http://x", 503, "busy", {}, None),
        ],
    )
    sleeps = []
    src = UrlRangeSource("http://x", opener=server, sleep=sleeps.append)
    assert src.size == 10
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]  # exponential backoff


def test_url_range_source_rejects_servers_without_range_support():
    with pytest.raises(RuntimeError, match="Range"):
        UrlRangeSource("http://x", opener=FakeServer(b"abc", honour_range=False), sleep=no_sleep)


def test_url_range_source_retries_short_reads_then_gives_up():
    class Truncating(FakeServer):
        def __call__(self, req, timeout=None):
            resp = super().__call__(req, timeout)
            if req.get_header("Range") == "bytes=0-3":
                return FakeResponse(206, resp.headers, resp.read()[:2])
            return resp

    src = UrlRangeSource("http://x", opener=Truncating(b"0123456789"), retries=2, sleep=no_sleep)
    with pytest.raises(http.client.IncompleteRead):
        src.read_range(0, 4)


def test_with_retries_does_not_retry_client_errors():
    calls = []

    def fn():
        calls.append(1)
        raise urllib.error.HTTPError("http://x", 404, "nope", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        with_retries(fn, sleep=no_sleep)
    assert len(calls) == 1


# --------------------------------------------------------------------------- fetch / resume


def fetch_from(blob: bytes, root: Path, names=("MOT20-01",), **kw):
    sources = []

    def factory(url):
        sources.append(BytesSource(blob))
        return sources[-1]

    stats = data.fetch(list(names), root, progress=None, source_factory=factory, **kw)
    return stats, sources


def test_fetch_extracts_only_needed_members(tmp_path):
    files = sequence_files()
    stats, _ = fetch_from(make_zip(files), tmp_path, block_size=300)
    seq = tmp_path / "MOT20" / "train" / "MOT20-01"
    assert stats["MOT20-01"]["extracted"] == 5  # seqinfo, gt, 3 images
    assert not (seq / "det").exists()
    for name, payload in files.items():
        if "/det/" not in name:
            assert (tmp_path / name).read_bytes() == payload
    assert not list(tmp_path.rglob("*.part"))
    assert [n for n, _ in frames(seq)] == [1, 2, 3]


def test_fetch_resumes_and_skips_complete_members(tmp_path):
    blob = make_zip(sequence_files())
    fetch_from(blob, tmp_path)
    seq = tmp_path / "MOT20" / "train" / "MOT20-01"
    (seq / "img1" / "000002.jpg").unlink()  # missing
    (seq / "img1" / "000003.jpg").write_bytes(b"trunc")  # wrong size
    stats, _ = fetch_from(blob, tmp_path)
    assert stats["MOT20-01"] == {"extracted": 2, "skipped": 3, "bytes": 2000 + 3000}
    assert (seq / "img1" / "000003.jpg").read_bytes() == bytes([3]) * 3000

    stats, _ = fetch_from(blob, tmp_path)
    assert stats["MOT20-01"]["extracted"] == 0


def test_fetch_resume_does_not_download_skipped_member_data(tmp_path):
    files = sequence_files(n=3)
    big = "MOT20/train/MOT20-01/img1/000002.jpg"
    files[big] = bytes(range(256)) * 400  # ~100 KB, stored so its bytes sit in the archive as-is
    blob = make_zip(files, compress=zipfile.ZIP_STORED)
    fetch_from(blob, tmp_path, block_size=4096)
    _, sources = fetch_from(blob, tmp_path, block_size=4096)
    zi = zipfile.ZipFile(io.BytesIO(blob)).getinfo(big)
    body_start, body_end = zi.header_offset + 4096, zi.header_offset + zi.file_size
    # a second run reads the central directory only: no request lies inside the big member's body
    assert [(s, e) for s, e in sources[0].calls if s < body_end and e > body_start] == []


def test_fetch_unknown_sequence_and_missing_members(tmp_path):
    with pytest.raises(KeyError):
        data.fetch(["nope"], tmp_path, progress=None)
    blob = make_zip({"MOT20/train/MOT20-01/seqinfo.ini": SEQINFO.encode()})
    with pytest.raises(RuntimeError, match="not found"):
        fetch_from(blob, tmp_path)


def test_safe_dest_rejects_traversal(tmp_path):
    for bad in ["../x", "a/../../x", "/abs", "C:/x"]:
        with pytest.raises(ValueError):
            data._safe_dest(tmp_path, bad)
    assert data._safe_dest(tmp_path, "a/b.txt") == tmp_path / "a" / "b.txt"


def test_names_for_split():
    assert data.names_for_split("dev") == ["MOT17-02", "MOT17-04", "MOT17-09"]
    assert data.names_for_split("test") == ["MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
    assert len(data.names_for_split("all")) == 7
    with pytest.raises(ValueError):
        data.names_for_split("train")


# --------------------------------------------------------------------------- manifest


def build_tree(root: Path, name="MOT20-01", n=3) -> Path:
    seq = data.seq_dir(name, root)
    for rel, payload in sequence_files("", n).items():
        if rel.startswith("det/"):
            continue
        path = seq / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return seq


def test_manifest_round_trip_and_verify(tmp_path):
    seq = build_tree(tmp_path / "raw")
    manifest_path = tmp_path / "manifest.json"
    written = data.write_manifest(manifest_path, tmp_path / "raw", names=["MOT20-01"])
    assert data.read_manifest(manifest_path) == written

    entry = written["sequences"]["MOT20-01"]
    assert entry["split"] == "test" and entry["dataset"] == "MOT20"
    assert (entry["frame_count"], entry["width"], entry["height"], entry["fps"]) == (
        3,
        1920,
        734,
        25,
    )
    assert entry["n_images"] == 3
    assert entry["gt_sha256"] == hashlib.sha256((seq / "gt" / "gt.txt").read_bytes()).hexdigest()
    listing = "".join(f"{i:06d}.jpg\t{1000 * i}\n" for i in (1, 2, 3))
    assert entry["images_sha256"] == hashlib.sha256(listing.encode()).hexdigest()
    assert "CC BY-NC-SA" in written["licence"]

    assert data.verify(tmp_path / "raw", manifest_path) == []
    first = manifest_path.read_bytes()  # deterministic bytes: rewriting gives an identical file
    data.write_manifest(manifest_path, tmp_path / "raw", names=["MOT20-01"])
    assert manifest_path.read_bytes() == first


def test_verify_detects_changes(tmp_path):
    seq = build_tree(tmp_path / "raw")
    manifest_path = tmp_path / "manifest.json"
    data.write_manifest(manifest_path, tmp_path / "raw", names=["MOT20-01"])

    (seq / "gt" / "gt.txt").write_bytes(b"tampered\n")
    (seq / "img1" / "000002.jpg").write_bytes(b"short")
    problems = data.verify(tmp_path / "raw", manifest_path)
    assert any("gt_sha256" in p for p in problems)
    assert any("images_sha256" in p for p in problems)

    (seq / "img1" / "000003.jpg").unlink()
    problems = data.verify(tmp_path / "raw", manifest_path)
    assert any("n_images" in p for p in problems)
    assert any("2 images but seqinfo says 3" in p for p in problems)


def test_verify_reports_missing_sequence(tmp_path):
    build_tree(tmp_path / "raw")
    manifest_path = tmp_path / "manifest.json"
    data.write_manifest(manifest_path, tmp_path / "raw", names=["MOT20-01"])
    problems = data.verify(tmp_path / "empty", manifest_path)
    assert len(problems) == 1 and "missing" in problems[0]


def test_cli_verify_exit_codes(tmp_path):
    raw = tmp_path / "raw"
    build_tree(raw)
    manifest = tmp_path / "m.json"
    data.write_manifest(manifest, raw, names=["MOT20-01"])
    args = ["verify", "--root", str(raw), "--manifest", str(manifest)]
    assert cli.main(args) == 0
    (data.seq_dir("MOT20-01", raw) / "gt" / "gt.txt").write_bytes(b"x")
    assert cli.main(args) == 1


# --------------------------------------------------------------------------- frames / load_frame


def test_frames_sorted_numerically_and_filters_other_files(tmp_path):
    img = tmp_path / "img1"
    img.mkdir()
    for n in (10, 2, 1, 100):
        (img / f"{n:06d}.jpg").write_bytes(b"x")
    (img / "Thumbs.db").write_bytes(b"x")
    (img / "000005.jpg.part").write_bytes(b"x")
    assert [n for n, _ in frames(tmp_path)] == [1, 2, 10, 100]


def test_load_frame_needs_opencv_only_when_called(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)  # simulate cv2 not installed
    with pytest.raises(ImportError, match="OpenCV"):
        data.load_frame(tmp_path / "x.jpg")


def test_load_frame_decodes_bgr(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    arr = np.zeros((8, 6, 3), np.uint8)
    arr[..., 0] = 200  # blue channel in BGR
    ok, enc = cv2.imencode(".png", arr)
    assert ok
    path = tmp_path / "a b.png"
    path.write_bytes(enc.tobytes())
    out = data.load_frame(path)
    assert out.shape == (8, 6, 3) and out[0, 0, 0] == 200 and out[0, 0, 2] == 0


def test_data_modules_import_no_heavy_dependencies():
    import subprocess

    code = (
        "import sys, vidan.data, vidan.layout, vidan.cli;"
        "print([m for m in ('torch','onnxruntime','cv2','supervision') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# --------------------------------------------------------------------------- real data


@pytest.mark.integration
def test_real_data_matches_committed_manifest():
    if not Path("data/raw").is_dir() or not Path("data/manifest.json").is_file():
        pytest.skip("data/raw not fetched (run `vidan fetch`)")
    assert data.verify(Path("data/raw"), Path("data/manifest.json")) == []
    manifest = json.loads(Path("data/manifest.json").read_text())
    assert set(manifest["sequences"]) == set(data.SEQUENCES)
