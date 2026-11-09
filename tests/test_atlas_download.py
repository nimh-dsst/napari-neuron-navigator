"""Atlas transport, cache publication, and native certificate verification."""

from __future__ import annotations

import io
import json
import ssl
import tarfile
import urllib.error
import urllib.request

import numpy as np
import pytest
import tifffile
from brainglobe_atlasapi.core import Atlas

from napari_neuron_navigator import atlas_download as downloads

NAME = "allen_mouse_50um"
FULL_NAME = f"{NAME}_v1.2"


def _archive(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


@pytest.fixture
def atlas_files():
    image = io.BytesIO()
    tifffile.imwrite(image, np.ones((2, 2, 2), dtype=np.uint16))
    return {
        f"{FULL_NAME}/metadata.json": json.dumps(
            {
                "name": NAME,
                "version": "1.2",
                "resolution": [50, 50, 50],
                "shape": [2, 2, 2],
                "orientation": "asr",
                "symmetric": True,
                "additional_references": [],
            }
        ).encode(),
        f"{FULL_NAME}/structures.json": json.dumps(
            [
                {
                    "id": 1,
                    "acronym": "root",
                    "name": "root",
                    "structure_id_path": [1],
                    "rgb_triplet": [255, 255, 255],
                }
            ]
        ).encode(),
        f"{FULL_NAME}/reference.tiff": image.getvalue(),
        f"{FULL_NAME}/annotation.tiff": image.getvalue(),
    }


class _Response(io.BytesIO):
    def __init__(self, content, *, length=None):
        super().__init__(content)
        self.headers = {
            "Content-Length": str(len(content) if length is None else length)
        }


class _Opener:
    def __init__(self, archive, *, length=None):
        self.archive = archive
        self.length = length
        self.urls = []

    def open(self, request, timeout):
        assert timeout > 0
        self.urls.append(request.full_url)
        if request.full_url.endswith("last_versions.conf"):
            return _Response(f"[atlases]\n{NAME} = 1.2\n".encode())
        assert request.full_url.endswith(f"/{FULL_NAME}.tar.gz")
        return _Response(self.archive, length=self.length)


def test_native_certificate_verification_is_scoped_to_opener():
    original_context = ssl.SSLContext
    original_urlopen = urllib.request.urlopen
    opener = downloads._make_opener()
    handler = next(
        h for h in opener.handlers if isinstance(h, urllib.request.HTTPSHandler)
    )
    assert handler._context.verify_mode == ssl.CERT_REQUIRED
    assert handler._context.check_hostname
    assert ssl.SSLContext is original_context
    assert urllib.request.urlopen is original_urlopen


# tifffile 2026.2 assigns ndarray.shape, deprecated by NumPy 2.5.
@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:DeprecationWarning:tifffile.tifffile"
)
@pytest.mark.parametrize("length", [None, 0])
def test_download_installs_loadable_atlas_and_reports_progress(
    monkeypatch, tmp_path, atlas_files, length
):
    archive = _archive(atlas_files)
    opener = _Opener(archive, length=length)
    monkeypatch.setattr(downloads, "_make_opener", lambda: opener)
    updates = []
    cache = tmp_path / "cache"
    temporary = tmp_path / "downloads"
    result = downloads.download_brainglobe_atlas(
        NAME,
        brainglobe_dir=cache,
        interm_download_dir=temporary,
        fn_update=lambda completed, total: updates.append((completed, total)),
    )
    assert result == cache / FULL_NAME
    atlas = Atlas(result)
    assert atlas.resolution == (50, 50, 50)
    assert len(atlas.structures) == 1
    np.testing.assert_array_equal(atlas.annotation, np.ones((2, 2, 2)))
    assert updates[-1] == (len(archive), len(archive))
    assert len(opener.urls) == 2
    assert list(temporary.iterdir()) == []
    assert list(cache.iterdir()) == [result]


@pytest.mark.parametrize(
    "reason",
    [
        ssl.SSLCertVerificationError("certificate issuer is not trusted"),
        OSError("Network is unreachable"),
    ],
)
def test_connection_error_preserves_real_cause(reason):
    class FailingOpener:
        def open(self, *args, **kwargs):
            raise urllib.error.URLError(reason)

    with pytest.raises(RuntimeError) as caught:
        downloads._open(FailingOpener(), "https://gin.g-node.org/")
    assert str(reason) in str(caught.value)
    assert "gin.g-node.org" in str(caught.value)
    assert "server is down" not in str(caught.value)
    assert isinstance(caught.value.__cause__, urllib.error.URLError)


def test_http_error_reports_status():
    class FailingOpener:
        def open(self, *args, **kwargs):
            raise urllib.error.HTTPError(
                "https://gin.g-node.org/", 503, "Unavailable", {}, None
            )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        downloads._open(FailingOpener(), "https://gin.g-node.org/")


@pytest.mark.parametrize(
    "failure", ["truncated", "bad_metadata", "wrong_root", "traversal"]
)
def test_failed_download_leaves_no_partial_cache(
    monkeypatch, tmp_path, atlas_files, failure
):
    length = None
    if failure == "bad_metadata":
        atlas_files[f"{FULL_NAME}/metadata.json"] = b"not JSON"
    elif failure == "wrong_root":
        atlas_files["different_atlas/metadata.json"] = b"{}"
    elif failure == "traversal":
        atlas_files[f"{FULL_NAME}/../../escaped.txt"] = b"outside"
    archive = _archive(atlas_files)
    if failure == "truncated":
        length = len(archive) + 1
    monkeypatch.setattr(
        downloads, "_make_opener", lambda: _Opener(archive, length=length)
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    sentinel = cache / "existing_atlas"
    sentinel.write_text("keep this")
    temporary = tmp_path / "downloads"
    with pytest.raises((ValueError, OSError)):
        downloads.download_brainglobe_atlas(
            NAME, brainglobe_dir=cache, interm_download_dir=temporary
        )
    assert list(cache.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep this"
    assert list(temporary.iterdir()) == []
    assert not (tmp_path / "escaped.txt").exists()


def test_archive_rejects_symlinks(tmp_path):
    archive_path = tmp_path / "atlas.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(f"{FULL_NAME}/metadata.json")
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/outside"
        archive.addfile(member)
    with pytest.raises(ValueError, match="Unsupported atlas archive entry"):
        downloads._extract_archive(archive_path, tmp_path / "stage", FULL_NAME)
