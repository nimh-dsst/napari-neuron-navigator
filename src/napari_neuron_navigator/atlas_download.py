"""Download official BrainGlobe archives using the system certificate store.

BrainGlobe 2.x uses Requests' CA bundle and replaces certificate errors with
"GIN server is down". Keep native TLS verification local to these requests;
do not modify SSL or Requests globally in the napari process.
"""

from __future__ import annotations

import configparser
import re
import shutil
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath


def _make_opener():
    from truststore import SSLContext

    context = SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))


def _open(opener, url: str):
    request = urllib.request.Request(
        url, headers={"User-Agent": "napari-neuron-navigator-atlas-downloader"}
    )
    try:
        return opener.open(request, timeout=30)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Atlas request failed: HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        failure = (
            "HTTPS certificate verification failed"
            if isinstance(exc.reason, ssl.SSLCertVerificationError)
            else "connection failed"
        )
        raise RuntimeError(f"Atlas {failure} for {url}: {exc.reason}") from exc


def _extract_archive(archive_path: Path, stage: Path, full_name: str) -> Path:
    """Extract regular files within the requested atlas directory only."""
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not parts:
                continue
            destination = stage.joinpath(*parts)
            if (
                parts[0] != full_name
                or ".." in parts
                or "\\" in member.name
                or not destination.resolve().is_relative_to(stage.resolve())
            ):
                raise ValueError(f"Unexpected atlas archive path: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.extractfile(member) as source,
                    destination.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target)
            else:
                raise ValueError(f"Unsupported atlas archive entry: {member.name}")

    atlas_dir = stage / full_name
    for name in (
        "metadata.json",
        "structures.json",
        "reference.tiff",
        "annotation.tiff",
    ):
        if not (atlas_dir / name).is_file():
            raise ValueError(f"Atlas archive is missing {full_name}/{name}")
    return atlas_dir


def download_brainglobe_atlas(
    atlas_name: str,
    *,
    brainglobe_dir: Path,
    interm_download_dir: Path,
    fn_update=None,
) -> Path:
    """Install an official archive atomically and return its cache directory."""
    from brainglobe_atlasapi.core import Atlas
    from brainglobe_atlasapi.descriptors import remote_url_base

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", atlas_name):
        raise ValueError(f"Invalid atlas name: {atlas_name}")

    opener = _make_opener()
    with _open(opener, remote_url_base.format("last_versions.conf")) as response:
        versions = configparser.ConfigParser()
        versions.read_string(response.read().decode("utf-8"))
    version = versions.get("atlases", atlas_name, fallback="")
    if not re.fullmatch(r"\d+\.\d+", version):
        raise ValueError(f"No valid published version found for atlas {atlas_name}")

    full_name = f"{atlas_name}_v{version}"
    archive_url = remote_url_base.format(f"{full_name}.tar.gz")
    brainglobe_dir.mkdir(parents=True, exist_ok=True)
    interm_download_dir.mkdir(parents=True, exist_ok=True)

    with (
        tempfile.TemporaryDirectory(
            prefix=".atlas-download-", dir=interm_download_dir
        ) as download_tmp,
        tempfile.TemporaryDirectory(
            prefix=".atlas-install-", dir=brainglobe_dir
        ) as install_tmp,
    ):
        archive_path = Path(download_tmp) / "atlas.tar.gz"
        with _open(opener, archive_url) as response, archive_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            completed = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                completed += len(chunk)
                if fn_update is not None:
                    fn_update(completed, total)
        if not completed or (total and completed != total):
            raise OSError(
                f"Incomplete atlas download from {archive_url}: {completed}/{total} bytes"
            )
        if fn_update is not None and not total:
            fn_update(completed, completed)

        staged = _extract_archive(archive_path, Path(install_tmp), full_name)
        # Validate metadata and the structure tree before publishing a cache.
        Atlas(staged)
        destination = brainglobe_dir / full_name
        if destination.exists():
            raise FileExistsError(
                f"Atlas cache appeared during download: {destination}"
            )
        staged.rename(destination)
    return destination
