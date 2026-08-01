"""Small safe/resumable download helpers for publicly hosted dataset artifacts."""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .sfm import cid_to_relative_path


@dataclass(frozen=True)
class SfmUrls:
    metadata_pickle: str = (
        "https://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/"
        "retrieval-SfM-120k.pkl"
    )
    names_clusters_mat: str = (
        "https://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/"
        "retrieval-SfM-30k-imagenames-clusterids.mat"
    )
    image_mat: str = (
        "https://cmp.felk.cvut.cz/cnnimageretrieval/data/train/dbs/"
        "retrieval-SfM-30k.mat"
    )
    original_images_archive: str = (
        "https://cmp.felk.cvut.cz/cnnimageretrieval/data/train/ims/ims.tar.gz"
    )


def download_with_resume(
    url: str,
    destination: Path,
    *,
    chunk_size: int = 1024 * 1024,
    timeout_seconds: int = 60,
) -> Path:
    """Resume a single HTTP download using a sidecar partial file when possible."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if start:
        request.add_header("Range", f"bytes={start}-")
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except Exception as error:
        raise RuntimeError(f"failed to download {url}: {error}") from error
    status = getattr(response, "status", response.getcode())
    append = start > 0 and status == 206
    if start > 0 and not append:
        # Server did not honor Range. Restart only the known partial file.
        partial.unlink(missing_ok=True)
    mode = "ab" if append else "wb"
    try:
        with response, partial.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception:
        # Preserve .part for a later resume attempt.
        raise
    partial.replace(destination)
    return destination


def extract_selected_sfm_images(
    archive_path: Path,
    destination_root: Path,
    image_ids: Iterable[str],
) -> tuple[Path, ...]:
    """Safely extract only selected CIDs from the official 120k tar archive."""
    destination_root = Path(destination_root)
    expected = {cid_to_relative_path(image_id).as_posix() for image_id in image_ids}
    expected_by_name = {Path(relative).name: relative for relative in expected}
    extracted: list[Path] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_path = Path(member.name)
            match = expected_by_name.get(member_path.name)
            if match is None:
                continue
            target = destination_root / match
            if target.is_file():
                extracted.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member {member.name}")
            temporary = target.with_name(f".{target.name}.part")
            try:
                with source, temporary.open("wb") as handle:
                    while chunk := source.read(1024 * 1024):
                        handle.write(chunk)
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            extracted.append(target)
    missing = expected - {path.relative_to(destination_root).as_posix() for path in extracted}
    if missing:
        raise RuntimeError(
            f"archive extraction found {len(extracted)} of {len(expected)} requested images; "
            f"first missing paths: {sorted(missing)[:5]}"
        )
    return tuple(extracted)
