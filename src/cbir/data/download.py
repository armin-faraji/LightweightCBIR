"""Small safe/resumable download helpers for publicly hosted dataset artifacts."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path


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
