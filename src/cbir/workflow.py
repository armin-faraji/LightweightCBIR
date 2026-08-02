"""Small reusable workflow helpers for the SfM cache lifecycle.

The command-line extractor and independent cloud notebooks must derive the
same cache location.  Keeping that derivation here removes fragile manual
``REPLACE_WITH_COMPLETED_CACHE_FOLDER`` paths from notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import CacheResolver, cache_dir_name
from .config import ProjectConfig, extraction_fingerprint
from .data.sfm import Sfm30kMetadata
from .utils import hash_strings


@dataclass(frozen=True)
class SfmCacheLocation:
    """Deterministic locations and identity of a full retrieval-SfM-30k cache."""

    fingerprint: str
    cache_name: str
    local_dir: Path
    drive_dir: Path | None
    source_ids_hash: str


def full_sfm_cache_location(
    config: ProjectConfig,
    metadata: Sfm30kMetadata | None = None,
) -> SfmCacheLocation:
    """Derive the full-cache identity used by ``extract_features.py``."""
    if metadata is None:
        if config.sfm.names_clusters_path is None:
            raise ValueError("config.sfm.names_clusters_path is required")
        metadata = Sfm30kMetadata.from_official_files(
            config.sfm.metadata_path,
            config.sfm.names_clusters_path,
        )
    records = (*metadata.records("train"), *metadata.records("val"))
    source_ids_hash = hash_strings(tuple(record.image_id for record in records))
    fingerprint = extraction_fingerprint(
        backbone=config.backbone,
        preprocess=config.preprocess,
        pooling=config.pooling,
        source_ids_hash=source_ids_hash,
    )
    cache_name = cache_dir_name(
        "sfm30k",
        config.backbone.entrypoint.replace("_", "-"),
        fingerprint,
    )
    return SfmCacheLocation(
        fingerprint=fingerprint,
        cache_name=cache_name,
        local_dir=config.cache.local_root / cache_name,
        drive_dir=None
        if config.cache.drive_root is None
        else config.cache.drive_root / cache_name,
        source_ids_hash=source_ids_hash,
    )


def restore_complete_sfm_cache(
    config: ProjectConfig,
    metadata: Sfm30kMetadata | None = None,
) -> SfmCacheLocation:
    """Ensure the deterministic full cache is locally available and validated.

    A valid local cache wins.  Otherwise a compatible complete Drive mirror is
    copied to fast runtime storage through ``CacheResolver``.  The returned
    location always points to the usable local directory.
    """
    location = full_sfm_cache_location(config, metadata)
    resolver = CacheResolver(location.local_dir, location.drive_dir)
    restored = resolver.resolve_existing(location.fingerprint)
    if restored is None:
        raise FileNotFoundError(
            "no compatible completed SfM cache exists locally or in persistent storage: "
            f"{location.cache_name}"
        )
    return location
