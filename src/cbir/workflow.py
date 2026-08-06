"""The small local workflow shared by the SfM cache notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backbone import FrozenDinoV2Extractor
from .cache import (
    MANIFEST_NAME,
    REVISITOP_FEATURES_NAME,
    FeatureManifest,
    FeatureShardWriter,
    save_revisitop_feature_cache,
    validate_feature_cache,
    validate_revisitop_feature_cache,
)
from .config import ProjectConfig, config_to_dict, extraction_fingerprint
from .data.download import SfmUrls, download_with_resume
from .data.revisitop import RevisitOPDataset
from .data.sfm import Sfm30kMetadata, SfmMatImageReader
from .features import FeatureExtractionRunner, extract_image_stream
from .utils import hash_strings, stable_hash


@dataclass(frozen=True)
class SfmCacheLocation:
    cache_dir: Path
    fingerprint: str
    source_ids_hash: str


@dataclass(frozen=True)
class RevisitOPCacheLocation:
    cache_dir: Path
    fingerprint: str


def ensure_sfm30k_sources(
    config: ProjectConfig,
    *,
    urls: SfmUrls | None = None,
) -> None:
    """Use local SfM files, downloading only a missing official source."""
    source_urls = urls or SfmUrls()
    for path, url in (
        (config.sfm.metadata_path, source_urls.metadata_pickle),
        (config.sfm.names_clusters_path, source_urls.names_clusters_mat),
        (config.sfm.image_mat_path, source_urls.image_mat),
    ):
        if not path.is_file():
            download_with_resume(url, path)


def full_sfm_cache_location(
    config: ProjectConfig,
    metadata: Sfm30kMetadata | None = None,
) -> SfmCacheLocation:
    """Derive the fixed local cache location and its JSON fingerprint."""
    metadata = metadata or Sfm30kMetadata.from_official_files(
        config.sfm.metadata_path,
        config.sfm.names_clusters_path,
    )
    records = (*metadata.records("train"), *metadata.records("val"))
    source_ids_hash = hash_strings(record.image_id for record in records)
    fingerprint = extraction_fingerprint(
        backbone=config.backbone,
        preprocess=config.preprocess,
        pooling=config.pooling,
        source_ids_hash=source_ids_hash,
    )
    return SfmCacheLocation(
        cache_dir=config.cache.root / "sfm30k",
        fingerprint=fingerprint,
        source_ids_hash=source_ids_hash,
    )


def build_sfm_feature_cache(
    config: ProjectConfig,
    *,
    metadata: Sfm30kMetadata | None = None,
    backbone_batch_size: int = 8,
    image_chunk_size: int = 128,
) -> SfmCacheLocation:
    """Resume or build 1,000-image SfM shards with bounded decoded-image RAM."""
    if backbone_batch_size <= 0:
        raise ValueError("backbone_batch_size must be positive")
    if image_chunk_size <= 0:
        raise ValueError("image_chunk_size must be positive")
    if metadata is None:
        ensure_sfm30k_sources(config)
        metadata = Sfm30kMetadata.from_official_files(
            config.sfm.metadata_path,
            config.sfm.names_clusters_path,
        )
    location = full_sfm_cache_location(config, metadata)
    existing = validate_feature_cache(
        location.cache_dir,
        expected_fingerprint=location.fingerprint,
    )
    if existing["valid"]:
        return location
    manifest_path = location.cache_dir / "manifest.json"
    if manifest_path.exists() and not validate_feature_cache(
        location.cache_dir,
        expected_fingerprint=location.fingerprint,
        require_complete=False,
    )["valid"]:
        raise ValueError(
            f"SfM cache is incompatible or damaged at {location.cache_dir}. Delete it "
            "explicitly before rebuilding it."
        )

    ensure_sfm30k_sources(config)

    records = (*metadata.records("train"), *metadata.records("val"))
    expected_ids = tuple(record.image_id for record in records)
    manifest = FeatureManifest(
        fingerprint=location.fingerprint,
        dataset_name="retrieval-SfM-30k",
        source_ids_hash=location.source_ids_hash,
        extraction_config={
            "backbone": config_to_dict(config.backbone),
            "preprocess": config_to_dict(config.preprocess),
            "pooling": config_to_dict(config.pooling),
        },
        layer_indices=config.pooling.all_layer_indices,
        token_dim=config.backbone.token_dim,
        feature_dtype=config.cache.feature_dtype,
        entropy_dtype=config.cache.entropy_dtype,
        expected_image_ids=expected_ids,
    )
    writer = FeatureShardWriter(location.cache_dir, manifest, config.cache)
    pending = [record for record in records if record.image_id not in writer.completed_ids]
    if not pending:
        writer.finalize()
        return location

    extractor = FrozenDinoV2Extractor(config.backbone)
    runner = FeatureExtractionRunner(extractor, config.preprocess, config.pooling)
    with SfmMatImageReader(config.sfm.image_mat_path) as images:
        for start in range(0, len(pending), config.cache.shard_size):
            shard_records = pending[start : start + config.cache.shard_size]
            if any(record.split is None or record.image_locator is None for record in shard_records):
                raise ValueError("SfM MAT extraction requires split and image locator")
            features = extract_image_stream(
                runner,
                (
                    (
                        record.image_id,
                        images.read(record.split, int(record.image_locator)),
                    )
                    for record in shard_records
                ),
                layer_indices=config.pooling.all_layer_indices,
                backbone_batch_size=backbone_batch_size,
                image_chunk_size=image_chunk_size,
            )
            writer.write_shard(
                features,
                {record.image_id: record for record in shard_records},
            )
    writer.finalize()
    return location


def revisitop_feature_cache_location(
    config: ProjectConfig,
    dataset: RevisitOPDataset,
    *,
    layer_indices: tuple[int, ...],
) -> RevisitOPCacheLocation:
    """Derive a cache identity that includes query crops and selected layers."""
    fingerprint = stable_hash(
        {
            "dataset": dataset.name,
            "database_ids": list(dataset.database_ids),
            "queries": [
                {
                    "query_id": query.query_id,
                    "source_image_id": query.source_image_id,
                    "bbox_xyxy": list(query.bbox_xyxy),
                }
                for query in dataset.queries
            ],
            "layer_indices": list(layer_indices),
            "backbone": config_to_dict(config.backbone),
            "preprocess": config_to_dict(config.preprocess),
            "pooling": config_to_dict(config.pooling),
        }
    )
    return RevisitOPCacheLocation(
        cache_dir=config.cache.root / "revisitop" / dataset.name,
        fingerprint=fingerprint,
    )


def build_revisitop_feature_cache(
    config: ProjectConfig,
    dataset: RevisitOPDataset,
    *,
    layer_indices: tuple[int, ...],
    backbone_batch_size: int = 8,
    image_chunk_size: int = 256,
    runner: FeatureExtractionRunner | None = None,
) -> RevisitOPCacheLocation:
    """Build a compact local database/query feature bundle for one benchmark."""
    location = revisitop_feature_cache_location(
        config,
        dataset,
        layer_indices=layer_indices,
    )
    existing = validate_revisitop_feature_cache(
        location.cache_dir,
        expected_fingerprint=location.fingerprint,
    )
    if existing["valid"]:
        return location
    # This bundle is small and derived only from preserved raw archives, unlike
    # the large SfM cache. Replacing an incompatible bundle is therefore safe.
    for name in (MANIFEST_NAME, REVISITOP_FEATURES_NAME):
        (location.cache_dir / name).unlink(missing_ok=True)

    active_runner = runner or FeatureExtractionRunner(
        FrozenDinoV2Extractor(config.backbone), config.preprocess, config.pooling
    )
    database = extract_image_stream(
        active_runner,
        ((image_id, dataset.read_database_image(image_id)) for image_id in dataset.database_ids),
        layer_indices=layer_indices,
        backbone_batch_size=backbone_batch_size,
        image_chunk_size=image_chunk_size,
    )
    queries = extract_image_stream(
        active_runner,
        ((query.query_id, dataset.crop_query(query)) for query in dataset.queries),
        layer_indices=layer_indices,
        backbone_batch_size=backbone_batch_size,
        image_chunk_size=image_chunk_size,
    )
    save_revisitop_feature_cache(
        location.cache_dir,
        fingerprint=location.fingerprint,
        extraction_config={
            "backbone": config_to_dict(config.backbone),
            "preprocess": config_to_dict(config.preprocess),
            "pooling": config_to_dict(config.pooling),
            "layer_indices": list(layer_indices),
        },
        database=database,
        queries=queries,
        config=config.cache,
    )
    return location
