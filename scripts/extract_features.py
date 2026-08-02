#!/usr/bin/env python3
"""Build or resume a frozen all-layer aggregate cache for retrieval-SfM-30k."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

from cbir.backbone import FrozenDinoV2Extractor
from cbir.cache import (
    CacheResolver,
    FeatureManifest,
    FeatureShardWriter,
    cache_dir_name,
    validate_feature_cache,
)
from cbir.config import config_to_dict, extraction_fingerprint, load_project_config
from cbir.data.sfm import (
    ImageRecord,
    Sfm30kMetadata,
    SfmImageDirectoryReader,
    SfmMatImageReader,
)
from cbir.features import FeatureExtractionRunner
from cbir.utils import atomic_write_json, hash_strings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/extraction_sfm30k.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Extract only first N IDs for a pilot.")
    parser.add_argument("--backbone-batch-size", type=int, default=32)
    parser.add_argument("--no-drive-mirror", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    metadata = Sfm30kMetadata.from_official_files(
        config.sfm.metadata_path,
        _require_path(config.sfm.names_clusters_path, "names_clusters_path"),
    )
    records = (*metadata.records("train"), *metadata.records("val"))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    expected_ids = tuple(record.image_id for record in records)
    source_ids_hash = hash_strings(expected_ids)
    fingerprint = extraction_fingerprint(
        backbone=config.backbone,
        preprocess=config.preprocess,
        pooling=config.pooling,
        source_ids_hash=source_ids_hash,
    )
    cache_name = cache_dir_name(
        "sfm30k" if args.limit is None else f"sfm30k-pilot{len(records)}",
        config.backbone.entrypoint.replace("_", "-"),
        fingerprint,
    )
    local_dir = config.cache.local_root / cache_name
    drive_dir = (
        None
        if config.cache.drive_root is None or args.no_drive_mirror
        else config.cache.drive_root / cache_name
    )
    resolver = CacheResolver(local_dir, drive_dir)
    # Extraction is the sole consumer allowed to accept an incomplete cache.
    # This restores a validated Drive prefix after a fresh Colab runtime, while
    # readers/training still require the COMPLETE marker by default.
    existing = resolver.resolve_existing(fingerprint, allow_incomplete=True)
    if existing is not None:
        complete = validate_feature_cache(existing, expected_fingerprint=fingerprint)
        if complete["valid"]:
            # A session can end after local finalization but before report/Drive
            # publication.  Repair that harmless state on the next invocation
            # instead of treating the cache as finished too early.
            atomic_write_json(existing / "extraction_report.json", complete)
            resolver.mirror_local_to_drive(extra_files=("extraction_report.json",))
            print(f"Using valid existing cache: {existing}")
            return
        partial = validate_feature_cache(
            existing,
            expected_fingerprint=fingerprint,
            require_complete=False,
        )
        print(
            f"Resuming validated cache: {existing} "
            f"({partial['completed_image_count']}/{partial['expected_image_count']} images)."
        )

    manifest = FeatureManifest(
        fingerprint=fingerprint,
        dataset_name="retrieval-SfM-30k",
        source_ids_hash=source_ids_hash,
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
    writer = FeatureShardWriter(local_dir, manifest, config.cache)
    records_by_id = {record.image_id: record for record in records}
    pending = [record for record in records if record.image_id not in writer.completed_ids]
    print(
        f"Cache {local_dir}: {len(writer.completed_ids)}/{len(records)} complete; "
        f"extracting {len(pending)} images."
    )

    if pending:
        extractor = FrozenDinoV2Extractor(config.backbone)
        runner = FeatureExtractionRunner(extractor, config.preprocess, config.pooling)
        reader_context = _make_image_reader(config)
        with reader_context as image_reader:
            for start in range(0, len(pending), config.cache.shard_size):
                shard_records = pending[start : start + config.cache.shard_size]
                images = [
                    (record.image_id, _read_sfm_image(image_reader, record))
                    for record in shard_records
                ]
                features = runner.extract_images(
                    images,
                    layer_indices=config.pooling.all_layer_indices,
                    backbone_batch_size=args.backbone_batch_size,
                )
                writer.write_shard(
                    features,
                    {record.image_id: records_by_id[record.image_id] for record in shard_records},
                )
                # Publish each committed local shard.  The resolver uploads and
                # validates the shard before atomically advancing Drive's manifest.
                # A runtime interruption can therefore lose at most the current
                # local-only shard, never invalidate previously published progress.
                resolver.mirror_local_to_drive(require_complete=False, incremental=True)
                print(f"Wrote shard ending at {start + len(shard_records)}/{len(pending)}")

    writer.finalize()
    report = validate_feature_cache(local_dir, expected_fingerprint=fingerprint)
    atomic_write_json(local_dir / "extraction_report.json", report)
    # Publish the report before COMPLETE is copied, so a Drive cache accepted as
    # final always carries the cache-validation evidence used to create it.
    resolver.mirror_local_to_drive(extra_files=("extraction_report.json",))
    print(f"Feature cache complete and valid: {local_dir}")


def _make_image_reader(config: object) -> object:
    # Keep the two supported sources explicit. The original archive layout is
    # the most robust fallback if a MAT release layout changes.
    sfm = config.sfm
    if sfm.image_root is not None:
        return nullcontext(SfmImageDirectoryReader(sfm.image_root))
    if sfm.image_mat_path is not None:
        return SfmMatImageReader(sfm.image_mat_path)
    raise ValueError("configure either sfm.image_root or sfm.image_mat_path")


def _read_sfm_image(reader: object, record: ImageRecord):
    if isinstance(reader, SfmImageDirectoryReader):
        return reader.read(record.image_id)
    if isinstance(reader, SfmMatImageReader):
        if record.split is None or record.image_locator is None:
            raise ValueError(f"MAT reader needs split/locator for {record.image_id}")
        return reader.read(record.split, int(record.image_locator))
    raise TypeError(f"unsupported SfM image reader {type(reader)!r}")


def _require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"config.sfm.{name} is required")
    return value


if __name__ == "__main__":
    main()
