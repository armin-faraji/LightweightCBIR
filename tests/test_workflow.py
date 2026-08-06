from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cbir.cache import FeatureManifest, FeatureShardWriter
from cbir.config import (
    BackboneConfig,
    FeatureCacheConfig,
    FusionConfig,
    PoolingConfig,
    PreprocessConfig,
    ProjectConfig,
    SfmConfig,
    TrainingConfig,
    extraction_fingerprint,
    train_fingerprint,
)
from cbir.data.sfm import ImageRecord, Sfm30kMetadata
from cbir.utils import hash_strings
from cbir.workflow import build_sfm_feature_cache, full_sfm_cache_location


class SfmCacheWorkflowTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> Sfm30kMetadata:
        return Sfm30kMetadata(
            images={
                "val": ImageRecord("val", split="val", cluster_id=2),
                "train_b": ImageRecord("train_b", split="train", cluster_id=1),
                "train_a": ImageRecord("train_a", split="train", cluster_id=1),
            },
            train_pairs=(),
            val_pairs=(),
        )

    @staticmethod
    def _config(root: Path) -> ProjectConfig:
        return ProjectConfig(
            backbone=BackboneConfig(token_dim=4, num_blocks=2, device="cpu"),
            preprocess=PreprocessConfig(long_side=28, patch_size=14),
            pooling=PoolingConfig(all_layer_indices=(0, 1)),
            cache=FeatureCacheConfig(root=root / "cache", shard_size=3),
            sfm=SfmConfig(
                metadata_path=root / "metadata.pkl",
                names_clusters_path=root / "selection.mat",
                image_mat_path=root / "images.mat",
            ),
        )

    def test_location_uses_fixed_local_directory_and_extraction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            metadata = self._metadata()
            location = full_sfm_cache_location(config, metadata)
            expected_ids = tuple(
                record.image_id
                for record in (*metadata.records("train"), *metadata.records("val"))
            )
            source_ids_hash = hash_strings(expected_ids)
            self.assertEqual(location.cache_dir, config.cache.root / "sfm30k")
            self.assertEqual(location.source_ids_hash, source_ids_hash)
            self.assertEqual(
                location.fingerprint,
                extraction_fingerprint(
                    backbone=config.backbone,
                    preprocess=config.preprocess,
                    pooling=config.pooling,
                    source_ids_hash=source_ids_hash,
                ),
            )

    def test_mismatched_manifest_requires_explicit_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            metadata = self._metadata()
            location = full_sfm_cache_location(config, metadata)
            ids = tuple(
                record.image_id
                for record in (*metadata.records("train"), *metadata.records("val"))
            )
            wrong = FeatureManifest(
                fingerprint="wrong",
                dataset_name="retrieval-SfM-30k",
                source_ids_hash=hash_strings(ids),
                extraction_config={},
                layer_indices=(0, 1),
                token_dim=4,
                feature_dtype="float16",
                entropy_dtype="float32",
                expected_image_ids=ids,
            )
            FeatureShardWriter(location.cache_dir, wrong, config.cache)
            with patch("cbir.workflow.ensure_sfm30k_sources"):
                with self.assertRaisesRegex(ValueError, "SfM cache is incompatible or damaged"):
                    build_sfm_feature_cache(config, metadata=metadata)

    def test_descriptor_dimension_changes_training_not_cache_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            metadata = self._metadata()
            fusion_128 = FusionConfig(token_dim=4, layer_indices=(0, 1), output_dim=128)
            fusion_256 = replace(fusion_128, output_dim=256)
            training = TrainingConfig(batch_size=2, epochs=1, device="cpu")
            self.assertEqual(
                full_sfm_cache_location(replace(config, fusion=fusion_128), metadata).fingerprint,
                full_sfm_cache_location(replace(config, fusion=fusion_256), metadata).fingerprint,
            )
            self.assertNotEqual(
                train_fingerprint(
                    cache_fingerprint="cache",
                    fusion=fusion_128,
                    training=training,
                ),
                train_fingerprint(
                    cache_fingerprint="cache",
                    fusion=fusion_256,
                    training=training,
                ),
            )


if __name__ == "__main__":
    unittest.main()
