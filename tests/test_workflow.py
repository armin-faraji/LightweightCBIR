from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from cbir.cache import FeatureManifest, FeatureShardReader, FeatureShardWriter, cache_dir_name
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
from cbir.data.transforms import PreprocessRecord
from cbir.features import AllLayerFeatures
from cbir.utils import hash_strings
from cbir.workflow import full_sfm_cache_location, restore_complete_sfm_cache


class SfmCacheWorkflowTests(unittest.TestCase):
    @staticmethod
    def _metadata() -> Sfm30kMetadata:
        # Deliberately insert the validation image first.  The cache identity
        # must still use extractor order: all train records, then validation.
        images = {
            "val-image": ImageRecord("val-image", split="val", cluster_id=2),
            "train-second": ImageRecord("train-second", split="train", cluster_id=1),
            "train-first": ImageRecord("train-first", split="train", cluster_id=1),
        }
        return Sfm30kMetadata(images=images, train_pairs=(), val_pairs=())

    @staticmethod
    def _config(root: Path) -> ProjectConfig:
        return ProjectConfig(
            backbone=BackboneConfig(
                entrypoint="tiny-backbone",
                token_dim=4,
                num_blocks=2,
                patch_size=14,
                num_register_tokens=0,
                device="cpu",
            ),
            preprocess=PreprocessConfig(long_side=28, patch_size=14),
            pooling=PoolingConfig(all_layer_indices=(0, 1)),
            cache=FeatureCacheConfig(
                local_root=root / "runtime-cache",
                drive_root=root / "drive-cache",
                shard_size=3,
            ),
            sfm=SfmConfig(
                metadata_path=root / "metadata.pkl",
                names_clusters_path=root / "selection.mat",
                image_mat_path=root / "images.mat",
            ),
        )

    @staticmethod
    def _features(ids: tuple[str, ...]) -> AllLayerFeatures:
        return AllLayerFeatures(
            image_ids=ids,
            cls=torch.randn(len(ids), 2, 4),
            mean_patch=torch.randn(len(ids), 2, 4),
            cls_guided_patch=torch.randn(len(ids), 2, 4),
            pooling_entropy=torch.rand(len(ids), 2),
            layer_indices=(0, 1),
            preprocess_records=tuple(
                PreprocessRecord(
                    image_id=image_id,
                    original_hw=(28, 28),
                    resized_hw=(28, 28),
                    final_hw=(28, 28),
                    patch_grid_hw=(2, 2),
                    extreme_aspect_crop=False,
                )
                for image_id in ids
            ),
        )

    def test_location_matches_the_extractor_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            metadata = self._metadata()
            location = full_sfm_cache_location(config, metadata)
            expected_ids = ("train-second", "train-first", "val-image")
            expected_source_hash = hash_strings(expected_ids)
            expected_fingerprint = extraction_fingerprint(
                backbone=config.backbone,
                preprocess=config.preprocess,
                pooling=config.pooling,
                source_ids_hash=expected_source_hash,
            )

            self.assertEqual(location.source_ids_hash, expected_source_hash)
            self.assertEqual(location.fingerprint, expected_fingerprint)
            self.assertEqual(
                location.cache_name,
                cache_dir_name("sfm30k", "tiny-backbone", expected_fingerprint),
            )
            self.assertEqual(location.local_dir, config.cache.local_root / location.cache_name)
            self.assertEqual(location.drive_dir, config.cache.drive_root / location.cache_name)

    def test_restore_completed_drive_cache_to_a_fresh_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            metadata = self._metadata()
            location = full_sfm_cache_location(config, metadata)
            assert location.drive_dir is not None
            expected_ids = ("train-second", "train-first", "val-image")
            manifest = FeatureManifest(
                fingerprint=location.fingerprint,
                dataset_name="retrieval-SfM-30k",
                source_ids_hash=location.source_ids_hash,
                extraction_config={"test": True},
                layer_indices=(0, 1),
                token_dim=4,
                feature_dtype="float16",
                entropy_dtype="float32",
                expected_image_ids=expected_ids,
            )
            writer = FeatureShardWriter(location.drive_dir, manifest, config.cache)
            records = {image_id: metadata.images[image_id] for image_id in expected_ids}
            writer.write_shard(self._features(expected_ids), records)
            writer.finalize()

            restored = restore_complete_sfm_cache(config, metadata)
            self.assertEqual(restored, location)
            self.assertTrue(restored.local_dir.is_dir())
            reader = FeatureShardReader(restored.local_dir)
            self.assertEqual(reader.image_ids, expected_ids)
            self.assertEqual(tuple(reader.fetch(("val-image",))["cls"].shape), (1, 2, 4))

    def test_descriptor_dimension_changes_training_not_cache_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_config = self._config(root)
            metadata = self._metadata()
            fusion_256 = FusionConfig(token_dim=4, layer_indices=(0, 1), output_dim=256)
            fusion_128 = replace(fusion_256, output_dim=128)
            config_256 = replace(base_config, fusion=fusion_256)
            config_128 = replace(base_config, fusion=fusion_128)
            training = TrainingConfig(batch_size=2, epochs=1, device="cpu")

            self.assertEqual(
                full_sfm_cache_location(config_256, metadata).fingerprint,
                full_sfm_cache_location(config_128, metadata).fingerprint,
            )
            self.assertNotEqual(
                train_fingerprint(
                    cache_fingerprint="cache-fingerprint",
                    fusion=fusion_256,
                    training=training,
                ),
                train_fingerprint(
                    cache_fingerprint="cache-fingerprint",
                    fusion=fusion_128,
                    training=training,
                ),
            )


if __name__ == "__main__":
    unittest.main()
