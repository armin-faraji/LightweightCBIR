from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from cbir.cache import (
    CacheResolver,
    FeatureManifest,
    FeatureShardReader,
    FeatureShardWriter,
    validate_feature_cache,
)
from cbir.config import FeatureCacheConfig
from cbir.data.sfm import ImageRecord
from cbir.data.transforms import PreprocessRecord
from cbir.features import AllLayerFeatures
from cbir.utils import atomic_write_json, hash_strings


class CacheTests(unittest.TestCase):
    @staticmethod
    def _manifest(ids: tuple[str, ...]) -> FeatureManifest:
        return FeatureManifest(
            fingerprint="f" * 64,
            dataset_name="test",
            source_ids_hash=hash_strings(ids),
            extraction_config={"test": True},
            layer_indices=(0, 1),
            token_dim=4,
            feature_dtype="float16",
            entropy_dtype="float32",
            expected_image_ids=ids,
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
                PreprocessRecord(image_id, (28, 28), (28, 28), (28, 28), (2, 2), False)
                for image_id in ids
            ),
        )

    @staticmethod
    def _records(ids: tuple[str, ...]) -> dict[str, ImageRecord]:
        return {
            image_id: ImageRecord(image_id, split="train", cluster_id=index)
            for index, image_id in enumerate(ids)
        }

    def test_cache_round_trip(self) -> None:
        ids = ("a", "b", "c")
        features = self._features(ids)
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            manifest = self._manifest(ids)
            writer = FeatureShardWriter(root, manifest, FeatureCacheConfig(local_root=root))
            writer.write_shard(features, self._records(ids))
            writer.finalize()
            self.assertTrue(validate_feature_cache(root)["valid"])
            reader = FeatureShardReader(root)
            fetched = reader.fetch(("c", "a"), layer_indices=(1,))
            self.assertEqual(tuple(fetched["cls"].shape), (2, 1, 4))
            self.assertTrue(
                torch.allclose(
                    fetched["cls"][0, 0],
                    features.cls[2, 1].float(),
                    atol=1e-3,
                    rtol=1e-3,
                )
            )

    def test_partial_cache_is_valid_only_with_explicit_opt_in(self) -> None:
        ids = ("a", "b", "c")
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            writer = FeatureShardWriter(
                root,
                self._manifest(ids),
                FeatureCacheConfig(local_root=root),
            )
            writer.write_shard(self._features(("a",)), self._records(("a",)))

            partial = validate_feature_cache(root, require_complete=False)
            self.assertTrue(partial["valid"])
            self.assertEqual(partial["completed_image_count"], 1)
            self.assertEqual(partial["expected_image_count"], 3)
            self.assertFalse(partial["is_complete"])
            self.assertFalse(validate_feature_cache(root)["valid"])
            with self.assertRaises(ValueError):
                FeatureShardReader(root)

    def test_partial_drive_cache_restores_resumes_and_publishes_report(self) -> None:
        ids = ("a", "b", "c")
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            first_local = root / "first_runtime"
            drive = root / "drive"
            manifest = self._manifest(ids)
            config = FeatureCacheConfig(local_root=first_local)
            writer = FeatureShardWriter(first_local, manifest, config)
            writer.write_shard(self._features(("a", "b")), self._records(("a", "b")))
            first_resolver = CacheResolver(first_local, drive)
            first_resolver.mirror_local_to_drive(require_complete=False, incremental=True)

            self.assertTrue(validate_feature_cache(drive, require_complete=False)["valid"])
            self.assertFalse(validate_feature_cache(drive)["valid"])

            # A completed shard upload may survive an interruption before its
            # manifest commit.  It must not be copied into a fresh runtime.
            shutil.copyfile(
                drive / "shards" / "shard_00000.pt",
                drive / "shards" / "shard_00001.pt",
            )
            second_local = root / "second_runtime"
            second_resolver = CacheResolver(second_local, drive)
            self.assertIsNone(second_resolver.resolve_existing(manifest.fingerprint))
            self.assertFalse(second_local.exists())
            self.assertEqual(
                second_resolver.resolve_existing(
                    manifest.fingerprint,
                    allow_incomplete=True,
                ),
                second_local,
            )
            self.assertFalse((second_local / "shards" / "shard_00001.pt").exists())
            self.assertTrue(validate_feature_cache(second_local, require_complete=False)["valid"])

            resumed_writer = FeatureShardWriter(
                second_local,
                manifest,
                FeatureCacheConfig(local_root=second_local),
            )
            self.assertEqual(resumed_writer.completed_ids, frozenset(("a", "b")))
            resumed_writer.write_shard(self._features(("c",)), self._records(("c",)))
            resumed_writer.finalize()
            report = validate_feature_cache(second_local, expected_fingerprint=manifest.fingerprint)
            atomic_write_json(second_local / "extraction_report.json", report)
            second_resolver.mirror_local_to_drive(extra_files=("extraction_report.json",))

            drive_report = validate_feature_cache(drive, expected_fingerprint=manifest.fingerprint)
            self.assertTrue(drive_report["valid"])
            self.assertTrue((drive / "extraction_report.json").is_file())
            self.assertEqual(
                (drive / "extraction_report.json").read_text(encoding="utf-8"),
                (second_local / "extraction_report.json").read_text(encoding="utf-8"),
            )
            third_local = root / "third_runtime"
            third_resolver = CacheResolver(third_local, drive)
            self.assertEqual(third_resolver.resolve_existing(manifest.fingerprint), third_local)
            self.assertTrue(validate_feature_cache(third_local)["valid"])
            self.assertTrue((third_local / "extraction_report.json").is_file())

    def test_valid_drive_cache_quarantines_an_invalid_local_cache_before_restore(self) -> None:
        ids = ("a", "b")
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            drive = root / "drive"
            manifest = self._manifest(ids)
            writer = FeatureShardWriter(
                drive,
                manifest,
                FeatureCacheConfig(local_root=drive),
            )
            writer.write_shard(self._features(ids), self._records(ids))
            writer.finalize()

            local = root / "local"
            local.mkdir()
            (local / "interrupted.tmp").write_text("incomplete", encoding="utf-8")
            resolver = CacheResolver(local, drive)
            with self.assertWarnsRegex(RuntimeWarning, "quarantined an invalid local"):
                restored = resolver.resolve_existing(manifest.fingerprint)

            self.assertEqual(restored, local)
            self.assertTrue(validate_feature_cache(local)["valid"])
            quarantined = list(root.glob(".local.invalid-*"))
            self.assertEqual(len(quarantined), 1)
            self.assertTrue((quarantined[0] / "interrupted.tmp").is_file())


if __name__ == "__main__":
    unittest.main()
