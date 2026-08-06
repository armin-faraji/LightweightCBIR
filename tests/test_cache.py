from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cbir.cache import (
    FeatureManifest,
    FeatureShardReader,
    FeatureShardWriter,
    load_revisitop_feature_cache,
    save_revisitop_feature_cache,
    validate_feature_cache,
)
from cbir.config import FeatureCacheConfig
from cbir.data.sfm import ImageRecord
from cbir.data.transforms import PreprocessRecord
from cbir.features import AllLayerFeatures
from cbir.utils import hash_strings


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

    def test_sharded_cache_round_trip_and_preload(self) -> None:
        ids = ("a", "b", "c")
        features = self._features(ids)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = FeatureShardWriter(root, self._manifest(ids), FeatureCacheConfig(root=root))
            writer.write_shard(features, self._records(ids))
            writer.finalize()

            report = validate_feature_cache(root, expected_fingerprint="f" * 64)
            self.assertTrue(report["valid"], report["errors"])
            self.assertTrue(report["manifest"]["complete"])
            self.assertNotIn("sha256", report["manifest"]["shards"][0])

            reader = FeatureShardReader(root, preload=True)
            self.assertEqual(len(reader._loaded), 1)
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

    def test_partial_cache_resumes_from_manifest(self) -> None:
        ids = ("a", "b", "c")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = FeatureCacheConfig(root=root)
            writer = FeatureShardWriter(root, self._manifest(ids), config)
            writer.write_shard(self._features(("a",)), self._records(("a",)))
            self.assertTrue(validate_feature_cache(root, require_complete=False)["valid"])
            self.assertFalse(validate_feature_cache(root)["valid"])

            resumed = FeatureShardWriter(root, self._manifest(ids), config)
            self.assertEqual(resumed.completed_ids, frozenset(("a",)))
            resumed.write_shard(self._features(("b", "c")), self._records(("b", "c")))
            resumed.finalize()
            self.assertTrue(validate_feature_cache(root)["valid"])

    def test_revisitop_bundle_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = FeatureCacheConfig(root=root)
            database = self._features(("database_a", "database_b"))
            queries = self._features(("query_a",))
            save_revisitop_feature_cache(
                root,
                fingerprint="bundle-fingerprint",
                extraction_config={"pooling": {"temperature": 0.025}},
                database=database,
                queries=queries,
                config=config,
            )
            bundle = load_revisitop_feature_cache(root, expected_fingerprint="bundle-fingerprint")
            self.assertEqual(tuple(bundle["database"]["image_ids"]), database.image_ids)
            self.assertEqual(tuple(bundle["queries"]["image_ids"]), queries.image_ids)
            self.assertEqual(tuple(bundle["database"]["cls"].shape), (2, 2, 4))


if __name__ == "__main__":
    unittest.main()
