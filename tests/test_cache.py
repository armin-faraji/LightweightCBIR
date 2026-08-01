from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cbir.cache import FeatureManifest, FeatureShardReader, FeatureShardWriter, validate_feature_cache
from cbir.config import FeatureCacheConfig
from cbir.data.sfm import ImageRecord
from cbir.data.transforms import PreprocessRecord
from cbir.features import AllLayerFeatures
from cbir.utils import hash_strings


class CacheTests(unittest.TestCase):
    def test_cache_round_trip(self) -> None:
        ids = ("a", "b", "c")
        features = AllLayerFeatures(
            image_ids=ids,
            cls=torch.randn(3, 2, 4),
            mean_patch=torch.randn(3, 2, 4),
            cls_guided_patch=torch.randn(3, 2, 4),
            pooling_entropy=torch.rand(3, 2),
            layer_indices=(0, 1),
            preprocess_records=tuple(
                PreprocessRecord(image_id, (28, 28), (28, 28), (28, 28), (2, 2), False)
                for image_id in ids
            ),
        )
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            manifest = FeatureManifest(
                fingerprint="f" * 64,
                dataset_name="test",
                source_ids_hash=hash_strings(ids),
                extraction_config={},
                layer_indices=(0, 1),
                token_dim=4,
                feature_dtype="float16",
                entropy_dtype="float32",
                expected_image_ids=ids,
            )
            writer = FeatureShardWriter(root, manifest, FeatureCacheConfig(local_root=root))
            writer.write_shard(
                features,
                {
                    image_id: ImageRecord(image_id, split="train", cluster_id=index)
                    for index, image_id in enumerate(ids)
                },
            )
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


if __name__ == "__main__":
    unittest.main()
