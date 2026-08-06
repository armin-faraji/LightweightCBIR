from __future__ import annotations

import unittest

import torch
from PIL import Image

from cbir.backbone import LayerTokens
from cbir.config import PoolingConfig, PreprocessConfig
from cbir.data.transforms import PreprocessRecord
from cbir.features import AllLayerFeatures, FeatureExtractionRunner, cls_guided_pool, extract_image_stream


class FeaturePoolingTests(unittest.TestCase):
    def test_weights_sum_to_one_and_entropy_is_bounded(self) -> None:
        cls = torch.tensor([[1.0, 0.0]])
        patches = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
        local, entropy, weights = cls_guided_pool(
            cls,
            patches,
            tau_p=0.1,
            return_weights=True,
        )
        self.assertEqual(tuple(local.shape), (1, 2))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(entropy), 0.0)
        self.assertLessEqual(float(entropy), 1.0)
        self.assertGreater(float(weights[0, 0]), 0.99)

    def test_single_patch_entropy_is_zero(self) -> None:
        cls = torch.randn(2, 4)
        patches = torch.randn(2, 1, 4)
        _, entropy, weights = cls_guided_pool(
            cls,
            patches,
            tau_p=0.2,
            return_weights=True,
        )
        self.assertTrue(torch.equal(entropy, torch.zeros_like(entropy)))
        self.assertTrue(torch.equal(weights, torch.ones_like(weights)))

    def test_stream_limits_decoded_image_chunks(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.chunk_sizes: list[int] = []

            def extract_images(self, images, *, layer_indices, backbone_batch_size):
                self.chunk_sizes.append(len(images))
                image_ids = tuple(image_id for image_id, _ in images)
                return AllLayerFeatures(
                    image_ids=image_ids,
                    cls=torch.zeros(len(images), 1, 2),
                    mean_patch=torch.zeros(len(images), 1, 2),
                    cls_guided_patch=torch.zeros(len(images), 1, 2),
                    pooling_entropy=torch.zeros(len(images), 1),
                    layer_indices=tuple(layer_indices),
                    preprocess_records=tuple(
                        PreprocessRecord(image_id, (14, 14), (14, 14), (14, 14), (1, 1), False)
                        for image_id in image_ids
                    ),
                )

        runner = FakeRunner()
        features = extract_image_stream(
            runner,
            ((str(index), Image.new("RGB", (1, 1))) for index in range(5)),
            layer_indices=(0,),
            backbone_batch_size=2,
            image_chunk_size=2,
        )
        self.assertEqual(runner.chunk_sizes, [2, 2, 1])
        self.assertEqual(features.image_ids, ("0", "1", "2", "3", "4"))

    def test_pooling_pilot_streams_decoded_images(self) -> None:
        class FakeExtractor:
            def extract_intermediate_tokens(self, batch, requested):
                return tuple(
                    LayerTokens(
                        block_index=index,
                        cls=torch.ones(batch.shape[0], 2),
                        patches=torch.ones(batch.shape[0], 1, 2),
                        patch_grid_hw=(1, 1),
                    )
                    for index in requested
                )

        runner = FeatureExtractionRunner(
            FakeExtractor(),  # type: ignore[arg-type]
            PreprocessConfig(long_side=14, patch_size=14),
            PoolingConfig(temperature=0.025, all_layer_indices=(0,)),
        )
        pilot = runner.pilot_pooling_temperatures(
            ((str(index), Image.new("RGB", (14, 14))) for index in range(5)),
            temperatures=(0.025,),
            layer_indices=(0,),
            backbone_batch_size=2,
            image_chunk_size=2,
        )
        self.assertEqual(pilot.image_ids, ("0", "1", "2", "3", "4"))
        self.assertEqual(tuple(pilot.entropy_by_temperature[0.025].shape), (5, 1))


if __name__ == "__main__":
    unittest.main()
