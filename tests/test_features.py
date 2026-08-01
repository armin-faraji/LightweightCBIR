from __future__ import annotations

import unittest

import torch

from cbir.features import cls_guided_pool


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


if __name__ == "__main__":
    unittest.main()

