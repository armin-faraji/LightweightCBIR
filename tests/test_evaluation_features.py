from __future__ import annotations

import unittest

import torch

from cbir.evaluation import descriptors_from_feature_tensors
from cbir.fusion import FinalClsProjectionHead


class EvaluationFeatureTests(unittest.TestCase):
    def test_in_memory_feature_descriptors_are_normalized(self) -> None:
        head = FinalClsProjectionHead(token_dim=4, layer_indices=(0,), output_dim=3)
        descriptors = descriptors_from_feature_tensors(
            torch.randn(5, 1, 4),
            head,
            batch_size=2,
        )
        self.assertEqual(tuple(descriptors.shape), (5, 3))
        self.assertTrue(torch.allclose(descriptors.norm(dim=1), torch.ones(5), atol=1e-6))
