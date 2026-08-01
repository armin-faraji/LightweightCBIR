from __future__ import annotations

import unittest

import torch

from cbir.fusion import ReliabilityGatedFusion


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cls = torch.randn(5, 3, 8)
        self.local = torch.randn(5, 3, 8)
        self.entropy = torch.rand(5, 3)

    def test_descriptor_norm_and_weight_sum(self) -> None:
        head = ReliabilityGatedFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
        )
        descriptor, diagnostics = head(
            self.cls,
            self.local,
            self.entropy,
            return_diagnostics=True,
        )
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(descriptor, dim=1),
                torch.ones(5),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                diagnostics.layer_weights.sum(dim=1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        self.assertGreaterEqual(float(diagnostics.lambda_value.detach()), 0.0)

    def test_uniform_gate_ignores_entropy(self) -> None:
        torch.manual_seed(1)
        head = ReliabilityGatedFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
            gate_mode="uniform",
        )
        _, first = head(self.cls, self.local, torch.zeros_like(self.entropy), return_diagnostics=True)
        _, second = head(self.cls, self.local, torch.ones_like(self.entropy), return_diagnostics=True)
        self.assertTrue(torch.allclose(first.layer_weights, second.layer_weights))
        self.assertTrue(torch.allclose(first.layer_weights, torch.full((5, 3), 1 / 3)))


if __name__ == "__main__":
    unittest.main()
