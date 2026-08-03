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

    def test_static_gate_base_logits_receive_gradient(self) -> None:
        torch.manual_seed(7)
        head = ReliabilityGatedFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
            gate_mode="static",
        )
        target = torch.randn(5, 6)
        descriptor = head(self.cls, self.local, self.entropy)
        self.assertIsInstance(descriptor, torch.Tensor)
        loss = (descriptor * target).sum()
        loss.backward()
        self.assertIsNotNone(head.base_logits.grad)
        self.assertGreater(float(head.base_logits.grad.abs().sum()), 1e-8)

    def test_reliability_gate_parameters_receive_gradient(self) -> None:
        torch.manual_seed(11)
        head = ReliabilityGatedFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
            gate_mode="reliability",
        )
        entropy = torch.tensor(
            [
                [0.05, 0.40, 0.95],
                [0.90, 0.25, 0.10],
                [0.20, 0.70, 0.60],
                [0.80, 0.15, 0.50],
                [0.55, 0.85, 0.30],
            ]
        )
        target = torch.randn(5, 6)
        descriptor = head(self.cls, self.local, entropy)
        self.assertIsInstance(descriptor, torch.Tensor)
        loss = (descriptor * target).sum()
        loss.backward()
        self.assertIsNotNone(head.base_logits.grad)
        self.assertIsNotNone(head.gate_raw.grad)
        self.assertGreater(float(head.base_logits.grad.abs().sum()), 1e-8)
        self.assertGreater(float(head.gate_raw.grad.abs().sum()), 1e-8)

    def test_reliability_weights_change_when_image_entropies_change(self) -> None:
        head = ReliabilityGatedFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
            gate_mode="reliability",
        )
        entropy = torch.tensor(
            [
                [0.05, 0.50, 0.95],
                [0.95, 0.50, 0.05],
                [0.20, 0.40, 0.60],
                [0.60, 0.40, 0.20],
                [0.10, 0.90, 0.30],
            ]
        )
        _, diagnostics = head(self.cls, self.local, entropy, return_diagnostics=True)
        self.assertGreater(
            float(diagnostics.layer_weights.detach().std(dim=0, unbiased=False).max()),
            1e-8,
        )


if __name__ == "__main__":
    unittest.main()
