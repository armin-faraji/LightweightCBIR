from __future__ import annotations

import unittest

import torch

from cbir.config import FusionConfig
from cbir.fusion import (
    FinalClsProjectionHead,
    MultiLevelClsConcatHead,
    MultiLevelGlobalLocalFusion,
    build_descriptor_head,
)


class FusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cls = torch.randn(5, 3, 8)
        self.local = torch.randn(5, 3, 8)
        self.entropy = torch.rand(5, 3)

    def test_global_local_descriptor_norm_and_weight_sum(self) -> None:
        head = MultiLevelGlobalLocalFusion(
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
        assert diagnostics.layer_weights is not None
        self.assertTrue(
            torch.allclose(
                diagnostics.layer_weights.sum(dim=1),
                torch.ones(5),
                atol=1e-6,
            )
        )
        assert diagnostics.entropy_penalty_scale is not None
        self.assertGreaterEqual(float(diagnostics.entropy_penalty_scale.detach()), 0.0)

    def test_uniform_layer_weighting_ignores_entropy(self) -> None:
        torch.manual_seed(1)
        head = MultiLevelGlobalLocalFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
            gate_mode="uniform",
        )
        _, first = head(self.cls, self.local, torch.zeros_like(self.entropy), return_diagnostics=True)
        _, second = head(self.cls, self.local, torch.ones_like(self.entropy), return_diagnostics=True)
        assert first.layer_weights is not None and second.layer_weights is not None
        self.assertTrue(torch.allclose(first.layer_weights, second.layer_weights))
        self.assertTrue(torch.allclose(first.layer_weights, torch.full((5, 3), 1 / 3)))

    def test_static_layer_weighting_base_logits_receive_gradient(self) -> None:
        torch.manual_seed(7)
        head = MultiLevelGlobalLocalFusion(
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

    def test_dynamic_layer_weighting_parameters_receive_gradient(self) -> None:
        torch.manual_seed(11)
        head = MultiLevelGlobalLocalFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
            gate_mode="dynamic",
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

    def test_dynamic_layer_weights_change_when_image_entropies_change(self) -> None:
        head = MultiLevelGlobalLocalFusion(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
            gate_mode="dynamic",
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
        assert diagnostics.layer_weights is not None
        self.assertGreater(
            float(diagnostics.layer_weights.detach().std(dim=0, unbiased=False).max()),
            1e-8,
        )

    def test_cls_concat_head_is_normalized_and_uses_only_cls(self) -> None:
        head = MultiLevelClsConcatHead(
            token_dim=8,
            layer_indices=(0, 1, 2),
            output_dim=6,
        )
        descriptor, diagnostics = head(self.cls, return_diagnostics=True)
        self.assertEqual(descriptor.shape, (5, 6))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(descriptor, dim=1),
                torch.ones(5),
                atol=1e-5,
            )
        )
        self.assertIsNone(diagnostics.layer_weights)
        self.assertEqual(sum(parameter.numel() for parameter in head.parameters()), 198)

    def test_final_cls_projection_is_normalized_and_requires_one_layer(self) -> None:
        head = FinalClsProjectionHead(
            token_dim=8,
            layer_indices=(2,),
            output_dim=6,
        )
        descriptor = head(self.cls[:, -1:].clone())
        self.assertIsInstance(descriptor, torch.Tensor)
        self.assertEqual(descriptor.shape, (5, 6))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(descriptor, dim=1),
                torch.ones(5),
                atol=1e-5,
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly one layer"):
            FinalClsProjectionHead(token_dim=8, layer_indices=(1, 2), output_dim=6)

    def test_head_factory_uses_head_kind(self) -> None:
        configs = (
            FusionConfig(
                token_dim=8,
                layer_indices=(0, 1, 2),
                output_dim=6,
                head_kind="global_local",
                gate_mode="dynamic",
            ),
            FusionConfig(
                token_dim=8,
                layer_indices=(0, 1, 2),
                output_dim=6,
                head_kind="cls_concat",
                gate_mode=None,
            ),
            FusionConfig(
                token_dim=8,
                layer_indices=(2,),
                output_dim=6,
                head_kind="final_cls_projection",
                gate_mode=None,
            ),
        )
        expected = (
            MultiLevelGlobalLocalFusion,
            MultiLevelClsConcatHead,
            FinalClsProjectionHead,
        )
        for config, expected_type in zip(configs, expected, strict=True):
            with self.subTest(head_kind=config.head_kind):
                self.assertIsInstance(build_descriptor_head(config), expected_type)


if __name__ == "__main__":
    unittest.main()
