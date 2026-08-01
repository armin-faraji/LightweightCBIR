from __future__ import annotations

import unittest

import torch
from torch import nn

from cbir.backbone import FrozenDinoV2Extractor
from cbir.config import BackboneConfig


class _PatchEmbed:
    patch_size = 14


class _FakeDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.num_register_tokens = 4
        self.embed_dim = 384
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.anchor = nn.Parameter(torch.zeros(()))

    def get_intermediate_layers(self, batch, *, n, reshape, return_class_token, norm):
        del reshape, return_class_token, norm
        patch_count = (batch.shape[-2] // 14) * (batch.shape[-1] // 14)
        return tuple(
            (
                torch.full((batch.shape[0], patch_count, 384), float(index)),
                torch.full((batch.shape[0], 384), float(index)),
            )
            for index in n
        )


class BackboneTokenTests(unittest.TestCase):
    def test_rectangular_intermediate_tokens_exclude_registers(self) -> None:
        extractor = FrozenDinoV2Extractor(
            BackboneConfig(device="cpu"),
            model=_FakeDino(),
        )
        output = extractor.extract_intermediate_tokens(
            torch.randn(2, 3, 224, 140),
            (3, 7, 11),
        )
        self.assertEqual(len(output), 3)
        self.assertEqual(output[0].patch_grid_hw, (16, 10))
        self.assertEqual(tuple(output[0].patches.shape), (2, 160, 384))
        self.assertEqual(tuple(output[0].cls.shape), (2, 384))


if __name__ == "__main__":
    unittest.main()

