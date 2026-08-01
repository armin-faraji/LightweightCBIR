"""Narrow adapter around the official frozen DINOv2 Torch Hub model."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

from .config import BackboneConfig
from .utils import assert_finite


@dataclass(frozen=True)
class LayerTokens:
    """One DINO block's CLS and genuine spatial patch tokens."""

    block_index: int
    cls: torch.Tensor
    patches: torch.Tensor
    patch_grid_hw: tuple[int, int]


class FrozenDinoV2Extractor:
    """Load DINOv2 once and expose validated intermediate-token extraction."""

    def __init__(
        self,
        config: BackboneConfig,
        *,
        model: nn.Module | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "BackboneConfig requests CUDA, but CUDA is unavailable. "
                "Set device='cpu' only for smoke tests."
            )
        self.model = model if model is not None else self._load_model()
        self.model = self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._validate_model_contract()

    def _load_model(self) -> nn.Module:
        repo = self.config.repo
        if self.config.revision:
            repo = f"{repo}:{self.config.revision}"
        return torch.hub.load(
            repo,
            self.config.entrypoint,
            trust_repo=True,
        )

    def _validate_model_contract(self) -> None:
        patch_embed = getattr(self.model, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        if patch_size is not None and int(patch_size) != self.config.patch_size:
            raise ValueError(
                f"expected patch size {self.config.patch_size}, got {patch_size}"
            )
        num_registers = getattr(self.model, "num_register_tokens", None)
        if num_registers is not None and int(num_registers) != self.config.num_register_tokens:
            raise ValueError(
                "configured register-token count does not match model: "
                f"{self.config.num_register_tokens} != {num_registers}"
            )
        embed_dim = getattr(self.model, "embed_dim", None)
        if embed_dim is not None and int(embed_dim) != self.config.token_dim:
            raise ValueError(
                f"expected token width {self.config.token_dim}, got {embed_dim}"
            )
        blocks = getattr(self.model, "blocks", None)
        if blocks is not None and len(blocks) != self.config.num_blocks:
            raise ValueError(
                f"expected {self.config.num_blocks} blocks, got {len(blocks)}"
            )
        if not hasattr(self.model, "get_intermediate_layers"):
            raise TypeError(
                "the supplied model does not implement DINOv2 "
                "get_intermediate_layers(); do not use a different backend "
                "without an explicit adapter"
            )

    @property
    def token_dim(self) -> int:
        return self.config.token_dim

    def extract_intermediate_tokens(
        self,
        batch: torch.Tensor,
        layer_indices: Sequence[int] | Iterable[int],
    ) -> tuple[LayerTokens, ...]:
        """Return selected normalized DINO block outputs without register tokens."""
        indices = tuple(int(index) for index in layer_indices)
        if not indices:
            raise ValueError("layer_indices must not be empty")
        if len(set(indices)) != len(indices):
            raise ValueError("layer_indices must not contain duplicates")
        if indices != tuple(sorted(indices)):
            raise ValueError("layer_indices must be ascending for deterministic layer alignment")
        if min(indices) < 0 or max(indices) >= self.config.num_blocks:
            raise ValueError(
                f"layer_indices must be in [0, {self.config.num_blocks - 1}]"
            )
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError("batch must have shape [B, 3, H, W]")
        height, width = int(batch.shape[-2]), int(batch.shape[-1])
        if height % self.config.patch_size or width % self.config.patch_size:
            raise ValueError(
                "DINOv2 input height and width must be divisible by patch size "
                f"{self.config.patch_size}, got {(height, width)}"
            )

        batch = batch.to(self.device, non_blocking=True)
        amp_context = (
            torch.autocast(device_type=self.device.type, enabled=True)
            if self.config.use_amp and self.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            raw_outputs = self.model.get_intermediate_layers(
                batch,
                n=list(indices),
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        if len(raw_outputs) != len(indices):
            raise RuntimeError(
                f"model returned {len(raw_outputs)} layers for {len(indices)} requests"
            )
        patch_grid_hw = (
            height // self.config.patch_size,
            width // self.config.patch_size,
        )
        expected_patches = patch_grid_hw[0] * patch_grid_hw[1]
        outputs: list[LayerTokens] = []
        for block_index, raw_output in zip(indices, raw_outputs, strict=True):
            if not isinstance(raw_output, tuple) or len(raw_output) != 2:
                raise TypeError(
                    "expected DINOv2 get_intermediate_layers(..., "
                    "return_class_token=True) to return (patches, cls)"
                )
            patches, cls = raw_output
            if cls.ndim != 2 or patches.ndim != 3:
                raise ValueError("unexpected DINOv2 intermediate tensor ranks")
            if cls.shape[0] != batch.shape[0] or patches.shape[0] != batch.shape[0]:
                raise ValueError("intermediate tensors have inconsistent batch size")
            if cls.shape[-1] != self.token_dim or patches.shape[-1] != self.token_dim:
                raise ValueError("intermediate tensor width does not match configuration")
            if patches.shape[1] != expected_patches:
                raise ValueError(
                    "DINOv2 patch output has unexpected count. This adapter "
                    "expects official get_intermediate_layers to exclude the "
                    "CLS and register tokens: "
                    f"expected {expected_patches}, got {patches.shape[1]}"
                )
            assert_finite("DINOv2 CLS tokens", cls)
            assert_finite("DINOv2 patch tokens", patches)
            outputs.append(
                LayerTokens(
                    block_index=block_index,
                    cls=cls.float(),
                    patches=patches.float(),
                    patch_grid_hw=patch_grid_hw,
                )
            )
        return tuple(outputs)
