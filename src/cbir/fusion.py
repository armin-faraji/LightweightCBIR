"""The trainable reliability-gated multi-level global-local fusion head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import nn

from .config import FusionConfig
from .utils import assert_finite, l2_normalize


@dataclass(frozen=True)
class FusionDiagnostics:
    layer_weights: torch.Tensor
    layer_vectors: torch.Tensor
    lambda_value: torch.Tensor
    base_logits: torch.Tensor


class ReliabilityGatedFusion(nn.Module):
    """Small RGMF descriptor head with uniform/static/reliability gate modes."""

    def __init__(
        self,
        token_dim: int = 384,
        layer_indices: tuple[int, ...] = (3, 7, 11),
        output_dim: int = 256,
        local_kind: Literal["cls_guided_patch", "mean_patch"] = "cls_guided_patch",
        gate_mode: Literal["uniform", "static", "reliability"] = "reliability",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        config = FusionConfig(
            token_dim=token_dim,
            layer_indices=layer_indices,
            output_dim=output_dim,
            local_kind=local_kind,
            gate_mode=gate_mode,
            eps=eps,
        )
        self.config = config
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(2 * token_dim) for _ in config.layer_indices
        )
        self.projectors = nn.ModuleList(
            nn.Linear(2 * token_dim, output_dim) for _ in config.layer_indices
        )
        self.base_logits = nn.Parameter(torch.zeros(len(config.layer_indices)))
        # softplus(-2) ~= 0.127: nonzero but initially modest entropy penalty.
        self.gate_raw = nn.Parameter(torch.tensor(-2.0))

    @classmethod
    def from_config(cls, config: FusionConfig) -> "ReliabilityGatedFusion":
        return cls(
            token_dim=config.token_dim,
            layer_indices=config.layer_indices,
            output_dim=config.output_dim,
            local_kind=config.local_kind,
            gate_mode=config.gate_mode,
            eps=config.eps,
        )

    @property
    def gate_mode(self) -> str:
        return self.config.gate_mode

    def forward(
        self,
        cls: torch.Tensor,
        local: torch.Tensor,
        entropy: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, FusionDiagnostics]:
        expected_layers = len(self.config.layer_indices)
        expected_shape = (cls.shape[0], expected_layers, self.config.token_dim)
        if cls.ndim != 3 or local.ndim != 3 or entropy.ndim != 2:
            raise ValueError("expected cls/local [B,K,D] and entropy [B,K]")
        if cls.shape != expected_shape or local.shape != expected_shape:
            raise ValueError(
                f"expected cls/local shape [B,{expected_layers},{self.config.token_dim}]"
            )
        if entropy.shape != cls.shape[:2]:
            raise ValueError("entropy must have shape [B,K]")
        assert_finite("fusion CLS input", cls)
        assert_finite("fusion local input", local)
        assert_finite("fusion entropy input", entropy)

        layer_vectors = []
        for position, (norm, projector) in enumerate(
            zip(self.layer_norms, self.projectors, strict=True)
        ):
            merged = torch.cat([cls[:, position], local[:, position]], dim=-1)
            layer_vectors.append(l2_normalize(projector(norm(merged)), eps=self.config.eps))
        projected = torch.stack(layer_vectors, dim=1)

        if self.gate_mode == "uniform":
            logits = torch.zeros_like(entropy)
            lambda_value = torch.zeros((), dtype=entropy.dtype, device=entropy.device)
        elif self.gate_mode == "static":
            logits = self.base_logits.unsqueeze(0).expand_as(entropy)
            lambda_value = torch.zeros((), dtype=entropy.dtype, device=entropy.device)
        elif self.gate_mode == "reliability":
            lambda_value = functional.softplus(self.gate_raw)
            logits = self.base_logits.unsqueeze(0) - lambda_value * entropy
        else:
            raise AssertionError(f"unknown gate mode {self.gate_mode}")
        weights = torch.softmax(logits, dim=1)
        descriptor = l2_normalize(
            torch.sum(weights.unsqueeze(-1) * projected, dim=1),
            eps=self.config.eps,
        )
        assert_finite("fusion descriptor", descriptor)
        if not torch.allclose(
            weights.sum(dim=1),
            torch.ones_like(weights[:, 0]),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError("fusion weights do not sum to one")
        if not return_diagnostics:
            return descriptor
        return descriptor, FusionDiagnostics(
            layer_weights=weights,
            layer_vectors=projected,
            lambda_value=lambda_value,
            base_logits=self.base_logits,
        )
