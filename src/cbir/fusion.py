"""Small trainable descriptor heads for cached multi-level DINOv2 features."""

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
    """Optional diagnostics emitted by a descriptor head."""

    layer_weights: torch.Tensor | None = None
    layer_vectors: torch.Tensor | None = None
    entropy_penalty_scale: torch.Tensor | None = None
    base_logits: torch.Tensor | None = None


class MultiLevelGlobalLocalFusion(nn.Module):
    """Fuse per-layer global and local features with configurable layer weighting."""

    def __init__(
        self,
        token_dim: int = 384,
        layer_indices: tuple[int, ...] = (3, 7, 11),
        output_dim: int = 256,
        local_kind: Literal["cls_guided_patch", "mean_patch"] = "cls_guided_patch",
        gate_mode: Literal["uniform", "static", "dynamic"] = "dynamic",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.config = FusionConfig(
            token_dim=token_dim,
            layer_indices=layer_indices,
            output_dim=output_dim,
            head_kind="global_local",
            local_kind=local_kind,
            gate_mode=gate_mode,
            eps=eps,
        )
        self.layer_norms = nn.ModuleList(
            nn.LayerNorm(2 * token_dim) for _ in self.config.layer_indices
        )
        self.projectors = nn.ModuleList(
            nn.Linear(2 * token_dim, output_dim) for _ in self.config.layer_indices
        )
        self.base_logits = nn.Parameter(torch.zeros(len(self.config.layer_indices)))
        # softplus(-2) is nonzero but initially modest.
        self.gate_raw = nn.Parameter(torch.tensor(-2.0))

    @classmethod
    def from_config(cls, config: FusionConfig) -> "MultiLevelGlobalLocalFusion":
        if config.head_kind != "global_local":
            raise ValueError("MultiLevelGlobalLocalFusion requires head_kind=global_local")
        if config.gate_mode not in {"uniform", "static", "dynamic"}:
            raise ValueError("global_local configuration has invalid layer weighting")
        return cls(
            token_dim=config.token_dim,
            layer_indices=config.layer_indices,
            output_dim=config.output_dim,
            local_kind=config.local_kind,
            gate_mode=config.gate_mode,
            eps=config.eps,
        )

    @property
    def gate_mode(self) -> Literal["uniform", "static", "dynamic"]:
        assert self.config.gate_mode in {"uniform", "static", "dynamic"}
        return self.config.gate_mode

    def forward(
        self,
        cls: torch.Tensor,
        local: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, FusionDiagnostics]:
        expected_layers = len(self.config.layer_indices)
        expected_shape = (cls.shape[0], expected_layers, self.config.token_dim)
        if cls.ndim != 3 or local is None or entropy is None:
            raise ValueError("global_local fusion expects cls/local [B,K,D] and entropy [B,K]")
        if local.ndim != 3 or entropy.ndim != 2:
            raise ValueError("global_local fusion expects cls/local [B,K,D] and entropy [B,K]")
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
            entropy_penalty_scale = torch.zeros(
                (), dtype=entropy.dtype, device=entropy.device
            )
        elif self.gate_mode == "static":
            logits = self.base_logits.unsqueeze(0).expand_as(entropy)
            entropy_penalty_scale = torch.zeros(
                (), dtype=entropy.dtype, device=entropy.device
            )
        else:
            entropy_penalty_scale = functional.softplus(self.gate_raw)
            logits = self.base_logits.unsqueeze(0) - entropy_penalty_scale * entropy
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
            raise RuntimeError("layer weights do not sum to one")
        if not return_diagnostics:
            return descriptor
        return descriptor, FusionDiagnostics(
            layer_weights=weights,
            layer_vectors=projected,
            entropy_penalty_scale=entropy_penalty_scale,
            base_logits=self.base_logits,
        )


class MultiLevelClsConcatHead(nn.Module):
    """Project concatenated CLS tokens from several selected transformer blocks."""

    def __init__(
        self,
        token_dim: int = 384,
        layer_indices: tuple[int, ...] = (3, 7, 11),
        output_dim: int = 256,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.config = FusionConfig(
            token_dim=token_dim,
            layer_indices=layer_indices,
            output_dim=output_dim,
            head_kind="cls_concat",
            gate_mode=None,
            eps=eps,
        )
        self.layer_norm = nn.LayerNorm(len(layer_indices) * token_dim)
        self.projector = nn.Linear(len(layer_indices) * token_dim, output_dim)

    @classmethod
    def from_config(cls, config: FusionConfig) -> "MultiLevelClsConcatHead":
        if config.head_kind != "cls_concat":
            raise ValueError("MultiLevelClsConcatHead requires head_kind=cls_concat")
        return cls(
            token_dim=config.token_dim,
            layer_indices=config.layer_indices,
            output_dim=config.output_dim,
            eps=config.eps,
        )

    def forward(
        self,
        cls: torch.Tensor,
        local: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, FusionDiagnostics]:
        _ = local, entropy
        expected_layers = len(self.config.layer_indices)
        expected_shape = (cls.shape[0], expected_layers, self.config.token_dim)
        if cls.ndim != 3 or cls.shape != expected_shape:
            raise ValueError(
                f"expected CLS shape [B,{expected_layers},{self.config.token_dim}]"
            )
        assert_finite("CLS concatenation input", cls)
        descriptor = l2_normalize(
            self.projector(self.layer_norm(cls.flatten(start_dim=1))),
            eps=self.config.eps,
        )
        assert_finite("CLS concatenation descriptor", descriptor)
        if not return_diagnostics:
            return descriptor
        return descriptor, FusionDiagnostics()


class FinalClsProjectionHead(nn.Module):
    """Project one selected final-layer CLS token into the descriptor space."""

    def __init__(
        self,
        token_dim: int = 384,
        layer_indices: tuple[int, ...] = (11,),
        output_dim: int = 256,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.config = FusionConfig(
            token_dim=token_dim,
            layer_indices=layer_indices,
            output_dim=output_dim,
            head_kind="final_cls_projection",
            gate_mode=None,
            eps=eps,
        )
        self.layer_norm = nn.LayerNorm(token_dim)
        self.projector = nn.Linear(token_dim, output_dim)

    @classmethod
    def from_config(cls, config: FusionConfig) -> "FinalClsProjectionHead":
        if config.head_kind != "final_cls_projection":
            raise ValueError(
                "FinalClsProjectionHead requires head_kind=final_cls_projection"
            )
        return cls(
            token_dim=config.token_dim,
            layer_indices=config.layer_indices,
            output_dim=config.output_dim,
            eps=config.eps,
        )

    def forward(
        self,
        cls: torch.Tensor,
        local: torch.Tensor | None = None,
        entropy: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, FusionDiagnostics]:
        _ = local, entropy
        expected_shape = (cls.shape[0], 1, self.config.token_dim)
        if cls.ndim != 3 or cls.shape != expected_shape:
            raise ValueError(
                f"expected CLS shape [B,1,{self.config.token_dim}] for final projection"
            )
        assert_finite("final CLS projection input", cls)
        descriptor = l2_normalize(
            self.projector(self.layer_norm(cls[:, 0])),
            eps=self.config.eps,
        )
        assert_finite("final CLS projection descriptor", descriptor)
        if not return_diagnostics:
            return descriptor
        return descriptor, FusionDiagnostics()


def build_descriptor_head(config: FusionConfig) -> nn.Module:
    """Construct the configured compact descriptor head."""
    if config.head_kind == "global_local":
        return MultiLevelGlobalLocalFusion.from_config(config)
    if config.head_kind == "cls_concat":
        return MultiLevelClsConcatHead.from_config(config)
    if config.head_kind == "final_cls_projection":
        return FinalClsProjectionHead.from_config(config)
    raise AssertionError(f"unsupported descriptor head: {config.head_kind}")
