"""Parameter-free global/local feature aggregation from frozen DINO tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as functional
from PIL import Image

from .backbone import FrozenDinoV2Extractor, LayerTokens
from .config import PoolingConfig, PreprocessConfig
from .data.transforms import PreprocessRecord, preprocess_many_by_shape
from .utils import assert_finite


@dataclass(frozen=True)
class LayerAggregateBatch:
    """Aggregated feature tensors for a shape-homogeneous image batch."""

    cls: torch.Tensor
    mean_patch: torch.Tensor
    cls_guided_patch: torch.Tensor
    pooling_entropy: torch.Tensor
    layer_indices: tuple[int, ...]


@dataclass(frozen=True)
class AllLayerFeatures:
    """A canonical-ID-ordered set of all cached aggregate tensors."""

    image_ids: tuple[str, ...]
    cls: torch.Tensor
    mean_patch: torch.Tensor
    cls_guided_patch: torch.Tensor
    pooling_entropy: torch.Tensor
    layer_indices: tuple[int, ...]
    preprocess_records: tuple[PreprocessRecord, ...]

    def select_layers(
        self,
        requested_indices: Sequence[int],
        *,
        local_kind: str = "cls_guided_patch",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = {index: position for position, index in enumerate(self.layer_indices)}
        try:
            selected = [positions[int(index)] for index in requested_indices]
        except KeyError as error:
            raise KeyError(f"requested layer {error.args[0]} was not cached") from error
        if local_kind == "cls_guided_patch":
            local = self.cls_guided_patch
        elif local_kind == "mean_patch":
            local = self.mean_patch
        else:
            raise ValueError("local_kind must be cls_guided_patch or mean_patch")
        return self.cls[:, selected], local[:, selected], self.pooling_entropy[:, selected]


@dataclass(frozen=True)
class PoolingTemperaturePilot:
    """Per-image diagnostics for several pool temperatures from one backbone pass."""

    image_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    entropy_by_temperature: dict[float, torch.Tensor]
    guided_mean_cosine_by_temperature: dict[float, torch.Tensor]


def mean_patch_pool(patches: torch.Tensor) -> torch.Tensor:
    if patches.ndim != 3 or patches.shape[1] < 1:
        raise ValueError("patches must have shape [B, N, D] with N >= 1")
    return patches.mean(dim=1)


def cls_guided_pool(
    cls: torch.Tensor,
    patches: torch.Tensor,
    *,
    tau_p: float,
    eps: float = 1e-12,
    return_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pool patches using cosine alignment with the frozen layer CLS token."""
    if tau_p <= 0 or eps <= 0:
        raise ValueError("tau_p and eps must be positive")
    if cls.ndim != 2 or patches.ndim != 3:
        raise ValueError("expected cls [B,D] and patches [B,N,D]")
    if cls.shape[0] != patches.shape[0] or cls.shape[-1] != patches.shape[-1]:
        raise ValueError("CLS and patch tensor shapes are incompatible")
    if patches.shape[1] < 1:
        raise ValueError("at least one patch token is required")

    cls_scores = functional.normalize(cls, dim=-1, eps=eps)
    patch_scores = functional.normalize(patches, dim=-1, eps=eps)
    logits = torch.einsum("bd,bnd->bn", cls_scores, patch_scores) / tau_p
    weights = torch.softmax(logits, dim=-1)
    local = torch.einsum("bn,bnd->bd", weights, patches)
    if patches.shape[1] == 1:
        entropy = torch.zeros_like(weights[:, 0])
    else:
        entropy = -(
            weights * weights.clamp_min(eps).log()
        ).sum(dim=-1) / torch.log(torch.tensor(float(patches.shape[1]), device=patches.device))
    assert_finite("CLS-guided local pool", local)
    assert_finite("pooling entropy", entropy)
    if (entropy < -1e-5).any() or (entropy > 1.00001).any():
        raise ValueError("normalized pooling entropy fell outside [0, 1]")
    return local, entropy, weights if return_weights else None


def aggregate_layers(
    layer_tokens: Sequence[LayerTokens],
    pooling: PoolingConfig,
) -> LayerAggregateBatch:
    if not layer_tokens:
        raise ValueError("at least one layer is required")
    batch_size = layer_tokens[0].cls.shape[0]
    token_dim = layer_tokens[0].cls.shape[-1]
    layer_indices: list[int] = []
    cls_values: list[torch.Tensor] = []
    mean_values: list[torch.Tensor] = []
    guided_values: list[torch.Tensor] = []
    entropy_values: list[torch.Tensor] = []
    for tokens in layer_tokens:
        if tokens.cls.shape != (batch_size, token_dim):
            raise ValueError("inconsistent CLS dimensions across layers")
        if tokens.patches.shape[0] != batch_size or tokens.patches.shape[-1] != token_dim:
            raise ValueError("inconsistent patch dimensions across layers")
        guided, entropy, _ = cls_guided_pool(
            tokens.cls,
            tokens.patches,
            tau_p=pooling.temperature,
            eps=pooling.eps,
        )
        layer_indices.append(tokens.block_index)
        cls_values.append(tokens.cls)
        mean_values.append(mean_patch_pool(tokens.patches))
        guided_values.append(guided)
        entropy_values.append(entropy)
    return LayerAggregateBatch(
        cls=torch.stack(cls_values, dim=1),
        mean_patch=torch.stack(mean_values, dim=1),
        cls_guided_patch=torch.stack(guided_values, dim=1),
        pooling_entropy=torch.stack(entropy_values, dim=1),
        layer_indices=tuple(layer_indices),
    )


class FeatureExtractionRunner:
    """Extract all selected aggregate layers while preserving canonical image order."""

    def __init__(
        self,
        extractor: FrozenDinoV2Extractor,
        preprocess: PreprocessConfig,
        pooling: PoolingConfig,
    ) -> None:
        self.extractor = extractor
        self.preprocess = preprocess
        self.pooling = pooling

    def extract_images(
        self,
        images: Sequence[tuple[str, Image.Image]],
        *,
        layer_indices: Sequence[int] | None = None,
        backbone_batch_size: int = 32,
    ) -> AllLayerFeatures:
        if not images:
            raise ValueError("cannot extract an empty image sequence")
        if backbone_batch_size <= 0:
            raise ValueError("backbone_batch_size must be positive")
        requested = tuple(
            self.pooling.all_layer_indices if layer_indices is None else layer_indices
        )
        if len({image_id for image_id, _ in images}) != len(images):
            raise ValueError("image IDs must be unique within an extraction call")

        buckets = preprocess_many_by_shape(list(images), self.preprocess)
        features_by_id: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PreprocessRecord]] = {}
        for _, (bucket_batch, bucket_records) in buckets.items():
            for start in range(0, bucket_batch.shape[0], backbone_batch_size):
                batch = bucket_batch[start : start + backbone_batch_size]
                records = bucket_records[start : start + backbone_batch_size]
                tokens = self.extractor.extract_intermediate_tokens(batch, requested)
                aggregate = aggregate_layers(tokens, self.pooling)
                for position, record in enumerate(records):
                    features_by_id[record.image_id] = (
                        aggregate.cls[position].cpu(),
                        aggregate.mean_patch[position].cpu(),
                        aggregate.cls_guided_patch[position].cpu(),
                        aggregate.pooling_entropy[position].cpu(),
                        record,
                    )

        ordered_ids = tuple(image_id for image_id, _ in images)
        if set(features_by_id) != set(ordered_ids):
            raise RuntimeError("extraction result IDs do not match requested IDs")
        ordered = [features_by_id[image_id] for image_id in ordered_ids]
        return AllLayerFeatures(
            image_ids=ordered_ids,
            cls=torch.stack([item[0] for item in ordered]),
            mean_patch=torch.stack([item[1] for item in ordered]),
            cls_guided_patch=torch.stack([item[2] for item in ordered]),
            pooling_entropy=torch.stack([item[3] for item in ordered]),
            layer_indices=requested,
            preprocess_records=tuple(item[4] for item in ordered),
        )

    def pilot_pooling_temperatures(
        self,
        images: Iterable[tuple[str, Image.Image]],
        *,
        temperatures: Sequence[float],
        layer_indices: Sequence[int] | None = None,
        backbone_batch_size: int = 32,
        image_chunk_size: int = 128,
    ) -> PoolingTemperaturePilot:
        """Compute pooling diagnostics in bounded decoded-image chunks."""
        if not temperatures:
            raise ValueError("at least one candidate pooling temperature is required")
        candidate_temperatures = tuple(float(value) for value in temperatures)
        if any(value <= 0 for value in candidate_temperatures):
            raise ValueError("candidate pooling temperatures must be positive")
        if len(set(candidate_temperatures)) != len(candidate_temperatures):
            raise ValueError("candidate pooling temperatures must be unique")
        if backbone_batch_size <= 0 or image_chunk_size <= 0:
            raise ValueError("backbone_batch_size and image_chunk_size must be positive")
        requested = tuple(
            self.pooling.all_layer_indices if layer_indices is None else layer_indices
        )
        by_temperature: dict[
            float,
            dict[str, tuple[torch.Tensor, torch.Tensor]],
        ] = {temperature: {} for temperature in candidate_temperatures}
        image_ids: list[str] = []
        seen_ids: set[str] = set()

        def process_chunk(chunk: list[tuple[str, Image.Image]]) -> None:
            buckets = preprocess_many_by_shape(chunk, self.preprocess)
            for _, (bucket_batch, bucket_records) in buckets.items():
                for start in range(0, bucket_batch.shape[0], backbone_batch_size):
                    batch = bucket_batch[start : start + backbone_batch_size]
                    records = bucket_records[start : start + backbone_batch_size]
                    tokens = self.extractor.extract_intermediate_tokens(batch, requested)
                    means = torch.stack([mean_patch_pool(token.patches) for token in tokens], dim=1)
                    for temperature in candidate_temperatures:
                        entropy = []
                        guided = []
                        for token in tokens:
                            local, layer_entropy, _ = cls_guided_pool(
                                token.cls,
                                token.patches,
                                tau_p=temperature,
                                eps=self.pooling.eps,
                            )
                            guided.append(local)
                            entropy.append(layer_entropy)
                        guided_tensor = torch.stack(guided, dim=1)
                        entropy_tensor = torch.stack(entropy, dim=1)
                        similarity = functional.cosine_similarity(
                            guided_tensor,
                            means,
                            dim=-1,
                            eps=self.pooling.eps,
                        )
                        for position, record in enumerate(records):
                            by_temperature[temperature][record.image_id] = (
                                entropy_tensor[position].cpu(),
                                similarity[position].cpu(),
                            )

        chunk: list[tuple[str, Image.Image]] = []
        for image_id, image in images:
            if image_id in seen_ids:
                raise ValueError("image IDs must be unique within a pooling pilot")
            seen_ids.add(image_id)
            image_ids.append(image_id)
            chunk.append((image_id, image))
            if len(chunk) == image_chunk_size:
                process_chunk(chunk)
                chunk = []
        if chunk:
            process_chunk(chunk)
        if not image_ids:
            raise ValueError("cannot pilot an empty image sequence")
        entropy_by_temperature = {
            temperature: torch.stack(
                [by_temperature[temperature][image_id][0] for image_id in image_ids]
            )
            for temperature in candidate_temperatures
        }
        guided_mean_cosine_by_temperature = {
            temperature: torch.stack(
                [by_temperature[temperature][image_id][1] for image_id in image_ids]
            )
            for temperature in candidate_temperatures
        }
        return PoolingTemperaturePilot(
            image_ids=tuple(image_ids),
            layer_indices=requested,
            entropy_by_temperature=entropy_by_temperature,
            guided_mean_cosine_by_temperature=guided_mean_cosine_by_temperature,
        )


def concat_all_layer_features(batches: Sequence[AllLayerFeatures]) -> AllLayerFeatures:
    """Join sequential extraction batches without retaining decoded images."""
    if not batches:
        raise ValueError("at least one feature batch is required")
    layer_indices = batches[0].layer_indices
    if any(batch.layer_indices != layer_indices for batch in batches):
        raise ValueError("feature batches use different layer indices")
    image_ids = tuple(image_id for batch in batches for image_id in batch.image_ids)
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("feature batches contain duplicate image IDs")
    return AllLayerFeatures(
        image_ids=image_ids,
        cls=torch.cat([batch.cls for batch in batches]),
        mean_patch=torch.cat([batch.mean_patch for batch in batches]),
        cls_guided_patch=torch.cat([batch.cls_guided_patch for batch in batches]),
        pooling_entropy=torch.cat([batch.pooling_entropy for batch in batches]),
        layer_indices=layer_indices,
        preprocess_records=tuple(
            record for batch in batches for record in batch.preprocess_records
        ),
    )


def extract_image_stream(
    runner: FeatureExtractionRunner,
    images: Iterable[tuple[str, Image.Image]],
    *,
    layer_indices: Sequence[int],
    backbone_batch_size: int,
    image_chunk_size: int = 256,
) -> AllLayerFeatures:
    """Extract a sequence in bounded decoded-image chunks."""
    if image_chunk_size <= 0:
        raise ValueError("image_chunk_size must be positive")
    batches: list[AllLayerFeatures] = []
    chunk: list[tuple[str, Image.Image]] = []
    for item in images:
        chunk.append(item)
        if len(chunk) == image_chunk_size:
            batches.append(
                runner.extract_images(
                    chunk,
                    layer_indices=layer_indices,
                    backbone_batch_size=backbone_batch_size,
                )
            )
            chunk = []
    if chunk:
        batches.append(
            runner.extract_images(
                chunk,
                layer_indices=layer_indices,
                backbone_batch_size=backbone_batch_size,
            )
        )
    return concat_all_layer_features(batches)
