"""Cached-pair training for compact descriptor heads using symmetric InfoNCE."""

from __future__ import annotations

import os
import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .cache import FeatureShardReader
from .config import FusionConfig, TrainingConfig, config_to_dict, fusion_config_from_dict
from .data.sfm import PairRecord, ValidationCase
from .evaluation import SfmRetrievalReport, descriptors_from_cache, evaluate_sfm_verified_pairs
from .fusion import FusionDiagnostics, build_descriptor_head
from .utils import assert_finite, l2_normalize, seed_everything


@dataclass(frozen=True)
class SampledPair:
    pair_index: int
    reverse: bool


class PairIndexDataset(Dataset[PairRecord]):
    """Lightweight pair metadata dataset; cache tensors are fetched in collate."""

    def __init__(self, pairs: Sequence[PairRecord]) -> None:
        self.pairs = tuple(pairs)
        if not self.pairs:
            raise ValueError("at least one train pair is required")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: SampledPair | int) -> PairRecord:
        if isinstance(index, SampledPair):
            pair = self.pairs[index.pair_index]
            if index.reverse:
                return PairRecord(
                    query_id=pair.positive_id,
                    positive_id=pair.query_id,
                    cluster_id=pair.cluster_id,
                    split=pair.split,
                )
            return pair
        return self.pairs[int(index)]


class ClusterUniquePairBatchSampler(Sampler[list[SampledPair]]):
    """Yield each pair once/epoch while allowing at most one pair per cluster/batch."""

    def __init__(
        self,
        pairs: Sequence[PairRecord],
        *,
        batch_size: int,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.pairs = tuple(pairs)
        if not self.pairs:
            raise ValueError("at least one pair is required")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self._by_cluster: dict[int, list[int]] = defaultdict(list)
        for index, pair in enumerate(self.pairs):
            self._by_cluster[pair.cluster_id].append(index)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[SampledPair]]:
        rng = random.Random(self.seed + self.epoch)
        pending = {
            cluster: _shuffled(indices, rng)
            for cluster, indices in self._by_cluster.items()
        }
        # Symmetric InfoNCE needs at least two pairs. Once one cluster is all
        # that remains, its residual pairs cannot form a cluster-safe batch and
        # are deferred to a later epoch rather than becoming false negatives.
        while len(pending) >= 2:
            available = list(pending)
            rng.shuffle(available)
            chosen_clusters = available[: self.batch_size]
            batch: list[SampledPair] = []
            for cluster in chosen_clusters:
                pair_index = pending[cluster].pop()
                batch.append(SampledPair(pair_index=pair_index, reverse=bool(rng.getrandbits(1))))
                if not pending[cluster]:
                    del pending[cluster]
            if len({self.pairs[item.pair_index].cluster_id for item in batch}) != len(batch):
                raise RuntimeError("sampler produced duplicate cluster in a batch")
            yield batch

    def __len__(self) -> int:
        # Lower-bound estimate; DataLoader does not use it for correctness.
        longest_cluster = max(len(indices) for indices in self._by_cluster.values())
        packed = (len(self.pairs) + self.batch_size - 1) // self.batch_size
        return max(longest_cluster, packed)


class CachedPairCollator:
    """Batch metadata pairs by reading all feature tensors from the cache once."""

    def __init__(
        self,
        reader: FeatureShardReader,
        *,
        layer_indices: Sequence[int],
        local_kind: str,
    ) -> None:
        self.reader = reader
        self.layer_indices = tuple(layer_indices)
        self.local_kind = local_kind

    def __call__(self, pairs: Sequence[PairRecord]) -> dict[str, Any]:
        if not pairs:
            raise ValueError("cannot collate an empty pair batch")
        clusters = [pair.cluster_id for pair in pairs]
        if len(clusters) != len(set(clusters)):
            raise ValueError("InfoNCE batch contains more than one pair per cluster")
        query_ids = [pair.query_id for pair in pairs]
        positive_ids = [pair.positive_id for pair in pairs]
        query = self.reader.fetch(
            query_ids,
            layer_indices=self.layer_indices,
            local_kind=self.local_kind,
        )
        positive = self.reader.fetch(
            positive_ids,
            layer_indices=self.layer_indices,
            local_kind=self.local_kind,
        )
        return {
            "query": query,
            "positive": positive,
            "cluster_ids": torch.tensor(clusters, dtype=torch.long),
            "query_ids": query_ids,
            "positive_ids": positive_ids,
        }


def symmetric_info_nce(
    query_descriptors: torch.Tensor,
    positive_descriptors: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Bidirectional batch InfoNCE for normalized query-positive descriptor pairs."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if query_descriptors.ndim != 2 or positive_descriptors.ndim != 2:
        raise ValueError("InfoNCE descriptors must be matrices")
    if query_descriptors.shape != positive_descriptors.shape:
        raise ValueError("query and positive descriptor shapes differ")
    if query_descriptors.shape[0] < 2:
        raise ValueError("InfoNCE requires at least two pairs per batch")
    assert_finite("InfoNCE query descriptors", query_descriptors)
    assert_finite("InfoNCE positive descriptors", positive_descriptors)
    query = l2_normalize(query_descriptors)
    positive = l2_normalize(positive_descriptors)
    logits = query @ positive.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        functional.cross_entropy(logits, targets)
        + functional.cross_entropy(logits.T, targets)
    )


@dataclass
class TrainingHistory:
    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int | None = None
    best_metric: float = float("-inf")
    best_checkpoint: Path | None = None


class HeadTrainer:
    """Train only a compact descriptor head from a validated feature cache."""

    def __init__(
        self,
        *,
        head: nn.Module,
        reader: FeatureShardReader,
        train_pairs: Sequence[PairRecord],
        fusion_config: FusionConfig,
        training_config: TrainingConfig,
        validation_cases: Sequence[ValidationCase] | None = None,
        validation_image_ids: Sequence[str] | None = None,
        output_dir: Path = Path("outputs/training"),
    ) -> None:
        if tuple(fusion_config.layer_indices) != tuple(head.config.layer_indices):
            raise ValueError("head layers and fusion config layers differ")
        if fusion_config.local_kind not in {"cls_guided_patch", "mean_patch"}:
            raise ValueError("unsupported local feature kind")
        self.head = head
        self.reader = reader
        self.train_pairs = tuple(train_pairs)
        self.fusion_config = fusion_config
        self.training_config = training_config
        self.validation_cases = tuple(validation_cases or ())
        self.validation_image_ids = tuple(validation_image_ids or ())
        if bool(self.validation_cases) != bool(self.validation_image_ids):
            raise ValueError(
                "validation_cases and validation_image_ids must be supplied together"
            )
        self.output_dir = Path(output_dir)
        self.device = torch.device(training_config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("training requests CUDA but CUDA is unavailable")
        self.head.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.epochs,
        )

    def fit(
        self,
        *,
        max_epochs: int | None = None,
        enable_early_stopping: bool = True,
        save_best_checkpoint: bool = True,
    ) -> TrainingHistory:
        """Train for at most ``max_epochs`` while retaining the configured schedule.

        A full-data final fit can stop at a validation-selected epoch without
        changing the scheduler's original ``T_max`` or consulting a test set.
        """
        epochs = self.training_config.epochs if max_epochs is None else max_epochs
        if not 1 <= epochs <= self.training_config.epochs:
            raise ValueError(
                "max_epochs must be between one and training_config.epochs"
            )
        seed_everything(self.training_config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        history = TrainingHistory()
        stale_epochs = 0
        dataset = PairIndexDataset(self.train_pairs)
        sampler = ClusterUniquePairBatchSampler(
            self.train_pairs,
            batch_size=self.training_config.batch_size,
            seed=self.training_config.seed,
        )
        collator = CachedPairCollator(
            self.reader,
            layer_indices=self.fusion_config.layer_indices,
            local_kind=self.fusion_config.local_kind,
        )

        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collator,
                num_workers=0,
            )
            train_metrics = self._train_epoch(loader)
            epoch_metrics = {"epoch": float(epoch), **train_metrics}
            if self.validation_cases:
                validation = self.evaluate_validation()
                epoch_metrics.update(
                    {
                        "val_recall_at_1": validation.recall_at_1,
                        "val_recall_at_5": validation.recall_at_5,
                        "val_recall_at_10": validation.recall_at_10,
                        "val_mrr": validation.mrr,
                    }
                )
                metric = validation.recall_at_1
            else:
                metric = -train_metrics["loss"]
            history.epochs.append(epoch_metrics)
            if metric > history.best_metric:
                history.best_metric = metric
                history.best_epoch = epoch
                if save_best_checkpoint:
                    history.best_checkpoint = self._save_checkpoint(epoch, history)
                stale_epochs = 0
            else:
                stale_epochs += 1
            self.scheduler.step()
            if (
                enable_early_stopping
                and stale_epochs >= self.training_config.early_stopping_patience
            ):
                break
        return history

    def save_final_checkpoint(self, history: TrainingHistory) -> Path:
        """Save the last fitted epoch as a distinct final-training checkpoint."""
        if not history.epochs:
            raise ValueError("cannot save a final checkpoint before training")
        final_epoch = int(history.epochs[-1]["epoch"])
        return self._save_checkpoint(
            final_epoch,
            history,
            filename="final.pt",
            checkpoint_kind="final",
        )

    def _train_epoch(self, loader: DataLoader[dict[str, Any]]) -> dict[str, float]:
        self.head.train()
        total_loss = 0.0
        batches = 0
        entropy_penalty_scales: list[float] = []
        for batch in loader:
            query = {name: value.to(self.device) for name, value in batch["query"].items()}
            positive = {
                name: value.to(self.device) for name, value in batch["positive"].items()
            }
            self.optimizer.zero_grad(set_to_none=True)
            query_output = self.head(
                query["cls"],
                query["local"],
                query["entropy"],
                return_diagnostics=True,
            )
            if not isinstance(query_output, tuple):
                raise TypeError("descriptor head did not return diagnostics")
            query_descriptor, query_diag = query_output
            if not isinstance(query_diag, FusionDiagnostics):
                raise TypeError("descriptor head returned unsupported diagnostics")
            positive_descriptor = self.head(
                positive["cls"],
                positive["local"],
                positive["entropy"],
            )
            if not isinstance(positive_descriptor, torch.Tensor):
                positive_descriptor = positive_descriptor[0]
            loss = symmetric_info_nce(
                query_descriptor,
                positive_descriptor,
                temperature=self.training_config.loss_temperature,
            )
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach().cpu())
            if query_diag.entropy_penalty_scale is not None:
                entropy_penalty_scales.append(
                    float(query_diag.entropy_penalty_scale.detach().cpu())
                )
            batches += 1
        if batches == 0:
            raise RuntimeError("training epoch produced no batches")
        return {
            "loss": total_loss / batches,
            "entropy_penalty_scale": (
                float(sum(entropy_penalty_scales) / len(entropy_penalty_scales))
                if entropy_penalty_scales
                else 0.0
            ),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    @torch.inference_mode()
    def evaluate_validation(self) -> SfmRetrievalReport:
        descriptors = descriptors_from_cache(
            self.reader,
            self.head,
            self.validation_image_ids,
            layer_indices=self.fusion_config.layer_indices,
            local_kind=self.fusion_config.local_kind,
            device=self.device,
        )
        return evaluate_sfm_verified_pairs(
            descriptors,
            self.validation_image_ids,
            self.validation_cases,
            device=self.device,
        )

    def _save_checkpoint(
        self,
        epoch: int,
        history: TrainingHistory,
        *,
        filename: str = "best.pt",
        checkpoint_kind: str = "best",
    ) -> Path:
        path = self.output_dir / filename
        payload = {
            "epoch": epoch,
            "checkpoint_kind": checkpoint_kind,
            "cache_fingerprint": self.reader.manifest.fingerprint,
            "extraction_config": dict(self.reader.manifest.extraction_config),
            "model_state_dict": self.head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "fusion_config": config_to_dict(self.fusion_config),
            "training_config": config_to_dict(self.training_config),
            "history": {"epochs": history.epochs, "best_metric": history.best_metric},
        }
        temporary = _temporary_checkpoint_path(path)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return path


def load_descriptor_head(
    checkpoint_path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a locally saved descriptor head and its small checkpoint payload."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "fusion_config" not in payload:
        raise ValueError(f"invalid descriptor checkpoint: {checkpoint_path}")
    fusion = fusion_config_from_dict(payload["fusion_config"])
    head = build_descriptor_head(fusion)
    head.load_state_dict(payload["model_state_dict"])
    return head.to(device).eval(), payload


def _shuffled(values: Sequence[int], rng: random.Random) -> list[int]:
    result = list(values)
    rng.shuffle(result)
    return result


def _temporary_checkpoint_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)
