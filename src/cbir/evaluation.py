"""Exact retrieval ranking and SfM/RevisitOP protocol evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .cache import FeatureShardReader
from .data.revisitop import RevisitGroundTruth
from .data.sfm import ValidationCase
from .utils import assert_finite, l2_normalize


@dataclass(frozen=True)
class SfmRetrievalReport:
    num_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ranks: Mapping[str, int]


@dataclass(frozen=True)
class ProtocolReport:
    map: float
    mean_precision_at_10: float
    ap_by_query: Mapping[str, float]
    precision_at_10_by_query: Mapping[str, float]


@dataclass(frozen=True)
class RevisitReport:
    easy: ProtocolReport | None
    medium: ProtocolReport
    hard: ProtocolReport


@torch.inference_mode()
def descriptors_from_cache(
    reader: FeatureShardReader,
    head: nn.Module,
    image_ids: Sequence[str],
    *,
    layer_indices: Sequence[int],
    local_kind: str = "cls_guided_patch",
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Apply a trained head to cached features in canonical requested order."""
    if not image_ids:
        raise ValueError("cannot compute descriptors for an empty image list")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    head = head.eval().to(device)
    outputs: list[torch.Tensor] = []
    for start in range(0, len(image_ids), batch_size):
        selected_ids = image_ids[start : start + batch_size]
        batch = reader.fetch(
            selected_ids,
            layer_indices=layer_indices,
            local_kind=local_kind,
        )
        descriptors = head(
            batch["cls"].to(device),
            batch["local"].to(device),
            batch["entropy"].to(device),
        )
        if not isinstance(descriptors, torch.Tensor):
            descriptors = descriptors[0]
        outputs.append(descriptors.detach().cpu())
    result = torch.cat(outputs, dim=0)
    assert_finite("cached descriptors", result)
    return l2_normalize(result)


@torch.inference_mode()
def descriptors_from_feature_tensors(
    cls: torch.Tensor,
    head: nn.Module,
    *,
    local: torch.Tensor | None = None,
    entropy: torch.Tensor | None = None,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Apply a descriptor head to an in-memory frozen-feature cache."""
    if cls.ndim != 3 or not len(cls):
        raise ValueError("cls must have shape [N, K, D] with N > 0")
    if local is not None and local.shape != cls.shape:
        raise ValueError("local features must match cls shape")
    if entropy is not None and entropy.shape != cls.shape[:2]:
        raise ValueError("entropy must have shape [N, K]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    head = head.eval().to(device)
    outputs: list[torch.Tensor] = []
    for start in range(0, cls.shape[0], batch_size):
        stop = start + batch_size
        result = head(
            cls[start:stop].float().to(device),
            None if local is None else local[start:stop].float().to(device),
            None if entropy is None else entropy[start:stop].float().to(device),
        )
        if not isinstance(result, torch.Tensor):
            result = result[0]
        outputs.append(result.cpu())
    return l2_normalize(torch.cat(outputs, dim=0))



def rank_exact(
    query_descriptors: torch.Tensor,
    database_descriptors: torch.Tensor,
    *,
    query_block_size: int = 256,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return database indices in descending cosine-score order for each query."""
    if query_descriptors.ndim != 2 or database_descriptors.ndim != 2:
        raise ValueError("query and database descriptors must be matrices")
    if query_descriptors.shape[1] != database_descriptors.shape[1]:
        raise ValueError("query/database descriptor dimensions differ")
    if query_block_size <= 0:
        raise ValueError("query_block_size must be positive")
    computation_device = (
        query_descriptors.device if device is None else torch.device(device)
    )
    queries = l2_normalize(query_descriptors.float()).to(computation_device)
    database = l2_normalize(database_descriptors.float()).to(computation_device)
    assert_finite("query descriptors", queries)
    assert_finite("database descriptors", database)
    ranks: list[torch.Tensor] = []
    database_t = database.T
    for start in range(0, queries.shape[0], query_block_size):
        scores = queries[start : start + query_block_size] @ database_t
        ranks.append(torch.argsort(scores, dim=1, descending=True).cpu())
    return torch.cat(ranks, dim=0)


def ranked_ids_from_descriptors(
    query_descriptors: torch.Tensor,
    database_descriptors: torch.Tensor,
    database_ids: Sequence[str],
    *,
    query_block_size: int = 256,
) -> list[list[str]]:
    if database_descriptors.shape[0] != len(database_ids):
        raise ValueError("database descriptor rows and IDs differ")
    ranks = rank_exact(
        query_descriptors,
        database_descriptors,
        query_block_size=query_block_size,
    )
    return [[database_ids[index] for index in row.tolist()] for row in ranks]


def final_cls_descriptors_from_cache(
    reader: FeatureShardReader,
    image_ids: Sequence[str],
    *,
    final_layer_index: int = 11,
) -> torch.Tensor:
    """Frozen native-DINO final CLS baseline, with no trainable head."""
    batch = reader.fetch(
        image_ids,
        layer_indices=(final_layer_index,),
        local_kind="cls_guided_patch",
    )
    return l2_normalize(batch["cls"][:, 0])


def evaluate_sfm_verified_pairs(
    descriptors: torch.Tensor,
    image_ids: Sequence[str],
    cases: Sequence[ValidationCase],
    *,
    query_block_size: int = 256,
    device: torch.device | str | None = None,
) -> SfmRetrievalReport:
    """Evaluate selected SfM positives while masking unjudged same-cluster views."""
    if descriptors.shape[0] != len(image_ids):
        raise ValueError("descriptor rows and image_ids length differ")
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("image_ids must be unique")
    if not cases:
        raise ValueError("at least one validation case is required")
    index = {image_id: position for position, image_id in enumerate(image_ids)}
    query_ids = [case.query_id for case in cases]
    for case in cases:
        if case.query_id not in index or case.positive_id not in index:
            raise KeyError("SfM validation case endpoint is absent from descriptors")

    query_descriptors = descriptors[[index[query_id] for query_id in query_ids]]
    ranked_indices = rank_exact(
        query_descriptors,
        descriptors,
        query_block_size=query_block_size,
        device=device,
    )
    ranks_by_query: dict[str, int] = {}
    for row, case in enumerate(cases):
        rank = 0
        found = False
        for database_index in ranked_indices[row].tolist():
            candidate_id = image_ids[database_index]
            if candidate_id in case.ignored_ids:
                continue
            rank += 1
            if candidate_id == case.positive_id:
                ranks_by_query[case.query_id] = rank
                found = True
                break
        if not found:
            raise RuntimeError(f"designated positive was not ranked for {case.query_id}")
    ranks = np.asarray(list(ranks_by_query.values()), dtype=np.int64)
    return SfmRetrievalReport(
        num_queries=len(cases),
        recall_at_1=float(np.mean(ranks <= 1)),
        recall_at_5=float(np.mean(ranks <= 5)),
        recall_at_10=float(np.mean(ranks <= 10)),
        mrr=float(np.mean(1.0 / ranks)),
        ranks=ranks_by_query,
    )


def evaluate_revisitop(
    ranked_database_ids: Sequence[Sequence[str]],
    query_ids: Sequence[str],
    ground_truth: Mapping[str, RevisitGroundTruth],
    *,
    protocols: Sequence[Literal["easy", "medium", "hard"]] = ("medium", "hard"),
) -> RevisitReport:
    """Reproduce official RevisitOP AP and mP@10 semantics using database IDs."""
    if len(ranked_database_ids) != len(query_ids):
        raise ValueError("number of ranked lists and query IDs differ")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    enabled = set(protocols)
    if not {"medium", "hard"}.issubset(enabled):
        raise ValueError("Medium and Hard are required final RevisitOP protocols")
    reports: dict[str, ProtocolReport] = {}
    for protocol in protocols:
        ap_by_query: dict[str, float] = {}
        p10_by_query: dict[str, float] = {}
        for query_id, ranking in zip(query_ids, ranked_database_ids, strict=True):
            try:
                gnd = ground_truth[query_id]
            except KeyError as error:
                raise KeyError(f"missing RevisitOP ground truth for {query_id}") from error
            positives, ignored = _protocol_sets(gnd, protocol)
            if not positives:
                continue
            _validate_ranking(ranking)
            positive_positions = np.asarray(
                [position for position, image_id in enumerate(ranking) if image_id in positives],
                dtype=np.int64,
            )
            junk_positions = np.asarray(
                [position for position, image_id in enumerate(ranking) if image_id in ignored],
                dtype=np.int64,
            )
            adjusted_positions = _subtract_junk_before_positives(
                positive_positions,
                junk_positions,
            )
            ap_by_query[query_id] = _compute_ap(
                adjusted_positions,
                number_of_positives=len(positives),
            )
            p10_by_query[query_id] = _compute_precision_at_k(
                adjusted_positions,
                kappa=10,
            )
        if not ap_by_query:
            raise ValueError(f"{protocol} has no nonempty RevisitOP queries")
        reports[protocol] = ProtocolReport(
            map=float(np.mean(list(ap_by_query.values()))),
            mean_precision_at_10=float(np.mean(list(p10_by_query.values()))),
            ap_by_query=ap_by_query,
            precision_at_10_by_query=p10_by_query,
        )
    return RevisitReport(
        easy=reports.get("easy"),
        medium=reports["medium"],
        hard=reports["hard"],
    )


def _protocol_sets(
    gnd: RevisitGroundTruth,
    protocol: Literal["easy", "medium", "hard"],
) -> tuple[frozenset[str], frozenset[str]]:
    if protocol == "easy":
        return gnd.easy, gnd.junk | gnd.hard
    if protocol == "medium":
        return gnd.easy | gnd.hard, gnd.junk
    if protocol == "hard":
        return gnd.hard, gnd.junk | gnd.easy
    raise ValueError(f"unknown RevisitOP protocol {protocol}")


def _validate_ranking(ranking: Sequence[str]) -> None:
    if len(ranking) != len(set(ranking)):
        raise ValueError("database ranking contains duplicate IDs")


def _subtract_junk_before_positives(
    positive_positions: np.ndarray,
    junk_positions: np.ndarray,
) -> np.ndarray:
    """Match official compute_map's zero-based position correction."""
    if not len(junk_positions):
        return positive_positions.copy()
    adjusted = positive_positions.copy()
    junk_cursor = 0
    counted = 0
    for position, positive_position in enumerate(adjusted):
        while junk_cursor < len(junk_positions) and positive_position > junk_positions[junk_cursor]:
            counted += 1
            junk_cursor += 1
        adjusted[position] = positive_position - counted
    return adjusted


def _compute_ap(positive_positions: np.ndarray, number_of_positives: int) -> float:
    if number_of_positives <= 0:
        raise ValueError("number_of_positives must be positive")
    ap = 0.0
    recall_step = 1.0 / number_of_positives
    for hit_index, rank in enumerate(positive_positions):
        precision_0 = 1.0 if rank == 0 else float(hit_index) / float(rank)
        precision_1 = float(hit_index + 1) / float(rank + 1)
        ap += (precision_0 + precision_1) * recall_step / 2.0
    return ap


def _compute_precision_at_k(positive_positions: np.ndarray, kappa: int) -> float:
    if not len(positive_positions):
        return 0.0
    positions_one_based = positive_positions + 1
    effective_k = min(int(positions_one_based.max()), kappa)
    return float(np.sum(positions_one_based <= effective_k) / effective_k)
