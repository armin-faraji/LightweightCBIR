#!/usr/bin/env python3
"""Run one locked checkpoint on ROxford or RParis with official protocol semantics."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Callable, Sequence

import torch

from cbir.backbone import FrozenDinoV2Extractor
from cbir.cache import sha256_file
from cbir.config import FusionConfig, config_to_dict, load_project_config
from cbir.data.revisitop import RevisitOPDataset
from cbir.evaluation import (
    descriptors_from_images,
    evaluate_revisitop,
    ranked_ids_from_descriptors,
)
from cbir.features import FeatureExtractionRunner
from cbir.fusion import ReliabilityGatedFusion
from cbir.utils import atomic_write_json, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/final.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revisit-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("roxford5k", "rparis6k"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/revisitop"))
    parser.add_argument("--backbone-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backbone_batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("batch/chunk sizes must be positive")
    config = load_project_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fusion_config = _fusion_from_checkpoint(checkpoint)
    head = ReliabilityGatedFusion.from_config(fusion_config)
    head.load_state_dict(checkpoint["model_state_dict"])
    head.eval()
    if "cache_fingerprint" not in checkpoint:
        raise ValueError(
            "checkpoint lacks cache_fingerprint provenance; retrain with the current "
            "HeadTrainer before final RevisitOP evaluation"
        )
    expected_extraction = {
        "backbone": config_to_dict(config.backbone),
        "preprocess": config_to_dict(config.preprocess),
        "pooling": config_to_dict(config.pooling),
    }
    if checkpoint.get("extraction_config") != expected_extraction:
        raise ValueError(
            "final config does not match the frozen feature extraction settings "
            "stored in the training checkpoint"
        )

    dataset_root = args.revisit_root / args.dataset
    dataset = RevisitOPDataset.from_ground_truth_pickle(
        name=args.dataset,
        ground_truth_path=dataset_root / f"gnd_{args.dataset}.pkl",
        image_root=dataset_root / "jpg",
    )
    extractor = FrozenDinoV2Extractor(config.backbone)
    runner = FeatureExtractionRunner(extractor, config.preprocess, config.pooling)
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor_fingerprint = stable_hash(
        {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_cache_fingerprint": checkpoint["cache_fingerprint"],
            "backbone": config_to_dict(config.backbone),
            "preprocess": config_to_dict(config.preprocess),
            "pooling": config_to_dict(config.pooling),
            "fusion": config_to_dict(fusion_config),
            "dataset": args.dataset,
        }
    )

    database_descriptors = _load_or_extract_bundle(
        output_dir / "database_descriptors.pt",
        fingerprint=descriptor_fingerprint,
        ids=dataset.database_ids,
        image_loader=dataset.read_database_image,
        runner=runner,
        head=head,
        fusion_config=fusion_config,
        chunk_size=args.chunk_size,
        backbone_batch_size=args.backbone_batch_size,
        device=config.backbone.device,
        force=args.force,
    )
    query_ids = tuple(query.query_id for query in dataset.queries)
    query_lookup = {query.query_id: query for query in dataset.queries}
    query_descriptors = _load_or_extract_bundle(
        output_dir / "query_descriptors.pt",
        fingerprint=descriptor_fingerprint,
        ids=query_ids,
        image_loader=lambda query_id: dataset.crop_query(query_lookup[query_id]),
        runner=runner,
        head=head,
        fusion_config=fusion_config,
        chunk_size=args.chunk_size,
        backbone_batch_size=args.backbone_batch_size,
        device=config.backbone.device,
        force=args.force,
    )
    rankings = ranked_ids_from_descriptors(
        query_descriptors,
        database_descriptors,
        dataset.database_ids,
        query_block_size=config.evaluation.query_block_size,
    )
    report = evaluate_revisitop(rankings, query_ids, dataset.ground_truth)
    output = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "checkpoint_cache_fingerprint": checkpoint["cache_fingerprint"],
        "descriptor_fingerprint": descriptor_fingerprint,
        "medium": _protocol_to_dict(report.medium),
        "hard": _protocol_to_dict(report.hard),
        "easy": None if report.easy is None else _protocol_to_dict(report.easy),
    }
    atomic_write_json(output_dir / "evaluation_report.json", output)
    print(
        f"{args.dataset} Medium mAP={report.medium.map:.4f}, "
        f"mP@10={report.medium.mean_precision_at_10:.4f}; "
        f"Hard mAP={report.hard.map:.4f}, "
        f"mP@10={report.hard.mean_precision_at_10:.4f}"
    )


def _fusion_from_checkpoint(checkpoint: dict) -> FusionConfig:
    raw = dict(checkpoint["fusion_config"])
    raw["layer_indices"] = tuple(int(index) for index in raw["layer_indices"])
    return FusionConfig(**raw)


def _load_or_extract_bundle(
    path: Path,
    *,
    fingerprint: str,
    ids: Sequence[str],
    image_loader: Callable[[str], object],
    runner: FeatureExtractionRunner,
    head: ReliabilityGatedFusion,
    fusion_config: FusionConfig,
    chunk_size: int,
    backbone_batch_size: int,
    device: str,
    force: bool,
) -> torch.Tensor:
    if path.is_file() and not force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("fingerprint") == fingerprint and tuple(payload.get("ids", ())) == tuple(ids):
            return payload["descriptors"].float()
    outputs: list[torch.Tensor] = []
    for start in range(0, len(ids), chunk_size):
        chunk_ids = ids[start : start + chunk_size]
        images = [(image_id, image_loader(image_id)) for image_id in chunk_ids]
        outputs.append(
            descriptors_from_images(
                runner,
                head,
                images,
                layer_indices=fusion_config.layer_indices,
                local_kind=fusion_config.local_kind,
                backbone_batch_size=backbone_batch_size,
                device=device,
            )
        )
    descriptors = torch.cat(outputs, dim=0)
    _atomic_torch_save(
        path,
        {
            "fingerprint": fingerprint,
            "ids": list(ids),
            "descriptors": descriptors.half(),
        },
    )
    return descriptors


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _protocol_to_dict(protocol: object) -> dict:
    return {
        "map": protocol.map,
        "mean_precision_at_10": protocol.mean_precision_at_10,
        "ap_by_query": dict(protocol.ap_by_query),
        "precision_at_10_by_query": dict(protocol.precision_at_10_by_query),
    }


if __name__ == "__main__":
    main()
