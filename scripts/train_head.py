#!/usr/bin/env python3
"""Train a frozen-feature compact descriptor head and save its best checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from cbir.cache import FeatureShardReader
from cbir.config import (
    config_to_dict,
    extraction_fingerprint,
    load_project_config,
    train_fingerprint,
)
from cbir.data.sfm import Sfm30kMetadata
from cbir.evaluation import evaluate_sfm_verified_pairs, final_cls_descriptors_from_cache
from cbir.fusion import build_descriptor_head
from cbir.training import HeadTrainer
from cbir.utils import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tuning.yaml"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    metadata = Sfm30kMetadata.from_official_files(
        config.sfm.metadata_path,
        _require_path(config.sfm.names_clusters_path, "names_clusters_path"),
    )
    reader = FeatureShardReader(args.cache_dir)
    expected_cache_fingerprint = extraction_fingerprint(
        backbone=config.backbone,
        preprocess=config.preprocess,
        pooling=config.pooling,
        source_ids_hash=reader.manifest.source_ids_hash,
    )
    if reader.manifest.fingerprint != expected_cache_fingerprint:
        raise ValueError(
            "training config does not reproduce this cache's extraction fingerprint. "
            "Use the exact backbone/preprocess/pooling configuration used for cache creation."
        )
    expected_ids = set(metadata.image_ids())
    if set(reader.image_ids) != expected_ids:
        raise ValueError(
            "training requires a complete full-SfM cache. Use a pilot cache only "
            "for smoke tests with an explicitly filtered metadata protocol."
        )
    validation_ids = metadata.image_ids("val")
    validation_cases = metadata.build_validation_cases()

    baseline = final_cls_descriptors_from_cache(reader, validation_ids)
    baseline_report = evaluate_sfm_verified_pairs(
        baseline,
        validation_ids,
        validation_cases,
        query_block_size=config.evaluation.query_block_size,
    )
    print(
        "Frozen final CLS baseline: "
        f"R@1={baseline_report.recall_at_1:.4f}, MRR={baseline_report.mrr:.4f}"
    )

    head = build_descriptor_head(config.fusion)
    run_fingerprint = train_fingerprint(
        cache_fingerprint=reader.manifest.fingerprint,
        fusion=config.fusion,
        training=config.training,
    )
    output_dir = args.output_dir / run_fingerprint[:12]
    trainer = HeadTrainer(
        head=head,
        reader=reader,
        train_pairs=metadata.train_pairs,
        fusion_config=config.fusion,
        training_config=config.training,
        validation_cases=validation_cases,
        validation_image_ids=validation_ids,
        output_dir=output_dir,
    )
    history = trainer.fit()
    atomic_write_json(
        output_dir / "run_summary.json",
        {
            "cache_fingerprint": reader.manifest.fingerprint,
            "run_fingerprint": run_fingerprint,
            "fusion_config": config_to_dict(config.fusion),
            "training_config": config_to_dict(config.training),
            "frozen_final_cls_baseline": {
                "recall_at_1": baseline_report.recall_at_1,
                "recall_at_5": baseline_report.recall_at_5,
                "recall_at_10": baseline_report.recall_at_10,
                "mrr": baseline_report.mrr,
            },
            "history": {
                "epochs": history.epochs,
                "best_epoch": history.best_epoch,
                "best_metric": history.best_metric,
                "best_checkpoint": None
                if history.best_checkpoint is None
                else str(history.best_checkpoint),
            },
        },
    )
    print(
        f"Best epoch={history.best_epoch}, "
        f"validation R@1={history.best_metric:.4f}, "
        f"checkpoint={history.best_checkpoint}"
    )


def _require_path(value: Path | None, name: str) -> Path:
    if value is None:
        raise ValueError(f"config.sfm.{name} is required")
    return value


if __name__ == "__main__":
    main()
