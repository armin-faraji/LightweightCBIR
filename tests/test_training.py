from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cbir.config import FusionConfig, TrainingConfig
from cbir.data.sfm import PairRecord
from cbir.fusion import MultiLevelGlobalLocalFusion, build_descriptor_head
from cbir.training import (
    ClusterUniquePairBatchSampler,
    HeadTrainer,
    load_descriptor_head,
    symmetric_info_nce,
)


class TrainingTests(unittest.TestCase):
    def test_sampler_has_unique_clusters_in_each_batch(self) -> None:
        pairs = tuple(
            PairRecord(f"q{index}", f"p{index}", index % 3, "train")
            for index in range(12)
        )
        sampler = ClusterUniquePairBatchSampler(pairs, batch_size=3, seed=3)
        for batch in sampler:
            clusters = [pairs[item.pair_index].cluster_id for item in batch]
            self.assertEqual(len(clusters), len(set(clusters)))
            self.assertGreaterEqual(len(batch), 2)

    def test_symmetric_infonce_is_finite(self) -> None:
        query = torch.eye(3)
        loss = symmetric_info_nce(query, query, temperature=0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_best_checkpoint_tracks_improvement(self) -> None:
        fusion_config = FusionConfig(
            token_dim=4,
            layer_indices=(0, 1),
            output_dim=3,
            head_kind="global_local",
            gate_mode="dynamic",
        )
        training_config = TrainingConfig(
            batch_size=2,
            epochs=3,
            early_stopping_patience=3,
            device="cpu",
        )
        head = MultiLevelGlobalLocalFusion.from_config(fusion_config)
        reader = SimpleNamespace(
            manifest=SimpleNamespace(
                fingerprint="cache-fingerprint",
                extraction_config={"backbone": "tiny"},
            )
        )
        pairs = (
            PairRecord("q0", "p0", 0, "train"),
            PairRecord("q1", "p1", 1, "train"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            trainer = HeadTrainer(
                head=head,
                reader=reader,  # type: ignore[arg-type]
                train_pairs=pairs,
                fusion_config=fusion_config,
                training_config=training_config,
                output_dir=Path(temporary),
            )
            metrics = iter(
                (
                    {"loss": 3.0, "entropy_penalty_scale": 0.1, "learning_rate": 1e-3},
                    {"loss": 2.0, "entropy_penalty_scale": 0.1, "learning_rate": 1e-3},
                    {"loss": 2.5, "entropy_penalty_scale": 0.1, "learning_rate": 1e-3},
                )
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with patch.object(
                    trainer,
                    "_train_epoch",
                    side_effect=lambda _loader: next(metrics),
                ):
                    history = trainer.fit()

            self.assertEqual(history.best_epoch, 1)
            self.assertEqual(history.best_metric, -2.0)
            self.assertEqual(history.best_checkpoint, Path(temporary) / "best.pt")
            payload = torch.load(history.best_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 1)

    def test_cls_concat_head_uses_the_common_training_pipeline(self) -> None:
        """CLS-only ablations must not need a separate trainer or cache format."""

        torch.manual_seed(19)
        fusion_config = FusionConfig(
            token_dim=4,
            layer_indices=(0, 1),
            output_dim=3,
            head_kind="cls_concat",
            gate_mode=None,
        )
        training_config = TrainingConfig(
            batch_size=2,
            epochs=1,
            early_stopping_patience=1,
            device="cpu",
        )
        feature_by_id = {
            image_id: {
                "cls": torch.randn(2, 4),
                "local": torch.randn(2, 4),
                "entropy": torch.rand(2),
            }
            for image_id in ("q0", "p0", "q1", "p1")
        }

        class TinyReader:
            manifest = SimpleNamespace(
                fingerprint="cache-fingerprint",
                extraction_config={"backbone": "tiny"},
            )

            @staticmethod
            def fetch(image_ids, *, layer_indices, local_kind):
                self.assertEqual(tuple(layer_indices), (0, 1))
                self.assertEqual(local_kind, "cls_guided_patch")
                return {
                    name: torch.stack([feature_by_id[image_id][name] for image_id in image_ids])
                    for name in ("cls", "local", "entropy")
                }

        pairs = (
            PairRecord("q0", "p0", 0, "train"),
            PairRecord("q1", "p1", 1, "train"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            trainer = HeadTrainer(
                head=build_descriptor_head(fusion_config),
                reader=TinyReader(),  # type: ignore[arg-type]
                train_pairs=pairs,
                fusion_config=fusion_config,
                training_config=training_config,
                output_dir=Path(temporary),
            )
            history = trainer.fit()

            self.assertEqual(history.best_epoch, 0)
            self.assertIsNotNone(history.best_checkpoint)
            checkpoint = torch.load(history.best_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["fusion_config"]["head_kind"], "cls_concat")
            self.assertIsNone(checkpoint["fusion_config"]["gate_mode"])
            loaded_head, loaded_payload = load_descriptor_head(history.best_checkpoint)
            self.assertEqual(loaded_head.config.output_dim, 3)
            self.assertEqual(loaded_payload["epoch"], 0)

    def test_final_checkpoint_uses_the_last_requested_epoch(self) -> None:
        fusion_config = FusionConfig(
            token_dim=4,
            layer_indices=(0, 1),
            output_dim=3,
            head_kind="global_local",
            gate_mode="dynamic",
        )
        training_config = TrainingConfig(
            batch_size=2,
            epochs=3,
            early_stopping_patience=1,
            device="cpu",
        )
        reader = SimpleNamespace(
            manifest=SimpleNamespace(
                fingerprint="cache-fingerprint",
                extraction_config={"backbone": "tiny"},
            )
        )
        pairs = (
            PairRecord("q0", "p0", 0, "train"),
            PairRecord("q1", "p1", 1, "train"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            trainer = HeadTrainer(
                head=MultiLevelGlobalLocalFusion.from_config(fusion_config),
                reader=reader,  # type: ignore[arg-type]
                train_pairs=pairs,
                fusion_config=fusion_config,
                training_config=training_config,
                output_dir=Path(temporary),
            )
            metrics = iter(
                {
                    "loss": loss,
                    "entropy_penalty_scale": 0.0,
                    "learning_rate": 1e-3,
                }
                for loss in (1.0, 2.0)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with patch.object(
                    trainer,
                    "_train_epoch",
                    side_effect=lambda _loader: next(metrics),
                ):
                    history = trainer.fit(
                        max_epochs=2,
                        enable_early_stopping=False,
                        save_best_checkpoint=False,
                    )
            final_checkpoint = trainer.save_final_checkpoint(history)
            payload = torch.load(final_checkpoint, map_location="cpu", weights_only=False)

            self.assertEqual(len(history.epochs), 2)
            self.assertIsNone(history.best_checkpoint)
            self.assertFalse((Path(temporary) / "best.pt").exists())
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["checkpoint_kind"], "final")


if __name__ == "__main__":
    unittest.main()
