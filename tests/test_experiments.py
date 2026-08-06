from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cbir.experiments import matching_experiment, remove_experiment, save_experiment


class ExperimentRecordTests(unittest.TestCase):
    def test_reuse_requires_matching_spec_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results.json"
            checkpoint = root / "checkpoints" / "trial" / "best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            save_experiment(
                results,
                name="trial",
                run_spec={"dimension": 128},
                summary={"recall_at_1": 0.9},
                history=[],
                checkpoint=checkpoint,
            )
            self.assertIsNotNone(matching_experiment(results, "trial", {"dimension": 128}))
            self.assertIsNone(matching_experiment(results, "trial", {"dimension": 64}))
            remove_experiment(results, "trial")
            self.assertIsNone(matching_experiment(results, "trial", {"dimension": 128}))
            self.assertFalse(checkpoint.parent.exists())
