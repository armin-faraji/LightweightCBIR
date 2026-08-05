from __future__ import annotations

import tempfile
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cbir.artifacts import (
    ARTIFACT_COMPLETE_NAME,
    ARTIFACT_MANIFEST_NAME,
    create_artifact_run,
    make_artifact_run_id,
    publish_artifact_directory,
    validate_artifact_directory,
)
from cbir.utils import atomic_write_json, read_json


class ArtifactRunTests(unittest.TestCase):
    def test_publish_finalizes_a_new_run_with_its_own_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = create_artifact_run(root / "outputs", "02", run_id="new-run")
            run.write_json("metrics.json", {"ok": True})

            destination = run.publish(root / "drive")

            self.assertEqual(destination, root / "drive" / "02" / "new-run")
            report = validate_artifact_directory(destination)
            self.assertTrue(report["valid"], report["errors"])

    def test_finalize_publish_and_idempotent_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = create_artifact_run(
                root / "outputs",
                "02",
                run_id="pilot-cache-1234",
                metadata={"git_sha": "abc123", "cache_fingerprint": "f" * 64},
            )
            run.write_json("metrics/pilot.json", {"temperature": 0.1, "score": 0.4})
            run.path_for("plots/entropy.png").write_bytes(b"not-a-real-png-but-a-report-file")

            manifest = run.finalize()
            local = validate_artifact_directory(run.local_dir)
            self.assertTrue(local["valid"], local["errors"])
            self.assertEqual(local["manifest"]["fingerprint"], manifest["fingerprint"])
            self.assertTrue((run.local_dir / ARTIFACT_MANIFEST_NAME).is_file())
            self.assertTrue((run.local_dir / ARTIFACT_COMPLETE_NAME).is_file())

            destination = run.publish(root / "drive" / "notebook_outputs")
            self.assertEqual(destination, root / "drive" / "notebook_outputs" / "02" / run.run_id)
            published = validate_artifact_directory(destination)
            self.assertTrue(published["valid"], published["errors"])
            self.assertEqual(published["manifest"]["fingerprint"], manifest["fingerprint"])

            # A retry after a completed copy must not overwrite or duplicate it.
            self.assertEqual(run.publish(root / "drive" / "notebook_outputs"), destination)

            # A notebook can make a new plot/metric after the first publish.
            # Preserve the old immutable report and publish the changed content
            # to a deterministic revision rather than raising an error.
            run.write_json("metrics/pilot.json", {"temperature": 0.2, "score": 0.5})
            revision = run.publish(root / "drive" / "notebook_outputs")
            self.assertNotEqual(revision, destination)
            self.assertTrue(revision.name.startswith(f"{run.run_id}__rev-"))
            self.assertTrue(validate_artifact_directory(revision)["valid"])
            self.assertEqual(
                validate_artifact_directory(destination)["manifest"]["fingerprint"],
                manifest["fingerprint"],
            )

    def test_rejects_unsafe_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = create_artifact_run(Path(temporary), "03", run_id="safe-run")
            with self.assertRaises(ValueError):
                run.path_for("../escape.json")
            with self.assertRaises(ValueError):
                run.path_for(ARTIFACT_MANIFEST_NAME)

    def test_detects_modification_after_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = create_artifact_run(Path(temporary), "04", run_id="final-run")
            path = run.path_for("report.json")
            path.write_text('{"value": 1}\n', encoding="utf-8")
            run.finalize()
            path.write_text('{"value": 2}\n', encoding="utf-8")
            report = validate_artifact_directory(run.local_dir)
            self.assertFalse(report["valid"])
            self.assertTrue(any("checksum mismatch" in error for error in report["errors"]))

    def test_finalize_repairs_legacy_unsorted_local_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = create_artifact_run(Path(temporary), "04", run_id="legacy-run")
            run.write_json("a.json", {"a": 1})
            run.write_json("z.json", {"z": 1})
            run.finalize()
            manifest_path = run.local_dir / ARTIFACT_MANIFEST_NAME
            legacy_manifest = read_json(manifest_path)
            legacy_manifest["files"] = list(reversed(legacy_manifest["files"]))
            atomic_write_json(manifest_path, legacy_manifest)

            run.finalize()

            report = validate_artifact_directory(run.local_dir)
            self.assertTrue(report["valid"], report["errors"])

    def test_requires_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = create_artifact_run(Path(temporary), "02", run_id="marker-run")
            run.write_json("report.json", {"ok": True})
            run.finalize()
            (run.local_dir / ARTIFACT_COMPLETE_NAME).unlink()
            report = validate_artifact_directory(run.local_dir)
            self.assertFalse(report["valid"])
            self.assertTrue(any("COMPLETE" in error for error in report["errors"]))

    def test_cleans_old_hidden_publish_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = create_artifact_run(root / "outputs", "02", run_id="stale-run")
            run.write_json("report.json", {"ok": True})
            run.finalize()
            parent = root / "drive" / "02"
            stale = parent / ".stale-run.publishing-abandoned"
            stale.mkdir(parents=True)
            old = time.time() - 5
            os.utime(stale, (old, old))
            publish_artifact_directory(
                run.local_dir,
                root / "drive",
                stale_publish_seconds=0,
            )
            self.assertFalse(stale.exists())

    def test_readable_run_id_contains_requested_provenance(self) -> None:
        run_id = make_artifact_run_id(
            notebook="03",
            git_sha="AbCdEf1234567890",
            config_fingerprint="c" * 64,
            cache_fingerprint="d" * 64,
            timestamp=datetime(2026, 8, 2, 12, 15, tzinfo=timezone.utc),
        )
        self.assertEqual(
            run_id,
            "20260802T121500Z__nb-03__git-abcdef123456__cfg-cccccccccccc__cache-dddddddddddd",
        )


if __name__ == "__main__":
    unittest.main()
