from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cbir.cloud import RuntimePaths, publish_file, runtime_report, stage_file, write_runtime_report
from cbir.utils import read_json


class CloudFileTests(unittest.TestCase):
    def test_stage_file_copies_once_and_keeps_an_existing_local_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            persistent = root / "drive" / "metadata.pkl"
            persistent.parent.mkdir(parents=True)
            persistent.write_bytes(b"first durable version")
            local = root / "runtime" / "metadata.pkl"

            self.assertEqual(stage_file(persistent, local), local)
            self.assertEqual(local.read_bytes(), b"first durable version")
            self.assertFalse((local.parent / ".metadata.pkl.part").exists())

            # Staging is deliberately local-first: a notebook can safely call
            # it repeatedly without re-copying a completed runtime input.
            persistent.write_bytes(b"new durable version")
            self.assertEqual(stage_file(persistent, local), local)
            self.assertEqual(local.read_bytes(), b"first durable version")

    def test_stage_file_requires_a_durable_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                stage_file(root / "missing.bin", root / "runtime" / "input.bin")

    def test_publish_file_replaces_the_completed_durable_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "runtime" / "checkpoint.pt"
            local.parent.mkdir(parents=True)
            local.write_bytes(b"new completed checkpoint")
            persistent = root / "drive" / "checkpoint.pt"
            persistent.parent.mkdir(parents=True)
            persistent.write_bytes(b"old checkpoint")

            self.assertEqual(publish_file(local, persistent), persistent)
            self.assertEqual(persistent.read_bytes(), b"new completed checkpoint")
            self.assertFalse((persistent.parent / ".checkpoint.pt.part").exists())

    def test_publish_file_requires_a_local_completed_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                publish_file(root / "missing.pt", root / "drive" / "checkpoint.pt")


class RuntimeProvenanceTests(unittest.TestCase):
    def test_runtime_paths_and_report_are_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RuntimePaths(
                platform="local",
                project_root=root / "project",
                runtime_root=root / "runtime",
                persistent_root=root / "drive",
            )
            paths.ensure_local_roots()
            self.assertTrue(paths.local_data_root.is_dir())
            self.assertTrue(paths.local_cache_root.is_dir())
            self.assertTrue(paths.local_output_root.is_dir())

            report = runtime_report(project_root=root, extra={"notebook": "02"})
            self.assertEqual(report["project_root"], str(root.resolve()))
            self.assertEqual(report["extra"], {"notebook": "02"})
            self.assertIn("python_version", report)
            self.assertIn("cuda_available", report)

            output = write_runtime_report(
                paths.local_output_root,
                project_root=root,
                extra={"notebook": "02"},
            )
            self.assertEqual(output.name, "runtime_environment.json")
            self.assertEqual(read_json(output)["extra"], {"notebook": "02"})


if __name__ == "__main__":
    unittest.main()
