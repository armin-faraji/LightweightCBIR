from __future__ import annotations

import pickle
import tempfile
import tarfile
import unittest
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cbir.data.revisitop_prepare import (
    _extract_expected_images_from_archive,
    _download_annotation_with_repair,
    _download_archive_with_repair,
    prepare_revisitop_datasets,
    publish_revisitop_datasets,
    validate_revisitop_dataset,
)


class RevisitOPPreparationTests(unittest.TestCase):
    def test_default_validation_rejects_nonofficial_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "roxford5k"
            _write_toy_roxford(source)
            strict = validate_revisitop_dataset(source, "roxford5k")
            self.assertFalse(strict["valid"])
            self.assertTrue(any("official count mismatch" in error for error in strict["errors"]))
            relaxed = validate_revisitop_dataset(
                source,
                "roxford5k",
                enforce_official_counts=False,
            )
            self.assertTrue(relaxed["valid"], relaxed["errors"])

    def test_stages_existing_dataset_in_evaluator_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "drive" / "datasets" / "roxford5k"
            _write_toy_roxford(source)

            output_root = root / "content" / "revisitop"
            report = prepare_revisitop_datasets(
                output_root,
                datasets=("roxford5k",),
                source_root=root / "drive",
                mode="stage",
                enforce_official_counts=False,
            )
            self.assertEqual(report["datasets"]["roxford5k"]["status"], "prepared")
            target = output_root / "roxford5k"
            validation = validate_revisitop_dataset(
                target,
                "roxford5k",
                enforce_official_counts=False,
            )
            self.assertTrue(validation["valid"], validation["errors"])
            self.assertEqual(validation["database_image_count"], 2)
            self.assertEqual(validation["query_count"], 1)
            self.assertFalse((output_root / ".roxford5k.staging").exists())

            # It is safe to call from a fresh notebook runtime when a full local
            # staging already exists.
            repeat = prepare_revisitop_datasets(
                output_root,
                datasets=("roxford5k",),
                source_root=root / "drive",
                mode="stage",
                enforce_official_counts=False,
            )
            self.assertEqual(repeat["datasets"]["roxford5k"]["status"], "already_prepared")

    def test_refuses_to_replace_invalid_visible_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "roxford5k"
            _write_toy_roxford(source)
            invalid_target = root / "output" / "roxford5k"
            invalid_target.mkdir(parents=True)
            with self.assertRaises(RuntimeError):
                prepare_revisitop_datasets(
                    root / "output",
                    datasets=("roxford5k",),
                    source_root=root / "source",
                    mode="stage",
                    enforce_official_counts=False,
                )

    def test_archive_member_path_cannot_escape_staging_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "images.tgz"
            with tarfile.open(archive_path, "w:gz") as archive:
                _add_tar_member(archive, "../../escape.jpg", b"unsafe")
                _add_tar_member(archive, "nested/expected.jpg", b"expected")
            output = root / "jpg"
            _extract_expected_images_from_archive(
                archive_path,
                output,
                ["expected"],
                verify_existing=False,
            )
            self.assertEqual((output / "expected.jpg").read_bytes(), b"expected")
            self.assertFalse((root / "escape.jpg").exists())

    def test_publish_to_persistent_root_uses_same_validated_stage_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "content" / "revisitop" / "roxford5k"
            _write_toy_roxford(local)
            persistent = root / "drive" / "datasets" / "revisitop"
            report = publish_revisitop_datasets(
                local.parent,
                persistent,
                datasets=("roxford5k",),
                enforce_official_counts=False,
            )
            self.assertEqual(report["datasets"]["roxford5k"]["status"], "prepared")
            self.assertTrue(
                validate_revisitop_dataset(
                    persistent / "roxford5k",
                    "roxford5k",
                    enforce_official_counts=False,
                )["valid"]
            )

    def test_refuses_symlinked_hidden_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "roxford5k"
            _write_toy_roxford(source)
            output = root / "output"
            output.mkdir()
            outside = root / "outside"
            outside.mkdir()
            os.symlink(outside, output / ".roxford5k.staging", target_is_directory=True)
            with self.assertRaises(RuntimeError):
                prepare_revisitop_datasets(
                    output,
                    datasets=("roxford5k",),
                    source_root=root / "source",
                    mode="stage",
                    enforce_official_counts=False,
                )

    def test_repair_replaces_corrupt_owned_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "download.tgz"
            archive_path.write_bytes(b"not-a-tar-file")
            calls = 0

            def fake_download(url: str, destination: Path) -> Path:
                nonlocal calls
                calls += 1
                if calls == 1:
                    # Mimic download_with_resume returning an existing corrupt
                    # complete filename without rewriting it.
                    return destination
                with tarfile.open(destination, "w:gz"):
                    pass
                return destination

            with patch("cbir.data.revisitop_prepare.download_with_resume", fake_download):
                _download_archive_with_repair(
                    "https://example.invalid/archive.tgz",
                    archive_path,
                    repair=True,
                )
            self.assertEqual(calls, 2)
            with tarfile.open(archive_path, "r:gz"):
                pass

    def test_repair_replaces_corrupt_owned_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            annotation_path = Path(temporary) / "gnd_roxford5k.pkl"
            annotation_path.write_bytes(b"not-a-pickle")
            calls = 0

            def fake_download(url: str, destination: Path) -> Path:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return destination
                raw = {
                    "imlist": ["a"],
                    "qimlist": ["a"],
                    "gnd": [{"bbx": [0, 0, 1, 1], "easy": ["a"], "hard": [], "junk": []}],
                }
                with destination.open("wb") as handle:
                    pickle.dump(raw, handle)
                return destination

            with patch("cbir.data.revisitop_prepare.download_with_resume", fake_download):
                dataset = _download_annotation_with_repair(
                    "https://example.invalid/gnd.pkl",
                    annotation_path,
                    "roxford5k",
                    repair=True,
                )
            self.assertEqual(calls, 2)
            self.assertEqual(dataset.database_ids, ("a",))


def _write_toy_roxford(dataset_dir: Path) -> None:
    image_root = dataset_dir / "jpg"
    image_root.mkdir(parents=True)
    raw = {
        "imlist": ["building_a", "building_b"],
        "qimlist": ["building_a"],
        "gnd": [
            {
                "bbx": [0, 0, 4, 4],
                "easy": ["building_a"],
                "hard": ["building_b"],
                "junk": [],
            }
        ],
    }
    with (dataset_dir / "gnd_roxford5k.pkl").open("wb") as handle:
        pickle.dump(raw, handle)
    Image.new("RGB", (8, 8), color=(200, 10, 20)).save(image_root / "building_a.jpg")
    Image.new("RGB", (8, 8), color=(10, 200, 20)).save(image_root / "building_b.jpg")


def _add_tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, BytesIO(payload))


if __name__ == "__main__":
    unittest.main()
