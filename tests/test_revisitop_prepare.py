from __future__ import annotations

import pickle
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from cbir.data.revisitop_prepare import (
    _extract_expected_images_from_archive,
    prepare_revisitop_datasets,
    validate_revisitop_dataset,
)


class RevisitOPPreparationTests(unittest.TestCase):
    def test_uses_local_archive_and_checks_filenames_without_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = root / "archives"
            archives.mkdir()
            _write_archive(archives / "oxbuild_images-v1.tgz", {"a.jpg": b"not-a-jpeg", "b.jpg": b"also-not-a-jpeg"})

            calls: list[Path] = []

            def fake_download(_: str, destination: Path) -> Path:
                calls.append(destination)
                _write_annotation(destination)
                return destination

            with patch("cbir.data.revisitop_prepare.download_with_resume", fake_download):
                report = prepare_revisitop_datasets(
                    root / "prepared",
                    archives_root=archives,
                    datasets=("roxford5k",),
                    enforce_official_counts=False,
                )
            self.assertEqual(len(calls), 1)  # Annotation only; archive was local.
            self.assertEqual(report["datasets"]["roxford5k"]["status"], "prepared")
            self.assertTrue(
                validate_revisitop_dataset(
                    root / "prepared" / "roxford5k",
                    "roxford5k",
                    enforce_official_counts=False,
                )["valid"]
            )

    def test_downloads_only_a_missing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archives = root / "archives"
            calls: list[Path] = []

            def fake_download(url: str, destination: Path) -> Path:
                calls.append(destination)
                if destination.suffix == ".pkl":
                    _write_annotation(destination)
                else:
                    _write_archive(destination, {"a.jpg": b"a", "b.jpg": b"b"})
                return destination

            with patch("cbir.data.revisitop_prepare.download_with_resume", fake_download):
                prepare_revisitop_datasets(
                    root / "prepared",
                    archives_root=archives,
                    datasets=("roxford5k",),
                    enforce_official_counts=False,
                )
            self.assertEqual(len(calls), 2)
            self.assertTrue((archives / "oxbuild_images-v1.tgz").is_file())

    def test_archive_member_path_cannot_escape_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "images.tgz"
            _write_archive(archive, {"../../escape.jpg": b"unsafe", "nested/expected.jpg": b"expected"})
            output = root / "jpg"
            _extract_expected_images_from_archive(archive, output, ("expected",))
            self.assertEqual((output / "expected.jpg").read_bytes(), b"expected")
            self.assertFalse((root / "escape.jpg").exists())


def _write_annotation(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "imlist": ["a", "b"],
        "qimlist": ["a"],
        "gnd": [{"bbx": [0, 0, 1, 1], "easy": [0], "hard": [1], "junk": []}],
    }
    with path.open("wb") as handle:
        pickle.dump(raw, handle)


def _write_archive(path: Path, payloads: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in payloads.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, BytesIO(payload))


if __name__ == "__main__":
    unittest.main()
