"""Local preparation of the official Revisited Oxford and Paris datasets."""

from __future__ import annotations

import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from .download import download_with_resume
from .revisitop import RevisitOPDataset


REVISITOP_DATASETS = ("roxford5k", "rparis6k")
OFFICIAL_REVISITOP_COUNTS = {
    "roxford5k": {"database_image_count": 4993, "query_count": 70},
    "rparis6k": {"database_image_count": 6322, "query_count": 70},
}


@dataclass(frozen=True)
class RevisitOPUrls:
    """Official sources for the archive files and annotation pickles."""

    oxford_archives: tuple[str, ...] = (
        "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images-v1.tgz",
    )
    paris_archives: tuple[str, ...] = (
        "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1-v1.tgz",
        "https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2-v1.tgz",
    )
    annotations_base_url: str = "https://cmp.felk.cvut.cz/revisitop/data/datasets"

    def annotation_url(self, dataset: str) -> str:
        _validate_dataset_name(dataset)
        return f"{self.annotations_base_url}/{dataset}/gnd_{dataset}.pkl"

    def archive_urls(self, dataset: str) -> tuple[str, ...]:
        _validate_dataset_name(dataset)
        return self.oxford_archives if dataset == "roxford5k" else self.paris_archives


def prepare_revisitop_datasets(
    output_root: Path,
    *,
    archives_root: Path | None = None,
    datasets: Sequence[str] = REVISITOP_DATASETS,
    urls: RevisitOPUrls | None = None,
    enforce_official_counts: bool = True,
) -> dict[str, Any]:
    """Use local archives when present and download only missing source files.

    The output layout is ``<output_root>/<dataset>/{gnd_*.pkl,jpg/*.jpg}``.
    Archives remain in ``archives_root`` and are never deleted.  Readiness is a
    parsed annotation plus the expected filenames; JPEGs are not decoded here.
    """
    output_root = Path(output_root)
    archive_root = Path(archives_root) if archives_root is not None else output_root / "archives"
    output_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    source_urls = urls or RevisitOPUrls()
    reports: dict[str, Any] = {}

    for dataset in _normalise_datasets(datasets):
        target = output_root / dataset
        existing = validate_revisitop_dataset(
            target,
            dataset,
            enforce_official_counts=enforce_official_counts,
        )
        if existing["valid"]:
            reports[dataset] = {"status": "already_prepared", **existing}
            continue

        target.mkdir(parents=True, exist_ok=True)
        image_root = target / "jpg"
        image_root.mkdir(parents=True, exist_ok=True)
        annotation = target / f"gnd_{dataset}.pkl"
        if not annotation.is_file():
            download_with_resume(source_urls.annotation_url(dataset), annotation)
        descriptor = RevisitOPDataset.from_ground_truth_pickle(
            name=dataset,
            ground_truth_path=annotation,
            image_root=image_root,
        )
        if enforce_official_counts:
            _assert_official_counts(descriptor, dataset)
        image_ids = _required_image_ids(descriptor)
        for url in source_urls.archive_urls(dataset):
            archive = archive_root / Path(url).name
            if not archive.is_file():
                download_with_resume(url, archive)
            _extract_expected_images_from_archive(archive, image_root, image_ids)

        validation = validate_revisitop_dataset(
            target,
            dataset,
            enforce_official_counts=enforce_official_counts,
        )
        if not validation["valid"]:
            raise RuntimeError(f"prepared {dataset} is incomplete: {validation['errors'][:3]}")
        reports[dataset] = {"status": "prepared", **validation}

    return {
        "output_root": str(output_root),
        "archives_root": str(archive_root),
        "datasets": reports,
    }


def validate_revisitop_dataset(
    dataset_dir: Path,
    dataset: str,
    *,
    enforce_official_counts: bool = True,
) -> dict[str, Any]:
    """Check the annotation, official counts, and required file names only."""
    _validate_dataset_name(dataset)
    dataset_dir = Path(dataset_dir)
    annotation = dataset_dir / f"gnd_{dataset}.pkl"
    image_root = dataset_dir / "jpg"
    errors: list[str] = []
    if not annotation.is_file():
        errors.append(f"missing annotation: {annotation.name}")
    if not image_root.is_dir():
        errors.append("missing jpg directory")
    if errors:
        return {"valid": False, "errors": errors, "dataset": dataset}
    try:
        descriptor = RevisitOPDataset.from_ground_truth_pickle(
            name=dataset,
            ground_truth_path=annotation,
            image_root=image_root,
        )
    except Exception as error:
        return {"valid": False, "errors": [f"invalid annotation: {error}"], "dataset": dataset}

    expected_counts = OFFICIAL_REVISITOP_COUNTS[dataset]
    official_counts_match = {
        "database_image_count": len(descriptor.database_ids),
        "query_count": len(descriptor.queries),
    } == expected_counts
    if enforce_official_counts:
        try:
            _assert_official_counts(descriptor, dataset)
        except ValueError as error:
            errors.append(str(error))
    image_ids = _required_image_ids(descriptor)
    missing = [
        _image_filename(image_id)
        for image_id in image_ids
        if not (image_root / _image_filename(image_id)).is_file()
    ]
    if missing:
        errors.append(f"missing {len(missing)} image files: {missing[:5]}")
    return {
        "valid": not errors,
        "errors": errors,
        "dataset": dataset,
        "database_image_count": len(descriptor.database_ids),
        "query_count": len(descriptor.queries),
        "required_image_count": len(image_ids),
        "official_counts_match": official_counts_match,
    }


def _extract_expected_images_from_archive(
    archive_path: Path,
    destination_root: Path,
    image_ids: Iterable[str],
) -> None:
    """Extract expected basenames only; never trust archive paths as output paths."""
    expected = {_image_filename(image_id) for image_id in image_ids}
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError(f"could not open RevisitOP archive {archive_path}: {error}") from error
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            filename = PurePosixPath(member.name).name
            if filename not in expected:
                continue
            destination = destination_root / filename
            if destination.is_file():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member {member.name}")
            _copy_stream_atomic(source, destination)


def _required_image_ids(dataset: RevisitOPDataset) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*dataset.database_ids, *(query.source_image_id for query in dataset.queries))))


def _assert_official_counts(dataset: RevisitOPDataset, name: str) -> None:
    expected = OFFICIAL_REVISITOP_COUNTS[name]
    actual = {"database_image_count": len(dataset.database_ids), "query_count": len(dataset.queries)}
    if actual != expected:
        raise ValueError(f"official count mismatch for {name}: expected {expected}, got {actual}")


def _image_filename(image_id: str) -> str:
    value = str(image_id)
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name in {".", ".."}
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe RevisitOP image ID: {image_id!r}")
    return f"{candidate.name}.jpg"


def _copy_stream_atomic(source: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            while chunk := source.read(1024 * 1024):
                handle.write(chunk)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        source.close()


def _normalise_datasets(datasets: Sequence[str]) -> tuple[str, ...]:
    if not datasets:
        raise ValueError("at least one RevisitOP dataset is required")
    result = tuple(dict.fromkeys(str(dataset).lower() for dataset in datasets))
    for dataset in result:
        _validate_dataset_name(dataset)
    return result


def _validate_dataset_name(dataset: str) -> None:
    if dataset not in REVISITOP_DATASETS:
        raise ValueError(f"unsupported RevisitOP dataset {dataset!r}")
