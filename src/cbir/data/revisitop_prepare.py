"""Safe, reproducible preparation of the official Revisited Oxford/Paris data.

The RevisitOP annotations and original Oxford/Paris images are hosted separately.
This module creates the single layout consumed by :class:`RevisitOPDataset`::

    <root>/roxford5k/gnd_roxford5k.pkl
    <root>/roxford5k/jpg/<image-id>.jpg
    <root>/rparis6k/gnd_rparis6k.pkl
    <root>/rparis6k/jpg/<image-id>.jpg

It can either stage an existing trusted copy (for example, from Google Drive) or
download the official archives.  Dataset directories are built in a hidden staging
directory and become visible only after their metadata and images validate.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal, Sequence

from PIL import Image

from ..utils import atomic_write_json
from .download import download_with_resume
from .revisitop import RevisitOPDataset


REVISITOP_DATASETS = ("roxford5k", "rparis6k")
OFFICIAL_REVISITOP_COUNTS = {
    "roxford5k": {"database_image_count": 4993, "query_count": 70},
    "rparis6k": {"database_image_count": 6322, "query_count": 70},
}
PrepareMode = Literal["auto", "stage", "download"]


@dataclass(frozen=True)
class RevisitOPUrls:
    """Official source URLs used by the reference RevisitOP Python downloader."""

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
    datasets: Sequence[str] = REVISITOP_DATASETS,
    source_root: Path | None = None,
    mode: PrepareMode = "auto",
    keep_archives: bool = False,
    verify_images: bool = True,
    repair: bool = False,
    enforce_official_counts: bool = True,
    urls: RevisitOPUrls | None = None,
) -> dict[str, Any]:
    """Stage or download complete RevisitOP datasets into ``output_root``.

    Args:
        output_root: Parent directory that will contain ``roxford5k`` and/or
            ``rparis6k``.  It is normally a fast local runtime path such as
            ``/content/cbir_data/revisitop``.
        datasets: One or both official dataset names.
        source_root: Optional existing trusted source.  Accepted layouts are
            ``<source>/<dataset>`` and ``<source>/datasets/<dataset>``.
        mode: ``stage`` requires ``source_root``; ``download`` uses only official
            URLs; ``auto`` stages when available and downloads a missing dataset.
        keep_archives: Retain completed official archives in
            ``<output_root>/.downloads`` after successful preparation.
        verify_images: Decode-check every required JPEG before publication.
        repair: Permit replacement of corrupt, module-owned downloaded annotation
            and archive files.  It never alters a user-supplied existing source.
        enforce_official_counts: Require the official 4,993/70 and 6,322/70
            ROxford/RParis database/query counts.  Disable only for a controlled
            synthetic test fixture.

    Returns a JSON-friendly report.  Existing valid target datasets are left alone,
    so calling this function is safe and idempotent.
    """
    if mode not in {"auto", "stage", "download"}:
        raise ValueError(f"unsupported RevisitOP preparation mode: {mode}")
    selected = _normalise_datasets(datasets)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = None if source_root is None else Path(source_root)
    if mode == "stage" and source_root is None:
        raise ValueError("--mode stage requires source_root")
    if source_root is not None and not source_root.is_dir() and mode == "stage":
        raise FileNotFoundError(f"RevisitOP source_root does not exist: {source_root}")
    source_urls = urls or RevisitOPUrls()

    reports: dict[str, Any] = {}
    for dataset in selected:
        target_dir = output_root / dataset
        existing = validate_revisitop_dataset(
            target_dir,
            dataset,
            verify_images=verify_images,
            enforce_official_counts=enforce_official_counts,
        )
        if existing["valid"]:
            reports[dataset] = {"status": "already_prepared", **existing}
            continue
        if target_dir.is_symlink():
            raise RuntimeError(f"refusing to use a symlinked RevisitOP target: {target_dir}")
        if target_dir.exists():
            raise RuntimeError(
                f"target RevisitOP dataset exists but is invalid: {target_dir}; "
                f"errors: {existing['errors'][:3]}. Refusing to overwrite it."
            )

        source_dataset = (
            _find_source_dataset(source_root, dataset) if source_root is not None else None
        )
        if mode == "stage":
            if source_dataset is None:
                raise FileNotFoundError(
                    f"could not find a {dataset} source below {source_root}"
                )
            chosen_mode = "stage"
        elif mode == "download":
            chosen_mode = "download"
        else:
            chosen_mode = "stage" if source_dataset is not None else "download"

        stage_dir = output_root / f".{dataset}.staging"
        _ensure_owned_directory(stage_dir)
        try:
            if chosen_mode == "stage":
                assert source_dataset is not None
                source_report = _stage_from_existing(
                    source_dataset,
                    stage_dir,
                    dataset,
                    enforce_official_counts=enforce_official_counts,
                )
            else:
                source_report = _stage_from_official_downloads(
                    output_root,
                    stage_dir,
                    dataset,
                    urls=source_urls,
                    verify_images=verify_images,
                    repair=repair,
                    enforce_official_counts=enforce_official_counts,
                )
            validation = validate_revisitop_dataset(
                stage_dir,
                dataset,
                verify_images=verify_images,
                enforce_official_counts=enforce_official_counts,
            )
            if not validation["valid"]:
                raise RuntimeError(
                    f"prepared {dataset} staging directory failed validation: "
                    f"{validation['errors'][:5]}"
                )
            atomic_write_json(
                stage_dir / "preparation_report.json",
                {
                    "dataset": dataset,
                    "mode": chosen_mode,
                    "output_root": str(output_root),
                    "source": source_report,
                    "validation": validation,
                },
            )
            # The staging and target paths are siblings on the same filesystem.
            # A target is visible only after all required images are present.
            os.replace(stage_dir, target_dir)
        except Exception:
            # Preserve the deterministic hidden staging directory and partial
            # downloads for a later retry; never delete user-visible data here.
            raise

        completed = validate_revisitop_dataset(
            target_dir,
            dataset,
            verify_images=verify_images,
            enforce_official_counts=enforce_official_counts,
        )
        if not completed["valid"]:
            raise RuntimeError(
                f"{dataset} failed validation after publication: {completed['errors'][:5]}"
            )
        if chosen_mode == "download" and not keep_archives:
            _remove_completed_archives(output_root, dataset, source_urls)
        reports[dataset] = {"status": "prepared", **completed, "source": source_report}

    report = {
        "output_root": str(output_root),
        "source_root": None if source_root is None else str(source_root),
        "mode": mode,
        "keep_archives": keep_archives,
        "repair": repair,
        "enforce_official_counts": enforce_official_counts,
        "datasets": reports,
    }
    atomic_write_json(output_root / "revisitop_preparation_report.json", report)
    return report


def publish_revisitop_datasets(
    source_root: Path,
    persistent_root: Path,
    *,
    datasets: Sequence[str] = REVISITOP_DATASETS,
    verify_images: bool = True,
    enforce_official_counts: bool = True,
) -> dict[str, Any]:
    """Validated local-to-persistent RevisitOP publication.

    This is intentionally a thin, explicit wrapper around the same hidden-staging
    and validation path used for Drive-to-local restoration.  It is the correct
    way to persist a first-time `/content` download; do not use an ad-hoc
    ``shutil.copytree`` in a notebook.
    """
    source_root = Path(source_root)
    persistent_root = Path(persistent_root)
    if _same_path(source_root, persistent_root):
        raise ValueError("RevisitOP source_root and persistent_root must differ")
    report = prepare_revisitop_datasets(
        persistent_root,
        datasets=datasets,
        source_root=source_root,
        mode="stage",
        verify_images=verify_images,
        enforce_official_counts=enforce_official_counts,
    )
    report["publication_source_root"] = str(source_root)
    atomic_write_json(persistent_root / "revisitop_preparation_report.json", report)
    return report


def validate_revisitop_dataset(
    dataset_dir: Path,
    dataset: str,
    *,
    verify_images: bool = True,
    enforce_official_counts: bool = True,
) -> dict[str, Any]:
    """Check the layout, official annotation structure, and required JPEG files."""
    _validate_dataset_name(dataset)
    dataset_dir = Path(dataset_dir)
    errors: list[str] = []
    if not dataset_dir.is_dir() or dataset_dir.is_symlink():
        return {
            "valid": False,
            "errors": [f"dataset directory is missing or unsafe: {dataset_dir}"],
            "dataset": dataset,
        }
    ground_truth_path = dataset_dir / f"gnd_{dataset}.pkl"
    image_root = dataset_dir / "jpg"
    if not ground_truth_path.is_file() or ground_truth_path.is_symlink():
        errors.append(f"missing annotation: {ground_truth_path.name}")
    if not image_root.is_dir() or image_root.is_symlink():
        errors.append("missing or unsafe jpg directory")
    if errors:
        return {"valid": False, "errors": errors, "dataset": dataset}

    try:
        descriptor = RevisitOPDataset.from_ground_truth_pickle(
            name=dataset,
            ground_truth_path=ground_truth_path,
            image_root=image_root,
        )
        image_ids = _required_image_ids(descriptor)
    except Exception as error:
        return {
            "valid": False,
            "errors": [f"invalid RevisitOP annotation: {error}"],
            "dataset": dataset,
        }

    database_image_count = len(descriptor.database_ids)
    query_count = len(descriptor.queries)
    expected_counts = dict(OFFICIAL_REVISITOP_COUNTS[dataset])
    official_counts_match = (
        database_image_count == expected_counts["database_image_count"]
        and query_count == expected_counts["query_count"]
    )
    if enforce_official_counts and not official_counts_match:
        errors.append(
            f"official count mismatch for {dataset}: expected "
            f"{expected_counts['database_image_count']} database / "
            f"{expected_counts['query_count']} queries, got "
            f"{database_image_count} / {query_count}"
        )

    for image_id in image_ids:
        try:
            filename = _image_filename(image_id)
        except ValueError as error:
            errors.append(str(error))
            continue
        path = image_root / filename
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing image: {filename}")
            if len(errors) >= 10:
                break
            continue
        if verify_images:
            try:
                _verify_jpeg(path)
            except Exception as error:
                errors.append(f"invalid JPEG {filename}: {error}")
                if len(errors) >= 10:
                    break
    return {
        "valid": not errors,
        "errors": errors,
        "dataset": dataset,
        "annotation_sha256": _sha256_file(ground_truth_path),
        "database_image_count": database_image_count,
        "query_count": query_count,
        "required_image_count": len(image_ids),
        "expected_official_counts": expected_counts,
        "official_counts_match": official_counts_match,
    }


def _stage_from_existing(
    source_dataset: Path,
    stage_dir: Path,
    dataset: str,
    *,
    enforce_official_counts: bool,
) -> dict[str, Any]:
    # Check source layout and every required filename first.  Decode validation is
    # intentionally deferred to the copied staging directory so Drive staging does
    # not decode roughly 11k images twice.
    source_validation = validate_revisitop_dataset(
        source_dataset,
        dataset,
        verify_images=False,
        enforce_official_counts=enforce_official_counts,
    )
    if not source_validation["valid"]:
        raise ValueError(
            f"existing RevisitOP source is invalid: {source_dataset}; "
            f"errors: {source_validation['errors'][:5]}"
        )
    source_gnd = source_dataset / f"gnd_{dataset}.pkl"
    target_gnd = stage_dir / source_gnd.name
    _ensure_owned_directory(stage_dir / "jpg")
    _copy_file_atomic(source_gnd, target_gnd)
    descriptor = RevisitOPDataset.from_ground_truth_pickle(
        name=dataset,
        ground_truth_path=target_gnd,
        image_root=stage_dir / "jpg",
    )
    source_image_root = source_dataset / "jpg"
    for image_id in _required_image_ids(descriptor):
        filename = _image_filename(image_id)
        _copy_file_atomic(source_image_root / filename, stage_dir / "jpg" / filename)
    return {
        "kind": "existing_source",
        "source_dataset": str(source_dataset),
        "validation": source_validation,
    }


def _stage_from_official_downloads(
    output_root: Path,
    stage_dir: Path,
    dataset: str,
    *,
    urls: RevisitOPUrls,
    verify_images: bool,
    repair: bool,
    enforce_official_counts: bool,
) -> dict[str, Any]:
    annotation_url = urls.annotation_url(dataset)
    ground_truth_path = stage_dir / f"gnd_{dataset}.pkl"
    descriptor = _download_annotation_with_repair(
        annotation_url,
        ground_truth_path,
        dataset,
        repair=repair,
    )
    if enforce_official_counts:
        _assert_official_counts(descriptor, dataset)
    image_ids = _required_image_ids(descriptor)
    _ensure_owned_directory(stage_dir / "jpg")
    download_dir = output_root / ".downloads" / dataset
    _ensure_owned_directory(download_dir)
    archives: list[dict[str, str]] = []
    for archive_url in urls.archive_urls(dataset):
        archive_path = download_dir / Path(archive_url).name
        _download_archive_with_repair(archive_url, archive_path, repair=repair)
        try:
            _extract_expected_images_from_archive(
                archive_path,
                stage_dir / "jpg",
                image_ids,
                verify_existing=verify_images,
            )
        except (RuntimeError, tarfile.TarError, OSError) as error:
            if not repair:
                raise RuntimeError(
                    f"could not extract downloaded archive {archive_path}: {error}. "
                    "Rerun with repair=True / --repair to redownload this module-owned file."
                ) from error
            _discard_owned_download(archive_path)
            _download_archive_with_repair(archive_url, archive_path, repair=False)
            _extract_expected_images_from_archive(
                archive_path,
                stage_dir / "jpg",
                image_ids,
                verify_existing=verify_images,
            )
        archives.append(
            {"url": archive_url, "path": str(archive_path), "sha256": _sha256_file(archive_path)}
        )
    return {
        "kind": "official_download",
        "annotation_url": annotation_url,
        "annotation_sha256": _sha256_file(ground_truth_path),
        "official_counts": dict(OFFICIAL_REVISITOP_COUNTS[dataset]),
        "enforce_official_counts": enforce_official_counts,
        "archives": archives,
    }


def _find_source_dataset(source_root: Path | None, dataset: str) -> Path | None:
    if source_root is None:
        return None
    if source_root.is_symlink():
        raise RuntimeError(f"refusing to stage RevisitOP from a symlinked source root: {source_root}")
    candidates = [source_root / dataset, source_root / "datasets" / dataset]
    if source_root.name == dataset:
        candidates.append(source_root)
    for candidate in candidates:
        annotation = candidate / f"gnd_{dataset}.pkl"
        image_root = candidate / "jpg"
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and annotation.is_file()
            and not annotation.is_symlink()
            and image_root.is_dir()
            and not image_root.is_symlink()
        ):
            return candidate
    return None


def _download_annotation_with_repair(
    annotation_url: str,
    destination: Path,
    dataset: str,
    *,
    repair: bool,
) -> RevisitOPDataset:
    """Download/parse the official pickle, optionally replacing a corrupt owned file."""
    if Path(destination).is_symlink():
        raise RuntimeError(f"refusing to download through a symlink: {destination}")
    for attempt in range(2 if repair else 1):
        download_with_resume(annotation_url, destination)
        try:
            return RevisitOPDataset.from_ground_truth_pickle(
                name=dataset,
                ground_truth_path=destination,
                image_root=destination.parent / "jpg",
            )
        except Exception as error:
            if attempt == 0 and repair:
                _discard_owned_download(destination)
                continue
            raise RuntimeError(
                f"downloaded RevisitOP annotation is invalid: {destination}: {error}. "
                "Use repair=True / --repair to replace this module-owned file."
            ) from error
    raise AssertionError("annotation repair loop should either return or raise")


def _download_archive_with_repair(
    archive_url: str,
    destination: Path,
    *,
    repair: bool,
) -> None:
    """Download and header-validate a tar archive, with explicit repair support."""
    if Path(destination).is_symlink():
        raise RuntimeError(f"refusing to download through a symlink: {destination}")
    for attempt in range(2 if repair else 1):
        download_with_resume(archive_url, destination)
        try:
            with tarfile.open(destination, mode="r:*") as archive:
                # Iterate headers now so a truncated archive fails before extraction.
                for _ in archive:
                    pass
            return
        except (tarfile.TarError, OSError) as error:
            if attempt == 0 and repair:
                _discard_owned_download(destination)
                continue
            raise RuntimeError(
                f"downloaded RevisitOP archive is invalid: {destination}: {error}. "
                "Use repair=True / --repair to replace this module-owned file."
            ) from error
    raise AssertionError("archive repair loop should either return or raise")


def _discard_owned_download(path: Path) -> None:
    """Remove one exact, module-owned download and resumable sidecar.

    Callers pass only paths under a hidden staging/download directory controlled by
    this module.  This intentionally never receives a user-supplied source path.
    """
    path = Path(path)
    if path.is_file() and not path.is_symlink():
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"refusing to repair an unsafe download path: {path}")
    partial = path.with_suffix(path.suffix + ".part")
    if partial.is_file() and not partial.is_symlink():
        partial.unlink()
    elif partial.exists():
        raise RuntimeError(f"refusing to repair an unsafe partial path: {partial}")


def _extract_expected_images_from_archive(
    archive_path: Path,
    destination_root: Path,
    image_ids: Iterable[str],
    *,
    verify_existing: bool,
) -> None:
    """Safely extract only expected JPG basenames from an official tar archive."""
    expected = {_image_filename(image_id): image_id for image_id in image_ids}
    _ensure_owned_directory(destination_root)
    try:
        archive = tarfile.open(archive_path, mode="r:*")
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError(f"could not open RevisitOP archive {archive_path}: {error}") from error
    with archive:
        for member in archive:
            if not member.isfile():
                continue
            # Never use the archive member path as an output path.  This avoids tar
            # traversal attacks and intentionally flattens official archive folders.
            filename = PurePosixPath(member.name).name
            if filename not in expected:
                continue
            destination = destination_root / filename
            if destination.is_file() and not destination.is_symlink():
                if not verify_existing:
                    continue
                try:
                    _verify_jpeg(destination)
                    continue
                except Exception:
                    # Replace only this exact controlled staging file.
                    pass
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive member {member.name}")
            _copy_stream_atomic(source, destination)


def _required_image_ids(dataset: RevisitOPDataset) -> tuple[str, ...]:
    # Preserve first occurrence order for reproducible staging and reporting.
    return tuple(dict.fromkeys((*dataset.database_ids, *(q.source_image_id for q in dataset.queries))))


def _assert_official_counts(dataset: RevisitOPDataset, name: str) -> None:
    expected = OFFICIAL_REVISITOP_COUNTS[name]
    actual = {"database_image_count": len(dataset.database_ids), "query_count": len(dataset.queries)}
    if actual != expected:
        raise RuntimeError(
            f"downloaded {name} annotation has nonofficial counts: expected {expected}, got {actual}"
        )


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


def _verify_jpeg(path: Path) -> None:
    with path.open("rb") as handle:
        image = Image.open(handle)
        image.verify()
    # PIL's verify() checks file structure but deliberately does not decode all
    # pixel data.  Reopen and load to catch truncated entropy-coded payloads.
    with path.open("rb") as handle:
        image = Image.open(handle)
        if image.format != "JPEG":
            raise ValueError(f"expected JPEG format, got {image.format!r}")
        image.load()


def _ensure_owned_directory(path: Path) -> Path:
    """Create/check a module-owned directory without following a symlink."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError(f"expected a real directory, not a symlink/special path: {path}")
        return path
    path.mkdir(parents=True, exist_ok=False)
    if not path.is_dir() or path.is_symlink():  # defensive race/FS check
        raise RuntimeError(f"could not create a safe directory: {path}")
    return path


def _copy_file_atomic(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"source file is missing or unsafe: {source}")
    if _same_path(source, destination):
        return
    with source.open("rb") as handle:
        _copy_stream_atomic(handle, destination)


def _copy_stream_atomic(source: Any, destination: Path) -> None:
    _ensure_owned_directory(destination.parent)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        try:
            source.close()
        except Exception:
            pass


def _remove_completed_archives(output_root: Path, dataset: str, urls: RevisitOPUrls) -> None:
    """Remove only exact archives created by this module after a verified publish."""
    download_dir = Path(output_root) / ".downloads" / dataset
    for archive_url in urls.archive_urls(dataset):
        archive_path = download_dir / Path(archive_url).name
        if archive_path.is_symlink():
            raise RuntimeError(f"refusing to remove a symlinked archive: {archive_path}")
        if archive_path.is_file():
            archive_path.unlink()


def _normalise_datasets(datasets: Sequence[str]) -> tuple[str, ...]:
    if not datasets:
        raise ValueError("at least one RevisitOP dataset is required")
    result = tuple(dict.fromkeys(str(dataset).lower() for dataset in datasets))
    for dataset in result:
        _validate_dataset_name(dataset)
    return result


def _validate_dataset_name(dataset: str) -> None:
    if dataset not in REVISITOP_DATASETS:
        raise ValueError(
            f"unsupported RevisitOP dataset {dataset!r}; expected one of {REVISITOP_DATASETS}"
        )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False
