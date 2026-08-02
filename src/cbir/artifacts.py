"""Durable experiment artifacts with validated, staged directory publishing.

The project treats a notebook runtime as disposable.  Each notebook therefore writes
figures, metrics, and provenance to a local run directory first, then publishes that
*completed* directory to persistent storage (for example, Google Drive).  The module
does not know how a cloud drive is mounted; a Drive path is simply a normal
``Path`` supplied by the caller.

Publishing never overwrites an existing run.  A completed copy is validated in a
hidden sibling directory before it is renamed into its visible destination.  A
``COMPLETE`` marker makes the completion contract explicit for consumers.  Directory
rename is atomic on normal local POSIX filesystems; a Google Drive FUSE mount can
only provide this staged, best-effort equivalent.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .utils import atomic_write_json, read_json, stable_hash


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
ARTIFACT_COMPLETE_NAME = "COMPLETE"
RUN_METADATA_NAME = "run_metadata.json"
DEFAULT_STALE_PUBLISH_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class ArtifactRun:
    """One local, notebook-owned artifact directory.

    ``local_dir`` has the conventional layout ``outputs/<notebook>/<run-id>``.
    Call :meth:`publish` as the last notebook step to copy the validated run to
    ``<drive-output-root>/<notebook>/<run-id>``.
    """

    local_dir: Path
    notebook: str
    run_id: str

    def path_for(self, relative_path: str | Path) -> Path:
        """Return a safe output path and create its parent directory.

        The manifest is module-owned, so callers cannot accidentally overwrite it.
        """
        relative = _safe_relative_path(relative_path)
        if relative.as_posix() in {ARTIFACT_MANIFEST_NAME, ARTIFACT_COMPLETE_NAME}:
            raise ValueError(f"{relative.as_posix()} is managed by ArtifactRun")
        path = self.local_dir / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        """Atomically write a JSON report under this run directory."""
        path = self.path_for(relative_path)
        atomic_write_json(path, payload)
        return path

    def save_figure(
        self,
        relative_path: str | Path,
        figure: Any,
        *,
        dpi: int = 300,
    ) -> Path:
        """Save a matplotlib-like figure as a report-ready image.

        ``figure`` deliberately uses a small duck-typed interface so importing this
        module does not import matplotlib in CLI-only environments.
        """
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        path = self.path_for(relative_path)
        try:
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
        except AttributeError as error:
            raise TypeError("figure must provide a savefig method") from error
        return path

    def finalize(self) -> dict[str, Any]:
        """Write and validate the manifest for all current artifacts."""
        return finalize_artifact_directory(
            self.local_dir,
            notebook=self.notebook,
            run_id=self.run_id,
        )

    def publish(self, drive_output_root: Path) -> Path:
        """Finalize, validate, and publish this run under persistent storage."""
        return publish_artifact_directory(self.local_dir, drive_output_root)


def create_artifact_run(
    output_root: Path,
    notebook: str | int,
    *,
    run_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRun:
    """Create an empty, uniquely named local artifact directory.

    The default ID combines a UTC timestamp and random suffix.  Notebooks should
    normally pass a more informative explicit ID such as
    ``"20260802T121500Z__git-a1b2c3d__cache-12345678"``.  The caller owns the
    meaning of metadata; a typical payload includes Git SHA, config fingerprint,
    device information, and command-line arguments.
    """
    notebook_name = _safe_component(str(notebook), field_name="notebook")
    chosen_run_id = _safe_component(
        run_id or _default_run_id(),
        field_name="run_id",
    )
    local_dir = Path(output_root) / notebook_name / chosen_run_id
    if local_dir.exists():
        raise FileExistsError(
            f"artifact run already exists; choose a new run_id: {local_dir}"
        )
    local_dir.mkdir(parents=True, exist_ok=False)
    run = ArtifactRun(local_dir=local_dir, notebook=notebook_name, run_id=chosen_run_id)
    run.write_json(
        RUN_METADATA_NAME,
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "notebook": notebook_name,
            "run_id": chosen_run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata or {}),
        },
    )
    return run


def make_artifact_run_id(
    *,
    notebook: str | int | None = None,
    git_sha: str | None = None,
    config_fingerprint: str | None = None,
    cache_fingerprint: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """Build a readable run ID from durable experiment identities.

    A typical notebook call is::

        run_id = make_artifact_run_id(
            notebook="03",
            git_sha=runtime["git_sha"],
            config_fingerprint=stable_hash(config_to_dict(cfg)),
            cache_fingerprint=reader.manifest.fingerprint,
        )

    The caller may append a seed/mode component before passing the result to
    :func:`create_artifact_run`.  IDs are intentionally deterministic within one
    UTC second; attempting to reuse one raises rather than overwriting artifacts.
    """
    current = timestamp or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("artifact run timestamp must be timezone-aware")
    parts = [current.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")]
    if notebook is not None:
        parts.append(f"nb-{_safe_component(str(notebook), field_name='notebook')}")
    if git_sha:
        parts.append(f"git-{_short_identity(git_sha)}")
    if config_fingerprint:
        parts.append(f"cfg-{_short_identity(config_fingerprint)}")
    if cache_fingerprint:
        parts.append(f"cache-{_short_identity(cache_fingerprint)}")
    return "__".join(parts)


def finalize_artifact_directory(
    artifact_dir: Path,
    *,
    notebook: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create an integrity manifest and return its validated contents.

    ``notebook`` and ``run_id`` are required only for a directory that has not
    already been finalized.  Passing them for an existing run additionally guards
    against publishing a directory under the wrong identity.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise FileNotFoundError(f"artifact directory does not exist: {artifact_dir}")

    existing = _read_manifest_if_present(artifact_dir)
    resolved_notebook = _safe_component(
        notebook or (existing or {}).get("notebook", ""),
        field_name="notebook",
    )
    resolved_run_id = _safe_component(
        run_id or (existing or {}).get("run_id", ""),
        field_name="run_id",
    )
    if existing is not None:
        if existing.get("notebook") != resolved_notebook:
            raise ValueError("artifact notebook does not match existing manifest")
        if existing.get("run_id") != resolved_run_id:
            raise ValueError("artifact run_id does not match existing manifest")

    files = _file_records(artifact_dir)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "notebook": resolved_notebook,
        "run_id": resolved_run_id,
        "files": files,
    }
    manifest = {**payload, "fingerprint": stable_hash(payload)}
    atomic_write_json(artifact_dir / ARTIFACT_MANIFEST_NAME, manifest)
    atomic_write_json(
        artifact_dir / ARTIFACT_COMPLETE_NAME,
        {"fingerprint": manifest["fingerprint"], "schema_version": ARTIFACT_SCHEMA_VERSION},
    )
    report = validate_artifact_directory(artifact_dir)
    if not report["valid"]:
        raise RuntimeError(f"artifact validation failed after finalize: {report['errors']}")
    return report["manifest"]


def validate_artifact_directory(
    artifact_dir: Path,
    *,
    expected_notebook: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate an artifact manifest and every managed file checksum."""
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / ARTIFACT_MANIFEST_NAME
    complete_path = artifact_dir / ARTIFACT_COMPLETE_NAME
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        return {
            "valid": False,
            "errors": [f"artifact directory is missing or unsafe: {artifact_dir}"],
            "manifest": None,
        }
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return {
            "valid": False,
            "errors": [f"{ARTIFACT_MANIFEST_NAME} is missing or unsafe"],
            "manifest": None,
        }
    if not complete_path.is_file() or complete_path.is_symlink():
        return {
            "valid": False,
            "errors": [f"{ARTIFACT_COMPLETE_NAME} marker is missing or unsafe"],
            "manifest": None,
        }

    errors: list[str] = []
    try:
        raw = read_json(manifest_path)
        manifest = _validate_manifest_structure(raw)
    except Exception as error:
        return {
            "valid": False,
            "errors": [f"invalid artifact manifest: {error}"],
            "manifest": None,
        }

    if expected_notebook is not None and manifest["notebook"] != str(expected_notebook):
        errors.append("artifact notebook does not match request")
    if expected_run_id is not None and manifest["run_id"] != str(expected_run_id):
        errors.append("artifact run_id does not match request")

    expected_payload = {
        "schema_version": manifest["schema_version"],
        "notebook": manifest["notebook"],
        "run_id": manifest["run_id"],
        "files": manifest["files"],
    }
    if manifest["fingerprint"] != stable_hash(expected_payload):
        errors.append("artifact manifest fingerprint is invalid")
    try:
        complete = read_json(complete_path)
        if not isinstance(complete, Mapping) or complete.get("fingerprint") != manifest["fingerprint"]:
            errors.append("artifact COMPLETE marker does not match manifest")
    except Exception as error:
        errors.append(f"invalid artifact COMPLETE marker: {error}")

    expected_by_path = {record["path"]: record for record in manifest["files"]}
    try:
        actual_paths = {record["path"] for record in _file_records(artifact_dir)}
    except Exception as error:
        errors.append(f"could not inspect artifact files: {error}")
        actual_paths = set()
    if actual_paths != set(expected_by_path):
        missing = sorted(set(expected_by_path) - actual_paths)
        unexpected = sorted(actual_paths - set(expected_by_path))
        if missing:
            errors.append(f"manifest files are missing: {missing[:3]}")
        if unexpected:
            errors.append(f"artifact directory has unmanifested files: {unexpected[:3]}")

    for relative_path, record in expected_by_path.items():
        path = artifact_dir / Path(PurePosixPath(relative_path))
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or unsafe artifact file: {relative_path}")
            continue
        if path.stat().st_size != record["bytes"]:
            errors.append(f"size mismatch for {relative_path}")
        if _sha256_file(path) != record["sha256"]:
            errors.append(f"checksum mismatch for {relative_path}")

    return {"valid": not errors, "errors": errors, "manifest": manifest}


def publish_artifact_directory(
    artifact_dir: Path,
    drive_output_root: Path,
    *,
    stale_publish_seconds: int = DEFAULT_STALE_PUBLISH_SECONDS,
) -> Path:
    """Publish a finalized artifact directory through a hidden staging directory.

    The destination is ``drive_output_root/<notebook>/<run-id>``.  A pre-existing
    destination is accepted only when it validates and has exactly the same manifest
    fingerprint, making retry after a successful publish idempotent.  Different
    content is never overwritten.
    """
    if stale_publish_seconds < 0:
        raise ValueError("stale_publish_seconds must be nonnegative")
    artifact_dir = Path(artifact_dir)
    manifest = finalize_artifact_directory(artifact_dir)
    destination = Path(drive_output_root) / manifest["notebook"] / manifest["run_id"]
    if _same_path(artifact_dir, destination):
        return artifact_dir
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise FileExistsError(f"refusing to publish through a symlink: {destination}")
        existing = validate_artifact_directory(
            destination,
            expected_notebook=manifest["notebook"],
            expected_run_id=manifest["run_id"],
        )
        if existing["valid"] and existing["manifest"]["fingerprint"] == manifest["fingerprint"]:
            return destination
        raise FileExistsError(
            "refusing to overwrite existing Drive artifact run: "
            f"{destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_publish_staging(
        destination.parent,
        destination.name,
        stale_publish_seconds=stale_publish_seconds,
    )
    staging = destination.parent / f".{destination.name}.publishing-{uuid.uuid4().hex}"
    try:
        _copytree_without_symlinks(artifact_dir, staging)
        staged = validate_artifact_directory(
            staging,
            expected_notebook=manifest["notebook"],
            expected_run_id=manifest["run_id"],
        )
        if not staged["valid"]:
            raise RuntimeError(f"published artifact validation failed: {staged['errors']}")
        if staged["manifest"]["fingerprint"] != manifest["fingerprint"]:
            raise RuntimeError("published artifact manifest fingerprint changed during copy")
        os.replace(staging, destination)
    except Exception:
        # This path is uniquely generated by this call, so deleting it cannot affect
        # an unrelated run.  The visible destination is left untouched on failure.
        if staging.exists():
            shutil.rmtree(staging)
        raise

    published = validate_artifact_directory(
        destination,
        expected_notebook=manifest["notebook"],
        expected_run_id=manifest["run_id"],
    )
    if not published["valid"]:
        raise RuntimeError(f"Drive artifact failed validation after publish: {published['errors']}")
    return destination


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}__{uuid.uuid4().hex[:8]}"


def _safe_component(value: object, *, field_name: str) -> str:
    result = str(value).strip()
    if not result or result in {".", ".."}:
        raise ValueError(f"{field_name} must be a non-empty path component")
    if "/" in result or "\\" in result or "\x00" in result:
        raise ValueError(f"{field_name} must be a single safe path component")
    return result


def _short_identity(value: str, *, max_length: int = 12) -> str:
    """Make an identifier readable in paths without trusting its original syntax."""
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in str(value))
    cleaned = cleaned.strip("-")
    if not cleaned:
        return stable_hash(str(value), length=max_length)
    return cleaned[:max_length]


def _safe_relative_path(value: str | Path) -> PurePosixPath:
    raw = str(value)
    if "\\" in raw or "\x00" in raw:
        raise ValueError("artifact path must use a safe relative POSIX path")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path must be a non-empty relative path without '..'")
    return path


def _read_manifest_if_present(artifact_dir: Path) -> dict[str, Any] | None:
    path = artifact_dir / ARTIFACT_MANIFEST_NAME
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"{ARTIFACT_MANIFEST_NAME} is not a regular file")
    return _validate_manifest_structure(read_json(path))


def _validate_manifest_structure(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("manifest must be a mapping")
    if int(raw.get("schema_version", -1)) != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported artifact manifest schema version")
    notebook = _safe_component(raw.get("notebook", ""), field_name="notebook")
    run_id = _safe_component(raw.get("run_id", ""), field_name="run_id")
    fingerprint = str(raw.get("fingerprint", ""))
    if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
        raise ValueError("manifest fingerprint must be a lowercase SHA-256 digest")
    raw_files = raw.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("manifest files must be a list")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in raw_files:
        if not isinstance(raw_record, Mapping):
            raise ValueError("manifest file records must be mappings")
        relative = _safe_relative_path(str(raw_record.get("path", ""))).as_posix()
        if relative in {ARTIFACT_MANIFEST_NAME, ARTIFACT_COMPLETE_NAME}:
            raise ValueError("manifest must not list its managed files as artifact files")
        if relative in seen:
            raise ValueError(f"duplicate artifact manifest path: {relative}")
        seen.add(relative)
        byte_count = raw_record.get("bytes")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"invalid byte count for {relative}")
        sha256 = str(raw_record.get("sha256", ""))
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"invalid SHA-256 digest for {relative}")
        files.append({"path": relative, "bytes": byte_count, "sha256": sha256})
    if files != sorted(files, key=lambda item: item["path"]):
        raise ValueError("manifest file records must be sorted by path")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "notebook": notebook,
        "run_id": run_id,
        "files": files,
        "fingerprint": fingerprint,
    }


def _file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact directories must not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact directories must contain regular files only: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in {ARTIFACT_MANIFEST_NAME, ARTIFACT_COMPLETE_NAME}:
            continue
        records.append(
            {
                "path": _safe_relative_path(relative).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copytree_without_symlinks(source: Path, destination: Path) -> None:
    # Validate before copying so shutil never follows a source symlink.
    _file_records(source)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def _cleanup_stale_publish_staging(
    parent: Path,
    run_id: str,
    *,
    stale_publish_seconds: int,
) -> None:
    """Remove only old, uniquely named hidden publisher directories.

    This prevents abandoned Drive FUSE copies from accumulating after a VM reset.
    A running publisher is protected by the age threshold; callers may use zero
    only when they know no concurrent publication of this exact run is active.
    """
    cutoff = time.time() - stale_publish_seconds
    for candidate in parent.glob(f".{run_id}.publishing-*"):
        try:
            if (
                candidate.is_symlink()
                or not candidate.is_dir()
                or candidate.stat().st_mtime > cutoff
            ):
                continue
            shutil.rmtree(candidate)
        except OSError:
            # A concurrent filesystem operation should not make a valid new
            # publication fail.  Its own validation/rename will still decide.
            continue


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False
