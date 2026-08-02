"""Small runtime helpers shared by the independent Colab notebooks.

This module deliberately does not clone repositories or set ``PYTHONPATH``.
Those tasks must happen in the first, self-contained cell of a notebook, before
the package can be imported.  Once installed, these helpers make the remaining
runtime and provenance setup explicit and repeatable.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .utils import atomic_write_json


CloudPlatform = Literal["colab", "kaggle", "local"]


@dataclass(frozen=True)
class RuntimePaths:
    """Paths used by one cloud notebook runtime and its durable artifact root."""

    platform: CloudPlatform
    project_root: Path
    runtime_root: Path
    persistent_root: Path | None

    @property
    def local_data_root(self) -> Path:
        return self.runtime_root / "data"

    @property
    def local_cache_root(self) -> Path:
        return self.runtime_root / "cache"

    @property
    def local_output_root(self) -> Path:
        return self.project_root / "outputs"

    def ensure_local_roots(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.local_data_root.mkdir(parents=True, exist_ok=True)
        self.local_cache_root.mkdir(parents=True, exist_ok=True)
        self.local_output_root.mkdir(parents=True, exist_ok=True)


def detect_platform() -> CloudPlatform:
    """Return the supported hosted runtime, or ``local`` outside one."""
    if Path("/kaggle").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir():
        return "colab"
    return "local"


def mount_colab_drive(mount_point: Path = Path("/content/drive")) -> Path:
    """Mount and verify Google Drive from an interactive Colab kernel.

    It intentionally raises outside Colab rather than creating a misleading
    ordinary ``/content/drive`` directory that would disappear with the VM.
    """
    if detect_platform() != "colab":
        raise RuntimeError("Google Drive mounting is available only in a Colab runtime")
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - defensive Colab diagnosis
        raise RuntimeError("Colab Drive API is unavailable in this notebook kernel") from error
    my_drive = Path(mount_point) / "MyDrive"
    if not my_drive.is_dir():
        drive.mount(str(mount_point))
    if not my_drive.is_dir():
        raise RuntimeError(f"Google Drive was not mounted at {my_drive}")
    return my_drive


def stage_file(persistent_path: Path, local_path: Path) -> Path:
    """Copy one durable input to fast runtime storage if it is not already present."""
    persistent_path = Path(persistent_path)
    local_path = Path(local_path)
    if local_path.is_file():
        return local_path
    if not persistent_path.is_file():
        raise FileNotFoundError(f"persistent input is missing: {persistent_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(f".{local_path.name}.part")
    try:
        with persistent_path.open("rb") as source, temporary.open("wb") as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
        temporary.replace(local_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return local_path


def publish_file(local_path: Path, persistent_path: Path) -> Path:
    """Atomically publish one completed runtime file to durable storage."""
    local_path = Path(local_path)
    persistent_path = Path(persistent_path)
    if not local_path.is_file():
        raise FileNotFoundError(f"local file to publish is missing: {local_path}")
    persistent_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = persistent_path.with_name(f".{persistent_path.name}.part")
    try:
        with local_path.open("rb") as source, temporary.open("wb") as destination:
            while block := source.read(1024 * 1024):
                destination.write(block)
        temporary.replace(persistent_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return persistent_path


def runtime_report(
    *,
    project_root: Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect compact, JSON-safe environment provenance for a notebook run."""
    report: dict[str, Any] = {
        "platform": detect_platform(),
        "cwd": str(Path.cwd()),
        "project_root": str(Path(project_root).resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "os": platform.platform(),
        "git_sha": _git_sha(project_root),
    }
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["torch_version"] = None
        report["cuda_available"] = False
    if extra:
        report["extra"] = dict(extra)
    return report


def write_runtime_report(
    output_dir: Path,
    *,
    project_root: Path,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``runtime_environment.json`` and return its path."""
    path = Path(output_dir) / "runtime_environment.json"
    atomic_write_json(path, runtime_report(project_root=project_root, extra=extra))
    return path


def _git_sha(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
