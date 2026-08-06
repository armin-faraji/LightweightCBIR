"""Small JSON records for reusable local notebook experiments."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

from .utils import atomic_write_json, read_json


def load_results(path: Path) -> dict[str, Any]:
    """Read one notebook's result book, or return an empty one."""
    if not path.exists():
        return {"schema_version": 1, "experiments": {}}
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), dict):
        raise ValueError(f"invalid experiment results file: {path}")
    return payload


def matching_experiment(
    results_path: Path,
    name: str,
    run_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a reusable result only when its exact JSON spec still matches."""
    record = load_results(results_path)["experiments"].get(_safe_name(name))
    if not isinstance(record, dict) or record.get("run_spec") != dict(run_spec):
        return None
    checkpoint = record.get("checkpoint")
    if checkpoint is not None and not (results_path.parent / checkpoint).is_file():
        return None
    return record


def remove_experiment(results_path: Path, name: str) -> None:
    """Discard one obsolete app-owned checkpoint and its result entry."""
    name = _safe_name(name)
    checkpoint_dir = results_path.parent / "checkpoints" / name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    results = load_results(results_path)
    if name in results["experiments"]:
        del results["experiments"][name]
        atomic_write_json(results_path, results)


def save_experiment(
    results_path: Path,
    *,
    name: str,
    run_spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    checkpoint: Path | None,
) -> dict[str, Any]:
    """Replace one experiment entry after its run has completed."""
    name = _safe_name(name)
    results = load_results(results_path)
    relative_checkpoint = (
        None
        if checkpoint is None
        else str(Path(checkpoint).relative_to(results_path.parent))
    )
    record = {
        "run_spec": dict(run_spec),
        "summary": dict(summary),
        "history": [dict(epoch) for epoch in history],
        "checkpoint": relative_checkpoint,
    }
    results["experiments"][name] = record
    atomic_write_json(results_path, results)
    return record


def write_selection(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a human-readable manual selection record."""
    atomic_write_json(path, dict(payload))


def read_selection(path: Path) -> dict[str, Any]:
    """Read a selection record with a clear error when its notebook was skipped."""
    if not path.is_file():
        raise FileNotFoundError(f"selection record is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid selection record: {path}")
    return payload


def _safe_name(name: str) -> str:
    name = str(name).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("experiment name must be one safe path component")
    return name
