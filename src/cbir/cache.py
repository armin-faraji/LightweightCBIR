"""Sharded, resumable cache for frozen all-layer feature aggregates."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import uuid
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .config import FeatureCacheConfig
from .data.sfm import ImageRecord
from .data.transforms import PreprocessRecord
from .features import AllLayerFeatures
from .utils import atomic_write_json, hash_strings, read_json


MANIFEST_NAME = "manifest.json"
COMPLETE_NAME = "COMPLETE"
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def cache_dir_name(dataset_name: str, model_name: str, fingerprint: str) -> str:
    """A readable cache-folder identity; full settings remain in manifest JSON."""
    safe_dataset = dataset_name.replace("/", "-").replace(" ", "-")
    safe_model = model_name.replace("/", "-").replace(" ", "-")
    return f"{safe_dataset}__{safe_model}__{fingerprint[:12]}"


@dataclass(frozen=True)
class FeatureShardInfo:
    name: str
    image_ids: tuple[str, ...]
    sha256: str
    bytes: int


@dataclass(frozen=True)
class FeatureManifest:
    fingerprint: str
    dataset_name: str
    source_ids_hash: str
    extraction_config: Mapping[str, Any]
    layer_indices: tuple[int, ...]
    token_dim: int
    feature_dtype: str
    entropy_dtype: str
    expected_image_ids: tuple[str, ...]
    shards: tuple[FeatureShardInfo, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "dataset_name": self.dataset_name,
            "source_ids_hash": self.source_ids_hash,
            "extraction_config": dict(self.extraction_config),
            "layer_indices": list(self.layer_indices),
            "token_dim": self.token_dim,
            "feature_dtype": self.feature_dtype,
            "entropy_dtype": self.entropy_dtype,
            "expected_image_ids": list(self.expected_image_ids),
            "shards": [
                {
                    **asdict(shard),
                    "image_ids": list(shard.image_ids),
                }
                for shard in self.shards
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureManifest":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported feature cache schema version")
        return cls(
            fingerprint=str(payload["fingerprint"]),
            dataset_name=str(payload["dataset_name"]),
            source_ids_hash=str(payload["source_ids_hash"]),
            extraction_config=dict(payload["extraction_config"]),
            layer_indices=tuple(int(value) for value in payload["layer_indices"]),
            token_dim=int(payload["token_dim"]),
            feature_dtype=str(payload["feature_dtype"]),
            entropy_dtype=str(payload["entropy_dtype"]),
            expected_image_ids=tuple(str(value) for value in payload["expected_image_ids"]),
            shards=tuple(
                FeatureShardInfo(
                    name=str(shard["name"]),
                    image_ids=tuple(str(value) for value in shard["image_ids"]),
                    sha256=str(shard["sha256"]),
                    bytes=int(shard["bytes"]),
                )
                for shard in payload.get("shards", [])
            ),
        )


class FeatureShardWriter:
    """Write validated shards locally and maintain an atomic JSON manifest."""

    def __init__(
        self,
        cache_dir: Path,
        manifest: FeatureManifest,
        config: FeatureCacheConfig,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.shards_dir = self.cache_dir / "shards"
        self.manifest = manifest
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_dir / MANIFEST_NAME
        if manifest_path.exists():
            existing = FeatureManifest.from_dict(read_json(manifest_path))
            if not _same_manifest_identity(existing, manifest):
                raise ValueError(
                    "refusing to write into a cache with incompatible immutable metadata"
                )
            self.manifest = existing
        else:
            atomic_write_json(manifest_path, self.manifest.to_dict())

    @property
    def completed_ids(self) -> frozenset[str]:
        return frozenset(
            image_id
            for shard in self.manifest.shards
            for image_id in shard.image_ids
        )

    def write_shard(
        self,
        features: AllLayerFeatures,
        records: Mapping[str, ImageRecord],
    ) -> FeatureShardInfo:
        if (self.cache_dir / COMPLETE_NAME).exists():
            raise RuntimeError("cannot append to a completed feature cache")
        if tuple(features.layer_indices) != tuple(self.manifest.layer_indices):
            raise ValueError("feature layers do not match cache manifest")
        if features.cls.shape[2] != self.manifest.token_dim:
            raise ValueError("feature width does not match cache manifest")
        image_ids = tuple(features.image_ids)
        if not image_ids:
            raise ValueError("cannot write an empty shard")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("feature shard has duplicate image IDs")
        unknown = set(image_ids) - set(self.manifest.expected_image_ids)
        if unknown:
            raise ValueError(f"feature shard IDs are not expected: {sorted(unknown)[:5]}")
        overlap = set(image_ids) & set(self.completed_ids)
        if overlap:
            raise ValueError(f"feature shard overlaps completed IDs: {sorted(overlap)[:5]}")
        if set(image_ids) != set(records):
            raise ValueError("records mapping must contain exactly the shard image IDs")
        _validate_feature_shapes(features)

        shard_index = len(self.manifest.shards)
        name = f"shard_{shard_index:05d}.pt"
        final_path = self.shards_dir / name
        if final_path.exists():
            raise FileExistsError(f"refusing to overwrite existing shard {final_path}")
        payload = _feature_payload(features, records, self.config)
        temporary_path = _temporary_path(final_path)
        try:
            torch.save(payload, temporary_path)
            _validate_shard_file(
                temporary_path,
                expected_ids=image_ids,
                token_dim=self.manifest.token_dim,
                layer_count=len(self.manifest.layer_indices),
            )
            os.replace(temporary_path, final_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        info = FeatureShardInfo(
            name=name,
            image_ids=image_ids,
            sha256=sha256_file(final_path),
            bytes=final_path.stat().st_size,
        )
        self.manifest = FeatureManifest(
            fingerprint=self.manifest.fingerprint,
            dataset_name=self.manifest.dataset_name,
            source_ids_hash=self.manifest.source_ids_hash,
            extraction_config=self.manifest.extraction_config,
            layer_indices=self.manifest.layer_indices,
            token_dim=self.manifest.token_dim,
            feature_dtype=self.manifest.feature_dtype,
            entropy_dtype=self.manifest.entropy_dtype,
            expected_image_ids=self.manifest.expected_image_ids,
            shards=(*self.manifest.shards, info),
        )
        atomic_write_json(self.cache_dir / MANIFEST_NAME, self.manifest.to_dict())
        return info

    def finalize(self) -> FeatureManifest:
        expected = set(self.manifest.expected_image_ids)
        completed = set(self.completed_ids)
        if completed != expected:
            missing = sorted(expected - completed)
            unexpected = sorted(completed - expected)
            raise ValueError(
                "cannot finalize incomplete cache: "
                f"{len(missing)} missing, {len(unexpected)} unexpected IDs"
            )
        report = validate_feature_cache(
            self.cache_dir,
            expected_fingerprint=self.manifest.fingerprint,
            require_complete=False,
        )
        if not report["valid"]:
            raise RuntimeError(f"cache validation failed before finalize: {report['errors']}")
        temporary = _temporary_path(self.cache_dir / COMPLETE_NAME)
        temporary.write_text(
            f"fingerprint={self.manifest.fingerprint}\nsource_ids_hash={self.manifest.source_ids_hash}\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.cache_dir / COMPLETE_NAME)
        return self.manifest


class FeatureShardReader:
    """Read cached tensors by canonical image IDs with a small shard LRU cache."""

    def __init__(self, cache_dir: Path, max_cached_shards: int = 3) -> None:
        self.cache_dir = Path(cache_dir)
        report = validate_feature_cache(self.cache_dir)
        if not report["valid"]:
            raise ValueError(f"invalid feature cache: {report['errors']}")
        self.manifest = FeatureManifest.from_dict(read_json(self.cache_dir / MANIFEST_NAME))
        self.max_cached_shards = max_cached_shards
        self._locations = {
            image_id: (shard_index, offset)
            for shard_index, shard in enumerate(self.manifest.shards)
            for offset, image_id in enumerate(shard.image_ids)
        }
        self._loaded: OrderedDict[int, Mapping[str, Any]] = OrderedDict()

    @property
    def image_ids(self) -> tuple[str, ...]:
        return self.manifest.expected_image_ids

    def fetch(
        self,
        image_ids: Sequence[str],
        *,
        layer_indices: Sequence[int] | None = None,
        local_kind: str = "cls_guided_patch",
    ) -> dict[str, torch.Tensor]:
        if local_kind not in {"cls_guided_patch", "mean_patch"}:
            raise ValueError("local_kind must be cls_guided_patch or mean_patch")
        if not image_ids:
            raise ValueError("cannot fetch an empty image list")
        positions = _layer_positions(self.manifest.layer_indices, layer_indices)
        rows = [self._fetch_row(image_id) for image_id in image_ids]
        return {
            "cls": torch.stack([row["cls"][positions] for row in rows]).float(),
            "local": torch.stack([row[local_kind][positions] for row in rows]).float(),
            "entropy": torch.stack([row["pooling_entropy"][positions] for row in rows]).float(),
        }

    def metadata_for(self, image_ids: Sequence[str]) -> list[dict[str, Any]]:
        return [dict(self._fetch_row(image_id)["metadata"]) for image_id in image_ids]

    def _fetch_row(self, image_id: str) -> Mapping[str, Any]:
        try:
            shard_index, offset = self._locations[image_id]
        except KeyError as error:
            raise KeyError(f"image ID {image_id} is not present in this cache") from error
        shard = self._load_shard(shard_index)
        return {
            "cls": shard["cls"][offset],
            "mean_patch": shard["mean_patch"][offset],
            "cls_guided_patch": shard["cls_guided_patch"][offset],
            "pooling_entropy": shard["pooling_entropy"][offset],
            "metadata": shard["metadata"][offset],
        }

    def _load_shard(self, shard_index: int) -> Mapping[str, Any]:
        if shard_index in self._loaded:
            self._loaded.move_to_end(shard_index)
            return self._loaded[shard_index]
        shard_info = self.manifest.shards[shard_index]
        path = self.cache_dir / "shards" / shard_info.name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(payload["image_ids"]) != shard_info.image_ids:
            raise ValueError(f"cached shard ID list changed: {path}")
        self._loaded[shard_index] = payload
        while len(self._loaded) > self.max_cached_shards:
            self._loaded.popitem(last=False)
        return payload


class CacheResolver:
    """Local-first cache resolution with resumable, validated Drive mirroring.

    A manifest is the commit record for a shard: a shard is copied and validated
    first, then the manifest is atomically replaced to expose it to a future
    runtime.  This matters for Google Drive/FUSE, where a Colab session can end
    between individual file operations.
    """

    def __init__(self, local_cache_dir: Path, drive_cache_dir: Path | None = None) -> None:
        self.local_cache_dir = Path(local_cache_dir)
        self.drive_cache_dir = None if drive_cache_dir is None else Path(drive_cache_dir)

    def resolve_existing(
        self,
        expected_fingerprint: str,
        *,
        allow_incomplete: bool = False,
    ) -> Path | None:
        """Return a compatible local cache, restoring it from Drive if needed.

        By default only a completed cache is accepted, preserving the original
        semantics used by training and evaluation.  Extraction should opt in to
        ``allow_incomplete=True`` so a new Colab runtime can restore a validated
        prefix of completed shards and continue from it.
        """
        require_complete = not allow_incomplete
        local = validate_feature_cache(
            self.local_cache_dir,
            expected_fingerprint=expected_fingerprint,
            require_complete=require_complete,
        )
        if local["valid"]:
            return self.local_cache_dir
        if self.drive_cache_dir is None:
            return None
        drive = validate_feature_cache(
            self.drive_cache_dir,
            expected_fingerprint=expected_fingerprint,
            require_complete=require_complete,
        )
        if not drive["valid"]:
            return None
        # A runtime can leave a corrupt local directory behind (for example
        # after a disk-full or interrupted write).  A separately validated
        # Drive cache is authoritative in that case.  Move—not delete—the bad
        # local path aside before restoring, so a notebook can recover without
        # manual cleanup and any forensic files remain available.
        if self.local_cache_dir.exists() or self.local_cache_dir.is_symlink():
            quarantined = self._quarantine_invalid_local_cache()
            warnings.warn(
                "quarantined an invalid local feature cache before restoring "
                f"the validated Drive cache: {quarantined}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.restore_drive_to_local(require_complete=require_complete)
        restored = validate_feature_cache(
            self.local_cache_dir,
            expected_fingerprint=expected_fingerprint,
            require_complete=require_complete,
        )
        if not restored["valid"]:
            raise RuntimeError(f"Drive restore failed validation: {restored['errors']}")
        return self.local_cache_dir

    def _quarantine_invalid_local_cache(self) -> Path:
        """Atomically move exactly one invalid local cache aside for recovery.

        This deliberately never removes data.  The resolver reaches this helper
        only after a compatible Drive cache has fully validated, and the source
        is the caller's explicit ``local_cache_dir`` rather than a wildcard.
        """
        parent = self.local_cache_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        quarantine = parent / (
            f".{self.local_cache_dir.name}.invalid-{uuid.uuid4().hex}"
        )
        os.replace(self.local_cache_dir, quarantine)
        return quarantine

    def restore_drive_to_local(self, *, require_complete: bool = True) -> None:
        """Restore a validated Drive cache without copying uncommitted files.

        Only shards referenced by the validated manifest are copied.  An
        interrupted publication may leave an orphan shard on Drive after the
        shard upload but before the manifest commit; copying the whole directory
        would make that orphan block a later resumed writer.
        """
        if self.drive_cache_dir is None:
            raise RuntimeError("Drive cache directory is not configured")
        source_report = validate_feature_cache(
            self.drive_cache_dir,
            require_complete=require_complete,
        )
        if not source_report["valid"]:
            raise ValueError(
                "refusing to restore an invalid Drive cache: " f"{source_report['errors']}"
            )
        if self.local_cache_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing local cache directory {self.local_cache_dir}"
            )
        manifest = FeatureManifest.from_dict(read_json(self.drive_cache_dir / MANIFEST_NAME))
        self.local_cache_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.local_cache_dir.name}.restore-",
                dir=self.local_cache_dir.parent,
            )
        )
        try:
            temporary_shards = temporary / "shards"
            temporary_shards.mkdir(parents=True, exist_ok=True)
            for shard in manifest.shards:
                _copy_shard_atomically(
                    self.drive_cache_dir / "shards" / shard.name,
                    temporary_shards / shard.name,
                    shard=shard,
                    token_dim=manifest.token_dim,
                    layer_count=len(manifest.layer_indices),
                )
            _copy_file_atomically(
                self.drive_cache_dir / MANIFEST_NAME,
                temporary / MANIFEST_NAME,
            )
            # The report is not required to resume extraction, but restoring it
            # keeps a completed cache self-describing after a fresh runtime.
            report_path = self.drive_cache_dir / "extraction_report.json"
            if report_path.is_file():
                _copy_file_atomically(report_path, temporary / report_path.name)
            if (self.drive_cache_dir / COMPLETE_NAME).is_file():
                _copy_file_atomically(
                    self.drive_cache_dir / COMPLETE_NAME,
                    temporary / COMPLETE_NAME,
                )
            restored = validate_feature_cache(temporary, require_complete=require_complete)
            if not restored["valid"]:
                raise RuntimeError(
                    "temporary Drive restore failed validation: " f"{restored['errors']}"
                )
            os.replace(temporary, self.local_cache_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def mirror_local_to_drive(
        self,
        *,
        require_complete: bool = True,
        extra_files: Sequence[str | Path] = (),
        incremental: bool = False,
    ) -> None:
        """Incrementally publish local cache progress to Drive.

        The source must validate at the requested completion level.  Shards are
        copied atomically and checksum-validated before an atomic manifest update
        makes them visible.  ``extra_files`` are relative cache paths, intended
        for final artifacts such as ``extraction_report.json``.  For a long
        extraction, ``incremental=True`` avoids rehashing every previously
        committed Drive shard after each new shard; it validates the new shard
        and manifest transaction, while final publication remains full-cache
        validated.
        """
        if self.drive_cache_dir is None:
            return
        if incremental and require_complete:
            raise ValueError("incremental mirroring is only valid for incomplete progress")
        if incremental and extra_files:
            raise ValueError("incremental mirroring cannot publish final cache artifacts")
        manifest = FeatureManifest.from_dict(read_json(self.local_cache_dir / MANIFEST_NAME))
        if incremental:
            _validate_manifest_progress(manifest)
        else:
            report = validate_feature_cache(
                self.local_cache_dir,
                require_complete=require_complete,
            )
            if not report["valid"]:
                raise ValueError(f"refusing to mirror invalid local cache: {report['errors']}")
        relative_extra_files = _normalise_relative_files(extra_files)
        self.drive_cache_dir.mkdir(parents=True, exist_ok=True)
        drive_manifest_path = self.drive_cache_dir / MANIFEST_NAME
        committed_shard_names: set[str] = set()
        if drive_manifest_path.is_file():
            try:
                existing_manifest = FeatureManifest.from_dict(read_json(drive_manifest_path))
            except Exception as error:
                raise ValueError(f"Drive target has an invalid manifest: {error}") from error
            _ensure_manifest_prefix(existing_manifest, manifest)
            committed_shard_names = {shard.name for shard in existing_manifest.shards}
            if (self.drive_cache_dir / COMPLETE_NAME).is_file() and (
                existing_manifest.shards != manifest.shards
            ):
                raise ValueError("refusing to modify a completed Drive cache")
        elif (self.drive_cache_dir / COMPLETE_NAME).exists():
            raise ValueError("Drive target has COMPLETE without a manifest")

        for shard in manifest.shards:
            if incremental and shard.name in committed_shard_names:
                continue
            _copy_shard_atomically(
                self.local_cache_dir / "shards" / shard.name,
                self.drive_cache_dir / "shards" / shard.name,
                shard=shard,
                token_dim=manifest.token_dim,
                layer_count=len(manifest.layer_indices),
                # A file absent from the prior manifest is an uncommitted
                # upload.  It is safe to repair/replace it; a referenced shard
                # is immutable and must match exactly.
                replace_existing=shard.name not in committed_shard_names,
            )

        # Commit the shard list only after every referenced shard exists and has
        # been independently verified on the Drive filesystem.
        atomic_write_json(drive_manifest_path, manifest.to_dict())
        if incremental:
            published_manifest = FeatureManifest.from_dict(read_json(drive_manifest_path))
            if published_manifest != manifest:
                raise RuntimeError("Drive manifest changed during incremental publication")
        else:
            partial_report = validate_feature_cache(self.drive_cache_dir, require_complete=False)
            if not partial_report["valid"]:
                raise RuntimeError(
                    "Drive cache failed validation after manifest publication: "
                    f"{partial_report['errors']}"
                )

        for relative_path in relative_extra_files:
            source = self.local_cache_dir / relative_path
            if not source.is_file():
                raise FileNotFoundError(f"requested cache artifact does not exist: {source}")
            _copy_file_atomically(
                source,
                self.drive_cache_dir / relative_path,
                replace_existing=True,
            )

        # COMPLETE is deliberately last: a reader will never accept this cache
        # as final until its manifest, all shards, and requested final artifacts
        # have been published.
        if require_complete:
            _copy_file_atomically(
                self.local_cache_dir / COMPLETE_NAME,
                self.drive_cache_dir / COMPLETE_NAME,
                replace_existing=True,
            )
        if not incremental:
            final_report = validate_feature_cache(
                self.drive_cache_dir,
                expected_fingerprint=manifest.fingerprint,
                require_complete=require_complete,
            )
            if not final_report["valid"]:
                raise RuntimeError(f"Drive mirror failed validation: {final_report['errors']}")
        for relative_path in relative_extra_files:
            _ensure_same_file(
                self.local_cache_dir / relative_path,
                self.drive_cache_dir / relative_path,
            )


def validate_feature_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: str | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate a frozen-feature cache and return a JSON-serializable report.

    ``require_complete=False`` is intentionally useful only for extraction and
    restoration.  It accepts a manifest whose validated shard IDs are a subset
    of ``expected_image_ids``; readers and training retain the default
    complete-cache requirement.
    """
    cache_dir = Path(cache_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return {
            "valid": False,
            "errors": ["manifest.json is missing"],
            "warnings": warnings,
            "manifest": None,
        }
    try:
        manifest = FeatureManifest.from_dict(read_json(manifest_path))
    except Exception as error:
        return {
            "valid": False,
            "errors": [f"invalid manifest: {error}"],
            "warnings": warnings,
            "manifest": None,
        }
    if expected_fingerprint is not None and manifest.fingerprint != expected_fingerprint:
        errors.append("cache fingerprint does not match request")
    if hash_strings(manifest.expected_image_ids) != manifest.source_ids_hash:
        errors.append("manifest source_ids_hash does not match expected IDs")
    expected_ids = set(manifest.expected_image_ids)
    if len(expected_ids) != len(manifest.expected_image_ids):
        errors.append("manifest expected_image_ids contains duplicates")
    seen: set[str] = set()
    seen_shard_names: set[str] = set()
    for shard in manifest.shards:
        if Path(shard.name).name != shard.name or not shard.name.endswith(".pt"):
            errors.append(f"invalid shard name {shard.name!r}")
            continue
        if shard.name in seen_shard_names:
            errors.append(f"duplicate shard name {shard.name}")
            continue
        seen_shard_names.add(shard.name)
        if not shard.image_ids:
            errors.append(f"shard {shard.name} has no image IDs")
        if len(set(shard.image_ids)) != len(shard.image_ids):
            errors.append(f"duplicate IDs inside shard {shard.name}")
        unknown = set(shard.image_ids) - expected_ids
        if unknown:
            errors.append(f"unexpected IDs in shard {shard.name}: {sorted(unknown)[:3]}")
        path = cache_dir / "shards" / shard.name
        if not path.is_file():
            errors.append(f"missing shard {shard.name}")
            continue
        if path.stat().st_size != shard.bytes:
            errors.append(f"size mismatch for shard {shard.name}")
        if sha256_file(path) != shard.sha256:
            errors.append(f"hash mismatch for shard {shard.name}")
        duplicate = seen & set(shard.image_ids)
        if duplicate:
            errors.append(f"duplicate IDs across shards: {sorted(duplicate)[:3]}")
        seen.update(shard.image_ids)
    missing = expected_ids - seen
    if require_complete and seen != expected_ids:
        errors.append(
            f"shard IDs do not exactly match manifest expectation "
            f"(missing={len(missing)})"
        )
    complete_marker = cache_dir / COMPLETE_NAME
    marker_exists = complete_marker.is_file()
    if marker_exists:
        _validate_complete_marker(complete_marker, manifest, errors)
        if seen != expected_ids:
            errors.append("COMPLETE marker is present but shard IDs are incomplete")
    elif require_complete:
        errors.append("COMPLETE marker is missing")
    # Unreferenced shard files are possible when a session ends after the shard
    # upload but before the manifest commit.  They are intentionally ignored by
    # restore (rather than trusted), and will be checked before reuse on a later
    # publication.
    shards_dir = cache_dir / "shards"
    if shards_dir.is_dir():
        unreferenced = sorted(
            path.name
            for path in shards_dir.glob("*.pt")
            if path.name not in seen_shard_names
        )
        if unreferenced:
            warnings.append(
                f"ignoring {len(unreferenced)} unreferenced shard file(s)"
            )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "manifest": manifest.to_dict(),
        "completed_image_count": len(seen),
        "expected_image_count": len(expected_ids),
        "is_complete": marker_exists and seen == expected_ids and not errors,
    }


def _same_manifest_identity(first: FeatureManifest, second: FeatureManifest) -> bool:
    """Compare immutable cache fields while deliberately excluding progress."""
    return (
        first.schema_version == second.schema_version
        and first.fingerprint == second.fingerprint
        and first.dataset_name == second.dataset_name
        and first.source_ids_hash == second.source_ids_hash
        and dict(first.extraction_config) == dict(second.extraction_config)
        and first.layer_indices == second.layer_indices
        and first.token_dim == second.token_dim
        and first.feature_dtype == second.feature_dtype
        and first.entropy_dtype == second.entropy_dtype
        and first.expected_image_ids == second.expected_image_ids
    )


def _ensure_manifest_prefix(existing: FeatureManifest, candidate: FeatureManifest) -> None:
    """Reject cache histories that cannot be a safe prefix of one another."""
    if not _same_manifest_identity(existing, candidate):
        raise ValueError("Drive cache immutable metadata is incompatible with local cache")
    if len(existing.shards) > len(candidate.shards):
        raise ValueError("Drive cache contains more shards than the local cache")
    if existing.shards != candidate.shards[: len(existing.shards)]:
        raise ValueError("Drive cache shard history diverges from the local cache")


def _validate_manifest_progress(manifest: FeatureManifest) -> None:
    """Cheap structural validation used between full extraction checkpoints."""
    errors: list[str] = []
    expected_ids = set(manifest.expected_image_ids)
    if len(expected_ids) != len(manifest.expected_image_ids):
        errors.append("manifest expected_image_ids contains duplicates")
    if hash_strings(manifest.expected_image_ids) != manifest.source_ids_hash:
        errors.append("manifest source_ids_hash does not match expected IDs")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for shard in manifest.shards:
        if Path(shard.name).name != shard.name or not shard.name.endswith(".pt"):
            errors.append(f"invalid shard name {shard.name!r}")
        if shard.name in seen_names:
            errors.append(f"duplicate shard name {shard.name}")
        seen_names.add(shard.name)
        if not shard.image_ids:
            errors.append(f"shard {shard.name} has no image IDs")
        if len(set(shard.image_ids)) != len(shard.image_ids):
            errors.append(f"duplicate IDs inside shard {shard.name}")
        unknown = set(shard.image_ids) - expected_ids
        if unknown:
            errors.append(f"unexpected IDs in shard {shard.name}: {sorted(unknown)[:3]}")
        duplicate = seen_ids & set(shard.image_ids)
        if duplicate:
            errors.append(f"duplicate IDs across shards: {sorted(duplicate)[:3]}")
        seen_ids.update(shard.image_ids)
    if errors:
        raise ValueError(f"invalid cache manifest progress: {errors}")


def _validate_complete_marker(
    marker_path: Path,
    manifest: FeatureManifest,
    errors: list[str],
) -> None:
    try:
        values = {}
        for line in marker_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", maxsplit=1)
            values[key.strip()] = value.strip()
    except OSError as error:
        errors.append(f"could not read COMPLETE marker: {error}")
        return
    if values.get("fingerprint") != manifest.fingerprint:
        errors.append("COMPLETE marker fingerprint does not match manifest")
    if values.get("source_ids_hash") != manifest.source_ids_hash:
        errors.append("COMPLETE marker source_ids_hash does not match manifest")


def _normalise_relative_files(files: Sequence[str | Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    for value in files:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError(f"cache artifact must be a relative file path: {value!r}")
        if path in result:
            continue
        result.append(path)
    return tuple(result)


def _copy_shard_atomically(
    source: Path,
    destination: Path,
    *,
    shard: FeatureShardInfo,
    token_dim: int,
    layer_count: int,
    replace_existing: bool = False,
) -> None:
    """Copy one shard only if its content matches the manifest checksum."""
    if not source.is_file():
        raise FileNotFoundError(f"source shard is missing: {source}")
    _ensure_file_matches_info(source, shard)
    if destination.is_file():
        try:
            _ensure_file_matches_info(destination, shard)
            _validate_shard_file(
                destination,
                expected_ids=shard.image_ids,
                token_dim=token_dim,
                layer_count=layer_count,
            )
            return
        except Exception:
            if not replace_existing:
                raise
    temporary = _temporary_path(destination)
    try:
        shutil.copyfile(source, temporary)
        _ensure_file_matches_info(temporary, shard)
        _validate_shard_file(
            temporary,
            expected_ids=shard.image_ids,
            token_dim=token_dim,
            layer_count=layer_count,
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_file_matches_info(path: Path, shard: FeatureShardInfo) -> None:
    if path.stat().st_size != shard.bytes:
        raise ValueError(f"shard size does not match manifest: {path}")
    if sha256_file(path) != shard.sha256:
        raise ValueError(f"shard hash does not match manifest: {path}")


def _copy_file_atomically(
    source: Path,
    destination: Path,
    *,
    replace_existing: bool = False,
) -> None:
    """Atomically copy a non-shard file and verify byte-for-byte equality."""
    if not source.is_file():
        raise FileNotFoundError(f"source file is missing: {source}")
    if destination.is_file() and not replace_existing:
        _ensure_same_file(source, destination)
        return
    temporary = _temporary_path(destination)
    try:
        shutil.copyfile(source, temporary)
        _ensure_same_file(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_same_file(first: Path, second: Path) -> None:
    if first.stat().st_size != second.stat().st_size:
        raise ValueError(f"copied file size mismatch: {first} vs {second}")
    if sha256_file(first) != sha256_file(second):
        raise ValueError(f"copied file hash mismatch: {first} vs {second}")


def _feature_payload(
    features: AllLayerFeatures,
    records: Mapping[str, ImageRecord],
    config: FeatureCacheConfig,
) -> dict[str, Any]:
    feature_dtype = getattr(torch, config.feature_dtype)
    entropy_dtype = getattr(torch, config.entropy_dtype)
    metadata = []
    for record in features.preprocess_records:
        image_record = records[record.image_id]
        metadata.append(
            {
                "image_id": record.image_id,
                "source": image_record.source,
                "split": image_record.split,
                "cluster_id": image_record.cluster_id,
                "image_locator": image_record.image_locator,
                "original_hw": record.original_hw,
                "resized_hw": record.resized_hw,
                "final_hw": record.final_hw,
                "patch_grid_hw": record.patch_grid_hw,
                "extreme_aspect_crop": record.extreme_aspect_crop,
            }
        )
    return {
        "image_ids": list(features.image_ids),
        "cls": features.cls.to(dtype=feature_dtype).contiguous(),
        "mean_patch": features.mean_patch.to(dtype=feature_dtype).contiguous(),
        "cls_guided_patch": features.cls_guided_patch.to(dtype=feature_dtype).contiguous(),
        "pooling_entropy": features.pooling_entropy.to(dtype=entropy_dtype).contiguous(),
        "metadata": metadata,
    }


def _validate_feature_shapes(features: AllLayerFeatures) -> None:
    image_count = len(features.image_ids)
    if len(features.preprocess_records) != image_count:
        raise ValueError("preprocess records do not align with image IDs")
    expected = (image_count, len(features.layer_indices))
    for name, tensor in (
        ("cls", features.cls),
        ("mean_patch", features.mean_patch),
        ("cls_guided_patch", features.cls_guided_patch),
    ):
        if tensor.ndim != 3 or tensor.shape[:2] != expected:
            raise ValueError(f"{name} has invalid shape {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains nonfinite values")
    if features.pooling_entropy.shape != expected:
        raise ValueError("pooling_entropy has invalid shape")
    if not torch.isfinite(features.pooling_entropy).all():
        raise ValueError("pooling_entropy contains nonfinite values")


def _validate_shard_file(
    path: Path,
    *,
    expected_ids: tuple[str, ...],
    token_dim: int,
    layer_count: int,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if tuple(payload.get("image_ids", ())) != expected_ids:
        raise ValueError("temporary shard image IDs differ from requested IDs")
    image_count = len(expected_ids)
    for name in ("cls", "mean_patch", "cls_guided_patch"):
        tensor = payload.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (
            image_count,
            layer_count,
            token_dim,
        ):
            raise ValueError(f"temporary shard has invalid {name} tensor")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"temporary shard {name} has nonfinite values")
    entropy = payload.get("pooling_entropy")
    if not isinstance(entropy, torch.Tensor) or entropy.shape != (image_count, layer_count):
        raise ValueError("temporary shard has invalid pooling entropy")
    if not torch.isfinite(entropy).all():
        raise ValueError("temporary shard entropy has nonfinite values")
    if len(payload.get("metadata", [])) != image_count:
        raise ValueError("temporary shard metadata length is invalid")


def _temporary_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{final_path.name}.", suffix=".tmp", dir=final_path.parent)
    os.close(fd)
    return Path(name)


def _layer_positions(
    cached_indices: Sequence[int],
    requested_indices: Sequence[int] | None,
) -> list[int]:
    requested = tuple(cached_indices if requested_indices is None else requested_indices)
    positions = {int(index): position for position, index in enumerate(cached_indices)}
    try:
        return [positions[int(index)] for index in requested]
    except KeyError as error:
        raise KeyError(f"layer {error.args[0]} is absent from cache") from error
