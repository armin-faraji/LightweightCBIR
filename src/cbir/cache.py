"""Sharded, resumable cache for frozen all-layer feature aggregates."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
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
            if existing.fingerprint != manifest.fingerprint:
                raise ValueError("refusing to write into a cache with another fingerprint")
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
    """Local-first cache resolution with optional validated Drive mirroring."""

    def __init__(self, local_cache_dir: Path, drive_cache_dir: Path | None = None) -> None:
        self.local_cache_dir = Path(local_cache_dir)
        self.drive_cache_dir = None if drive_cache_dir is None else Path(drive_cache_dir)

    def resolve_existing(self, expected_fingerprint: str) -> Path | None:
        local = validate_feature_cache(
            self.local_cache_dir,
            expected_fingerprint=expected_fingerprint,
        )
        if local["valid"]:
            return self.local_cache_dir
        if self.drive_cache_dir is None:
            return None
        drive = validate_feature_cache(
            self.drive_cache_dir,
            expected_fingerprint=expected_fingerprint,
        )
        if not drive["valid"]:
            return None
        self.restore_drive_to_local()
        restored = validate_feature_cache(
            self.local_cache_dir,
            expected_fingerprint=expected_fingerprint,
        )
        if not restored["valid"]:
            raise RuntimeError(f"Drive restore failed validation: {restored['errors']}")
        return self.local_cache_dir

    def restore_drive_to_local(self) -> None:
        if self.drive_cache_dir is None:
            raise RuntimeError("Drive cache directory is not configured")
        if self.local_cache_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing local cache directory {self.local_cache_dir}"
            )
        temporary = self.local_cache_dir.with_name(f".{self.local_cache_dir.name}.restore")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(self.drive_cache_dir, temporary)
        os.replace(temporary, self.local_cache_dir)

    def mirror_local_to_drive(self) -> None:
        if self.drive_cache_dir is None:
            return
        report = validate_feature_cache(self.local_cache_dir)
        if not report["valid"]:
            raise ValueError(f"refusing to mirror invalid local cache: {report['errors']}")
        if self.drive_cache_dir.exists():
            existing = validate_feature_cache(
                self.drive_cache_dir,
                expected_fingerprint=report["manifest"]["fingerprint"],
            )
            if existing["valid"]:
                return
            raise FileExistsError(
                f"Drive target exists but is invalid/incompatible: {self.drive_cache_dir}"
            )
        temporary = self.drive_cache_dir.with_name(f".{self.drive_cache_dir.name}.mirror")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(self.local_cache_dir, temporary)
        os.replace(temporary, self.drive_cache_dir)


def validate_feature_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: str | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    errors: list[str] = []
    manifest_path = cache_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return {"valid": False, "errors": ["manifest.json is missing"], "manifest": None}
    try:
        manifest = FeatureManifest.from_dict(read_json(manifest_path))
    except Exception as error:
        return {"valid": False, "errors": [f"invalid manifest: {error}"], "manifest": None}
    if expected_fingerprint is not None and manifest.fingerprint != expected_fingerprint:
        errors.append("cache fingerprint does not match request")
    if hash_strings(manifest.expected_image_ids) != manifest.source_ids_hash:
        errors.append("manifest source_ids_hash does not match expected IDs")
    seen: set[str] = set()
    for shard in manifest.shards:
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
    if seen != set(manifest.expected_image_ids):
        errors.append(
            f"shard IDs do not exactly match manifest expectation "
            f"(missing={len(set(manifest.expected_image_ids) - seen)})"
        )
    if require_complete and not (cache_dir / COMPLETE_NAME).is_file():
        errors.append("COMPLETE marker is missing")
    return {"valid": not errors, "errors": errors, "manifest": manifest.to_dict()}


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
