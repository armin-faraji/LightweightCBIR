"""Local sharded cache for frozen image features."""

from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import FeatureCacheConfig
from .data.sfm import ImageRecord
from .features import AllLayerFeatures
from .utils import atomic_write_json, hash_strings, read_json


MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2
REVISITOP_FEATURES_NAME = "features.pt"
REVISITOP_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureShardInfo:
    name: str
    image_ids: tuple[str, ...]
    bytes: int


@dataclass(frozen=True)
class FeatureManifest:
    """The one durable record of a local feature cache."""

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
    complete: bool = False
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
                {**asdict(shard), "image_ids": list(shard.image_ids)}
                for shard in self.shards
            ],
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureManifest":
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported feature-cache schema version")
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
                    bytes=int(shard["bytes"]),
                )
                for shard in payload.get("shards", [])
            ),
            complete=bool(payload.get("complete", False)),
        )


class FeatureShardWriter:
    """Append atomic feature shards and update one lightweight manifest."""

    def __init__(
        self,
        cache_dir: Path,
        manifest: FeatureManifest,
        config: FeatureCacheConfig,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.shards_dir = self.cache_dir / "shards"
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.cache_dir / MANIFEST_NAME
        if manifest_path.is_file():
            existing = FeatureManifest.from_dict(read_json(manifest_path))
            if not _same_manifest_identity(existing, manifest):
                raise ValueError("cache manifest does not match this extraction")
            self.manifest = existing
        else:
            self.manifest = manifest
            atomic_write_json(manifest_path, manifest.to_dict())

    @property
    def completed_ids(self) -> frozenset[str]:
        return frozenset(image_id for shard in self.manifest.shards for image_id in shard.image_ids)

    def write_shard(
        self,
        features: AllLayerFeatures,
        records: Mapping[str, ImageRecord],
    ) -> FeatureShardInfo:
        if self.manifest.complete:
            raise RuntimeError("cannot append to a completed feature cache")
        _validate_features(features, self.manifest)
        image_ids = tuple(features.image_ids)
        if set(image_ids) != set(records):
            raise ValueError("records must match the shard image IDs")
        unknown = set(image_ids) - set(self.manifest.expected_image_ids)
        overlap = set(image_ids) & self.completed_ids
        if unknown or overlap:
            raise ValueError("feature shard contains unexpected or already cached image IDs")

        name = self._next_shard_name()
        destination = self.shards_dir / name
        temporary = _temporary_path(destination)
        try:
            torch.save(_feature_payload(features, records, self.config), temporary)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        info = FeatureShardInfo(name=name, image_ids=image_ids, bytes=destination.stat().st_size)
        self.manifest = _replace_manifest(self.manifest, shards=(*self.manifest.shards, info))
        atomic_write_json(self.cache_dir / MANIFEST_NAME, self.manifest.to_dict())
        return info

    def finalize(self) -> FeatureManifest:
        expected = set(self.manifest.expected_image_ids)
        if self.completed_ids != expected:
            raise ValueError(f"cannot finalize incomplete cache ({len(expected - self.completed_ids)} images missing)")
        self.manifest = _replace_manifest(self.manifest, complete=True)
        atomic_write_json(self.cache_dir / MANIFEST_NAME, self.manifest.to_dict())
        return self.manifest

    def _next_shard_name(self) -> str:
        index = len(self.manifest.shards)
        while (self.shards_dir / f"shard_{index:05d}.pt").exists():
            index += 1
        return f"shard_{index:05d}.pt"


class FeatureShardReader:
    """Read cached tensors by image ID; ``preload=True`` keeps all shards in RAM."""

    def __init__(
        self,
        cache_dir: Path,
        max_cached_shards: int | None = 3,
        *,
        preload: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        report = validate_feature_cache(self.cache_dir)
        if not report["valid"]:
            raise ValueError(f"invalid feature cache: {report['errors']}")
        self.manifest = FeatureManifest.from_dict(read_json(self.cache_dir / MANIFEST_NAME))
        self._locations = {
            image_id: (shard_index, offset)
            for shard_index, shard in enumerate(self.manifest.shards)
            for offset, image_id in enumerate(shard.image_ids)
        }
        self.max_cached_shards = max_cached_shards
        self._loaded: OrderedDict[int, Mapping[str, Any]] = OrderedDict()
        if preload:
            self.preload()

    @property
    def image_ids(self) -> tuple[str, ...]:
        return self.manifest.expected_image_ids

    def preload(self) -> None:
        """Load every shard into CPU RAM for repeated training experiments."""
        self.max_cached_shards = None
        for index in range(len(self.manifest.shards)):
            self._load_shard(index)

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
        }

    def _load_shard(self, shard_index: int) -> Mapping[str, Any]:
        if shard_index in self._loaded:
            self._loaded.move_to_end(shard_index)
            return self._loaded[shard_index]
        info = self.manifest.shards[shard_index]
        payload = torch.load(self.cache_dir / "shards" / info.name, map_location="cpu", weights_only=False)
        if tuple(payload.get("image_ids", ())) != info.image_ids:
            raise ValueError(f"shard IDs do not match manifest: {info.name}")
        self._loaded[shard_index] = payload
        if self.max_cached_shards is not None:
            while len(self._loaded) > self.max_cached_shards:
                self._loaded.popitem(last=False)
        return payload


def validate_feature_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: str | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Do cheap manifest and file-presence checks without rehashing/decompressing shards."""
    cache_dir = Path(cache_dir)
    try:
        manifest = FeatureManifest.from_dict(read_json(cache_dir / MANIFEST_NAME))
    except Exception as error:
        return {"valid": False, "errors": [f"invalid manifest: {error}"], "manifest": None}

    errors: list[str] = []
    if expected_fingerprint is not None and manifest.fingerprint != expected_fingerprint:
        errors.append("cache fingerprint does not match request")
    if hash_strings(manifest.expected_image_ids) != manifest.source_ids_hash:
        errors.append("manifest image-ID hash does not match expected IDs")
    expected = set(manifest.expected_image_ids)
    if len(expected) != len(manifest.expected_image_ids):
        errors.append("manifest contains duplicate expected image IDs")
    seen: set[str] = set()
    names: set[str] = set()
    for shard in manifest.shards:
        if Path(shard.name).name != shard.name or not shard.name.endswith(".pt"):
            errors.append(f"invalid shard name {shard.name!r}")
        if shard.name in names:
            errors.append(f"duplicate shard name {shard.name}")
        names.add(shard.name)
        if not shard.image_ids or len(set(shard.image_ids)) != len(shard.image_ids):
            errors.append(f"invalid image IDs in {shard.name}")
        if not set(shard.image_ids).issubset(expected):
            errors.append(f"unexpected image IDs in {shard.name}")
        if seen.intersection(shard.image_ids):
            errors.append(f"duplicate image IDs across shards near {shard.name}")
        seen.update(shard.image_ids)
        path = cache_dir / "shards" / shard.name
        if not path.is_file():
            errors.append(f"missing shard {shard.name}")
        elif path.stat().st_size != shard.bytes:
            errors.append(f"size mismatch for shard {shard.name}")
    if manifest.complete and seen != expected:
        errors.append("complete cache does not contain every expected image")
    if require_complete and not manifest.complete:
        errors.append("cache is incomplete")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest": manifest.to_dict(),
        "completed_image_count": len(seen),
        "expected_image_count": len(expected),
        "is_complete": manifest.complete and seen == expected and not errors,
    }


def save_revisitop_feature_cache(
    cache_dir: Path,
    *,
    fingerprint: str,
    extraction_config: Mapping[str, Any],
    database: AllLayerFeatures,
    queries: AllLayerFeatures,
    config: FeatureCacheConfig,
) -> Path:
    """Atomically save one compact database/query cache for a RevisitOP benchmark."""
    if database.layer_indices != queries.layer_indices:
        raise ValueError("database and query feature layers must match")
    token_dim = int(database.cls.shape[-1])
    manifest = {
        "schema_version": REVISITOP_SCHEMA_VERSION,
        "kind": "revisitop_feature_cache",
        "fingerprint": fingerprint,
        "extraction_config": dict(extraction_config),
        "layer_indices": list(database.layer_indices),
        "token_dim": token_dim,
        "feature_dtype": config.feature_dtype,
        "entropy_dtype": config.entropy_dtype,
        "database_image_ids": list(database.image_ids),
        "query_image_ids": list(queries.image_ids),
        "complete": True,
    }
    _validate_bundle_features(database, token_dim)
    _validate_bundle_features(queries, token_dim)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / MANIFEST_NAME
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("fingerprint") != fingerprint:
            raise ValueError("RevisitOP cache exists with a different fingerprint")
        if validate_revisitop_feature_cache(
            cache_dir,
            expected_fingerprint=fingerprint,
        )["valid"]:
            return cache_dir
    destination = cache_dir / REVISITOP_FEATURES_NAME
    temporary = _temporary_path(destination)
    try:
        torch.save(
            {
                "database": _bundle_payload(database, config),
                "queries": _bundle_payload(queries, config),
            },
            temporary,
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    atomic_write_json(manifest_path, manifest)
    return cache_dir


def validate_revisitop_feature_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Check RevisitOP cache metadata and file presence without reopening tensors."""
    cache_dir = Path(cache_dir)
    try:
        manifest = read_json(cache_dir / MANIFEST_NAME)
    except Exception as error:
        return {"valid": False, "errors": [f"invalid manifest: {error}"], "manifest": None}
    errors: list[str] = []
    if manifest.get("schema_version") != REVISITOP_SCHEMA_VERSION:
        errors.append("unsupported RevisitOP cache schema")
    if manifest.get("kind") != "revisitop_feature_cache":
        errors.append("manifest is not a RevisitOP feature cache")
    if expected_fingerprint is not None and manifest.get("fingerprint") != expected_fingerprint:
        errors.append("cache fingerprint does not match request")
    if not manifest.get("complete"):
        errors.append("cache is incomplete")
    if not (cache_dir / REVISITOP_FEATURES_NAME).is_file():
        errors.append("features.pt is missing")
    for field in ("database_image_ids", "query_image_ids"):
        values = tuple(str(value) for value in manifest.get(field, ()))
        if not values or len(values) != len(set(values)):
            errors.append(f"invalid {field}")
    return {"valid": not errors, "errors": errors, "manifest": manifest}


def load_revisitop_feature_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Load database/query feature tensors into CPU RAM."""
    cache_dir = Path(cache_dir)
    report = validate_revisitop_feature_cache(
        cache_dir,
        expected_fingerprint=expected_fingerprint,
    )
    if not report["valid"]:
        raise ValueError(f"invalid RevisitOP feature cache: {report['errors']}")
    payload = torch.load(cache_dir / REVISITOP_FEATURES_NAME, map_location="cpu", weights_only=False)
    manifest = report["manifest"]
    for key, manifest_key in (("database", "database_image_ids"), ("queries", "query_image_ids")):
        if tuple(payload.get(key, {}).get("image_ids", ())) != tuple(manifest[manifest_key]):
            raise ValueError(f"{key} image IDs do not match the manifest")
    return payload


def _same_manifest_identity(first: FeatureManifest, second: FeatureManifest) -> bool:
    return (
        first.fingerprint == second.fingerprint
        and first.dataset_name == second.dataset_name
        and first.source_ids_hash == second.source_ids_hash
        and dict(first.extraction_config) == dict(second.extraction_config)
        and first.layer_indices == second.layer_indices
        and first.token_dim == second.token_dim
        and first.feature_dtype == second.feature_dtype
        and first.entropy_dtype == second.entropy_dtype
        and first.expected_image_ids == second.expected_image_ids
    )


def _replace_manifest(
    manifest: FeatureManifest,
    *,
    shards: tuple[FeatureShardInfo, ...] | None = None,
    complete: bool | None = None,
) -> FeatureManifest:
    return FeatureManifest(
        fingerprint=manifest.fingerprint,
        dataset_name=manifest.dataset_name,
        source_ids_hash=manifest.source_ids_hash,
        extraction_config=manifest.extraction_config,
        layer_indices=manifest.layer_indices,
        token_dim=manifest.token_dim,
        feature_dtype=manifest.feature_dtype,
        entropy_dtype=manifest.entropy_dtype,
        expected_image_ids=manifest.expected_image_ids,
        shards=manifest.shards if shards is None else shards,
        complete=manifest.complete if complete is None else complete,
    )


def _feature_payload(
    features: AllLayerFeatures,
    records: Mapping[str, ImageRecord],
    config: FeatureCacheConfig,
) -> dict[str, Any]:
    feature_dtype = getattr(torch, config.feature_dtype)
    entropy_dtype = getattr(torch, config.entropy_dtype)
    metadata = [
        {
            "image_id": record.image_id,
            "source": records[record.image_id].source,
            "split": records[record.image_id].split,
            "cluster_id": records[record.image_id].cluster_id,
            "image_locator": records[record.image_id].image_locator,
            "original_hw": record.original_hw,
            "resized_hw": record.resized_hw,
            "final_hw": record.final_hw,
            "patch_grid_hw": record.patch_grid_hw,
            "extreme_aspect_crop": record.extreme_aspect_crop,
        }
        for record in features.preprocess_records
    ]
    return {
        "image_ids": list(features.image_ids),
        "cls": features.cls.to(dtype=feature_dtype).contiguous(),
        "mean_patch": features.mean_patch.to(dtype=feature_dtype).contiguous(),
        "cls_guided_patch": features.cls_guided_patch.to(dtype=feature_dtype).contiguous(),
        "pooling_entropy": features.pooling_entropy.to(dtype=entropy_dtype).contiguous(),
        "metadata": metadata,
    }


def _bundle_payload(features: AllLayerFeatures, config: FeatureCacheConfig) -> dict[str, Any]:
    return {
        "image_ids": list(features.image_ids),
        "cls": features.cls.to(dtype=getattr(torch, config.feature_dtype)).contiguous(),
        "mean_patch": features.mean_patch.to(dtype=getattr(torch, config.feature_dtype)).contiguous(),
        "cls_guided_patch": features.cls_guided_patch.to(
            dtype=getattr(torch, config.feature_dtype)
        ).contiguous(),
        "pooling_entropy": features.pooling_entropy.to(
            dtype=getattr(torch, config.entropy_dtype)
        ).contiguous(),
    }


def _validate_features(features: AllLayerFeatures, manifest: FeatureManifest) -> None:
    image_count = len(features.image_ids)
    if not image_count or len(set(features.image_ids)) != image_count:
        raise ValueError("feature shard needs unique image IDs")
    if tuple(features.layer_indices) != manifest.layer_indices:
        raise ValueError("feature layers do not match cache manifest")
    expected = (image_count, len(manifest.layer_indices), manifest.token_dim)
    for name, tensor in (
        ("cls", features.cls),
        ("mean_patch", features.mean_patch),
        ("cls_guided_patch", features.cls_guided_patch),
    ):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected}")
    if tuple(features.pooling_entropy.shape) != expected[:2]:
        raise ValueError("pooling_entropy shape does not match cache manifest")
    if len(features.preprocess_records) != image_count:
        raise ValueError("preprocess records do not match features")


def _validate_bundle_features(features: AllLayerFeatures, token_dim: int) -> None:
    if not features.image_ids or len(set(features.image_ids)) != len(features.image_ids):
        raise ValueError("RevisitOP features need unique image IDs")
    expected = (len(features.image_ids), len(features.layer_indices), token_dim)
    for name, tensor in (
        ("cls", features.cls),
        ("mean_patch", features.mean_patch),
        ("cls_guided_patch", features.cls_guided_patch),
    ):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} has an invalid RevisitOP feature shape")
    if tuple(features.pooling_entropy.shape) != expected[:2]:
        raise ValueError("pooling_entropy has an invalid RevisitOP feature shape")


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    return Path(name)


def _layer_positions(cached: Sequence[int], requested: Sequence[int] | None) -> list[int]:
    positions = {int(index): position for position, index in enumerate(cached)}
    selected = tuple(cached if requested is None else requested)
    try:
        return [positions[int(index)] for index in selected]
    except KeyError as error:
        raise KeyError(f"layer {error.args[0]} is absent from cache") from error
