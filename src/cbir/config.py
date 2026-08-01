"""Configuration objects and stable run/cache fingerprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from .utils import stable_hash


@dataclass(frozen=True)
class BackboneConfig:
    repo: str = "facebookresearch/dinov2"
    entrypoint: str = "dinov2_vits14_reg"
    revision: str | None = None
    checkpoint_id: str | None = None
    token_dim: int = 384
    num_blocks: int = 12
    patch_size: int = 14
    num_register_tokens: int = 4
    device: str = "cuda"
    use_amp: bool = False


@dataclass(frozen=True)
class PreprocessConfig:
    long_side: int = 224
    patch_size: int = 14
    interpolation: Literal["bicubic"] = "bicubic"
    antialias: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    rgb_only: bool = True
    extreme_aspect_policy: Literal["crop_long_axis"] = "crop_long_axis"

    def __post_init__(self) -> None:
        if self.long_side <= 0 or self.patch_size <= 0:
            raise ValueError("long_side and patch_size must be positive")
        if self.long_side % self.patch_size != 0:
            raise ValueError("long_side must be divisible by patch_size")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each contain three RGB values")


@dataclass(frozen=True)
class PoolingConfig:
    temperature: float = 0.1
    eps: float = 1e-12
    aggregation_version: str = "v1"
    all_layer_indices: tuple[int, ...] = tuple(range(12))

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("pooling temperature must be positive")
        if self.eps <= 0:
            raise ValueError("pooling eps must be positive")


@dataclass(frozen=True)
class FeatureCacheConfig:
    local_root: Path = Path("data/cache")
    drive_root: Path | None = None
    shard_size: int = 750
    feature_dtype: Literal["float16", "float32"] = "float16"
    entropy_dtype: Literal["float32"] = "float32"
    cache_format_version: str = "v1"

    def __post_init__(self) -> None:
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


@dataclass(frozen=True)
class SfmConfig:
    metadata_path: Path = Path("data/sfm/metadata.pkl")
    names_clusters_path: Path | None = None
    image_mat_path: Path | None = Path("data/sfm/retrieval-SfM-30k.mat")
    image_root: Path | None = None
    source_name: str = "retrieval-SfM-30k"


@dataclass(frozen=True)
class FusionConfig:
    token_dim: int = 384
    layer_indices: tuple[int, ...] = (3, 7, 11)
    output_dim: int = 256
    local_kind: Literal["cls_guided_patch", "mean_patch"] = "cls_guided_patch"
    gate_mode: Literal["uniform", "static", "reliability"] = "reliability"
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not self.layer_indices:
            raise ValueError("at least one layer must be selected")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must not contain duplicates")
        if self.token_dim <= 0 or self.output_dim <= 0 or self.eps <= 0:
            raise ValueError("invalid fusion dimensions or epsilon")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_temperature: float = 0.07
    num_workers: int = 0
    early_stopping_patience: int = 8
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.batch_size <= 1 or self.epochs <= 0:
            raise ValueError("batch_size must exceed one and epochs must be positive")
        if self.learning_rate <= 0 or self.loss_temperature <= 0:
            raise ValueError("learning_rate and loss_temperature must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    query_block_size: int = 256
    bootstrap_replicates: int = 0
    bootstrap_seed: int = 13

    def __post_init__(self) -> None:
        if self.query_block_size <= 0 or self.bootstrap_replicates < 0:
            raise ValueError("invalid evaluation configuration")


@dataclass(frozen=True)
class ProjectConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)
    sfm: SfmConfig = field(default_factory=SfmConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_project_config(path: Path) -> ProjectConfig:
    """Load a YAML project configuration with tuple/path coercion."""
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to load project configurations") from error
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("project YAML must contain a top-level mapping")
    return ProjectConfig(
        backbone=BackboneConfig(**dict(raw.get("backbone", {}))),
        preprocess=PreprocessConfig(**_coerce_preprocess(raw.get("preprocess", {}))),
        pooling=PoolingConfig(**_coerce_pooling(raw.get("pooling", {}))),
        cache=FeatureCacheConfig(**_coerce_cache(raw.get("cache", {}))),
        sfm=SfmConfig(**_coerce_sfm(raw.get("sfm", {}))),
        fusion=FusionConfig(**_coerce_fusion(raw.get("fusion", {}))),
        training=TrainingConfig(**dict(raw.get("training", {}))),
        evaluation=EvaluationConfig(**dict(raw.get("evaluation", {}))),
    )


def _coerce_preprocess(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("mean", "std"):
        if key in result:
            result[key] = tuple(float(item) for item in result[key])
    return result


def _coerce_pooling(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "all_layer_indices" in result:
        result["all_layer_indices"] = tuple(int(item) for item in result["all_layer_indices"])
    return result


def _coerce_cache(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("local_root", "drive_root"):
        if result.get(key) is not None:
            result[key] = Path(result[key])
    return result


def _coerce_sfm(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("metadata_path", "names_clusters_path", "image_mat_path", "image_root"):
        if result.get(key) is not None:
            result[key] = Path(result[key])
    return result


def _coerce_fusion(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "layer_indices" in result:
        result["layer_indices"] = tuple(int(item) for item in result["layer_indices"])
    return result


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass configuration to JSON-friendly nested primitives."""
    value = asdict(config)
    return _normalise(value)


def _normalise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalise(item) for item in value]
    if isinstance(value, list):
        return [_normalise(item) for item in value]
    return value


def extraction_fingerprint(
    *,
    backbone: BackboneConfig,
    preprocess: PreprocessConfig,
    pooling: PoolingConfig,
    source_ids_hash: str,
) -> str:
    """Hash only choices that change frozen extracted features."""
    payload = {
        "backbone": config_to_dict(backbone),
        "preprocess": config_to_dict(preprocess),
        "pooling": config_to_dict(pooling),
        "source_ids_hash": source_ids_hash,
    }
    return stable_hash(payload)


def train_fingerprint(
    *,
    cache_fingerprint: str,
    fusion: FusionConfig,
    training: TrainingConfig,
) -> str:
    """Hash a trainable-head experiment without invalidating the feature cache."""
    return stable_hash(
        {
            "cache_fingerprint": cache_fingerprint,
            "fusion": config_to_dict(fusion),
            "training": config_to_dict(training),
        }
    )
