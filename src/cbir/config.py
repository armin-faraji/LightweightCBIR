"""Small, local-only configuration objects and reproducibility fingerprints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from .utils import stable_hash


@dataclass(frozen=True)
class BackboneConfig:
    repo: str = "facebookresearch/dinov2"
    entrypoint: str = "dinov2_vits14_reg"
    revision: str | None = None
    token_dim: int = 384
    num_blocks: int = 12
    patch_size: int = 14
    num_register_tokens: int = 4
    device: str = "cuda"
    use_amp: bool = True


@dataclass(frozen=True)
class PreprocessConfig:
    long_side: int = 224
    patch_size: int = 14
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.long_side <= 0 or self.patch_size <= 0:
            raise ValueError("long_side and patch_size must be positive")
        if self.long_side % self.patch_size:
            raise ValueError("long_side must be divisible by patch_size")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each contain three RGB values")


@dataclass(frozen=True)
class PoolingConfig:
    temperature: float = 0.025
    eps: float = 1e-12
    all_layer_indices: tuple[int, ...] = tuple(range(12))

    def __post_init__(self) -> None:
        if self.temperature <= 0 or self.eps <= 0:
            raise ValueError("pooling temperature and eps must be positive")


@dataclass(frozen=True)
class FeatureCacheConfig:
    root: Path = Path("data/cache")
    shard_size: int = 1000
    feature_dtype: Literal["float16", "float32"] = "float16"
    entropy_dtype: Literal["float16", "float32"] = "float32"

    def __post_init__(self) -> None:
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")


@dataclass(frozen=True)
class SfmConfig:
    metadata_path: Path = Path("data/raw/sfm30k/retrieval-SfM-120k.pkl")
    names_clusters_path: Path = Path(
        "data/raw/sfm30k/retrieval-SfM-30k-imagenames-clusterids.mat"
    )
    image_mat_path: Path = Path("data/raw/sfm30k/retrieval-SfM-30k.mat")


@dataclass(frozen=True)
class FusionConfig:
    token_dim: int = 384
    layer_indices: tuple[int, ...] = (7, 9, 11)
    output_dim: int = 128
    head_kind: Literal["global_local", "cls_concat", "final_cls_projection"] = "cls_concat"
    local_kind: Literal["cls_guided_patch", "mean_patch"] = "cls_guided_patch"
    gate_mode: Literal["uniform", "static", "dynamic"] | None = None
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must be nonempty and unique")
        if self.token_dim <= 0 or self.output_dim <= 0 or self.eps <= 0:
            raise ValueError("invalid fusion dimensions or epsilon")
        if self.head_kind == "global_local":
            if self.gate_mode not in {"uniform", "static", "dynamic"}:
                raise ValueError("global_local requires a layer-weighting mode")
        elif self.gate_mode is not None:
            raise ValueError(f"{self.head_kind} must use gate_mode=None")
        if self.head_kind == "final_cls_projection" and len(self.layer_indices) != 1:
            raise ValueError("final_cls_projection requires exactly one layer")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_temperature: float = 0.07
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

    def __post_init__(self) -> None:
        if self.query_block_size <= 0:
            raise ValueError("query_block_size must be positive")


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
    """Load the one local project YAML file."""
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
    if "root" in result:
        result["root"] = Path(result["root"])
    return result


def _coerce_sfm(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: Path(item) for key, item in dict(value).items()}


def _coerce_fusion(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "layer_indices" in result:
        result["layer_indices"] = tuple(int(item) for item in result["layer_indices"])
    return result


def fusion_config_from_dict(value: Mapping[str, Any]) -> FusionConfig:
    """Recreate a fusion configuration stored in a checkpoint or selection."""
    return FusionConfig(**_coerce_fusion(value))


def config_to_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass configuration to JSON-friendly primitives."""
    return _normalise(asdict(config))


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
    return stable_hash(
        {
            "backbone": config_to_dict(backbone),
            "preprocess": config_to_dict(preprocess),
            "pooling": config_to_dict(pooling),
            "source_ids_hash": source_ids_hash,
        }
    )


def train_fingerprint(
    *,
    cache_fingerprint: str,
    fusion: FusionConfig,
    training: TrainingConfig,
) -> str:
    """Hash an experiment without invalidating the frozen feature cache."""
    return stable_hash(
        {
            "cache_fingerprint": cache_fingerprint,
            "fusion": config_to_dict(fusion),
            "training": config_to_dict(training),
        }
    )
