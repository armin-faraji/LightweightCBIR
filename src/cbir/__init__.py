"""Lightweight content-based image retrieval with frozen DINOv2 features."""

from .config import (
    BackboneConfig,
    EvaluationConfig,
    FeatureCacheConfig,
    FusionConfig,
    PoolingConfig,
    PreprocessConfig,
    ProjectConfig,
    SfmConfig,
    TrainingConfig,
    load_project_config,
)
from .fusion import (
    FinalClsProjectionHead,
    MultiLevelClsConcatHead,
    MultiLevelGlobalLocalFusion,
    build_descriptor_head,
)

__all__ = [
    "BackboneConfig",
    "EvaluationConfig",
    "FeatureCacheConfig",
    "FusionConfig",
    "PoolingConfig",
    "PreprocessConfig",
    "ProjectConfig",
    "FinalClsProjectionHead",
    "MultiLevelClsConcatHead",
    "MultiLevelGlobalLocalFusion",
    "SfmConfig",
    "TrainingConfig",
    "build_descriptor_head",
    "load_project_config",
]
