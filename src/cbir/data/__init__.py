"""Dataset metadata, readers, and deterministic image transforms."""

from .revisitop import RevisitGroundTruth, RevisitOPDataset, RevisitQuery
from .revisitop_prepare import (
    REVISITOP_DATASETS,
    OFFICIAL_REVISITOP_COUNTS,
    RevisitOPUrls,
    prepare_revisitop_datasets,
    validate_revisitop_dataset,
)
from .download import SfmUrls, download_with_resume
from .sfm import (
    ImageRecord,
    PairRecord,
    Sfm30kMetadata,
    SfmMatImageReader,
    ValidationCase,
)
from .transforms import PreprocessRecord, ResizeLongestSideToPatchGrid, preprocess_retrieval_image

__all__ = [
    "ImageRecord",
    "PairRecord",
    "PreprocessRecord",
    "ResizeLongestSideToPatchGrid",
    "RevisitGroundTruth",
    "RevisitOPDataset",
    "RevisitQuery",
    "REVISITOP_DATASETS",
    "OFFICIAL_REVISITOP_COUNTS",
    "RevisitOPUrls",
    "SfmUrls",
    "Sfm30kMetadata",
    "SfmMatImageReader",
    "ValidationCase",
    "download_with_resume",
    "preprocess_retrieval_image",
    "prepare_revisitop_datasets",
    "validate_revisitop_dataset",
]
