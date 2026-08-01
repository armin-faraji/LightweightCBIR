"""Deterministic aspect-preserving preprocessing for DINOv2 retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as tvf
from torchvision.transforms.functional import InterpolationMode

from ..config import PreprocessConfig


@dataclass(frozen=True)
class PreprocessRecord:
    image_id: str
    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    final_hw: tuple[int, int]
    patch_grid_hw: tuple[int, int]
    extreme_aspect_crop: bool


def _round_half_up(value: float) -> int:
    return max(1, int(value + 0.5))


def preflight_image_dimensions(
    height: int,
    width: int,
    config: PreprocessConfig,
) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """Return resized HW, final HW, and whether the extreme policy is used."""
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    long_side = max(height, width)
    scale = config.long_side / long_side
    resized_h = _round_half_up(height * scale)
    resized_w = _round_half_up(width * scale)
    patch = config.patch_size

    if min(resized_h, resized_w) >= patch:
        final_h = (resized_h // patch) * patch
        final_w = (resized_w // patch) * patch
        if final_h <= 0 or final_w <= 0:
            raise AssertionError("normal preprocessing produced an empty image")
        return (resized_h, resized_w), (final_h, final_w), False

    # Rare extreme aspect ratio: raise the short axis to one patch and crop
    # only the overlong axis back to the configured long side.
    short_is_height = height <= width
    scale = patch / (height if short_is_height else width)
    resized_h = _round_half_up(height * scale)
    resized_w = _round_half_up(width * scale)
    if short_is_height:
        resized_h = patch
        final_h, final_w = patch, config.long_side
    else:
        resized_w = patch
        final_h, final_w = config.long_side, patch
    return (resized_h, resized_w), (final_h, final_w), True


def _center_crop_to_hw(image: Image.Image, output_hw: tuple[int, int]) -> Image.Image:
    output_h, output_w = output_hw
    width, height = image.size
    if output_h > height or output_w > width:
        raise ValueError(
            f"cannot crop {output_hw} from image size {(height, width)}"
        )
    top = (height - output_h) // 2
    left = (width - output_w) // 2
    return image.crop((left, top, left + output_w, top + output_h))


class ResizeLongestSideToPatchGrid:
    """Resize long side, then trim centrally to a valid DINO patch grid."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config

    def __call__(self, image: Image.Image) -> tuple[Image.Image, tuple[int, int], tuple[int, int], bool]:
        original_h, original_w = image.height, image.width
        resized_hw, final_hw, extreme = preflight_image_dimensions(
            original_h,
            original_w,
            self.config,
        )
        resized_h, resized_w = resized_hw
        image = tvf.resize(
            image,
            [resized_h, resized_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=self.config.antialias,
        )
        image = _center_crop_to_hw(image, final_hw)
        return image, resized_hw, final_hw, extreme


def preprocess_retrieval_image(
    image: Image.Image,
    *,
    image_id: str,
    config: PreprocessConfig,
) -> tuple[torch.Tensor, PreprocessRecord]:
    """Convert one stored-pixel image to normalized CHW tensor and audit record."""
    if config.rgb_only:
        image = image.convert("RGB")
    original_hw = (image.height, image.width)
    transformed, resized_hw, final_hw, extreme = ResizeLongestSideToPatchGrid(config)(image)
    tensor = tvf.to_tensor(transformed)
    tensor = tvf.normalize(tensor, mean=config.mean, std=config.std)
    record = PreprocessRecord(
        image_id=image_id,
        original_hw=original_hw,
        resized_hw=resized_hw,
        final_hw=final_hw,
        patch_grid_hw=(final_hw[0] // config.patch_size, final_hw[1] // config.patch_size),
        extreme_aspect_crop=extreme,
    )
    return tensor, record


def preprocess_many_by_shape(
    images: list[tuple[str, Image.Image]],
    config: PreprocessConfig,
) -> dict[tuple[int, int], tuple[torch.Tensor, list[PreprocessRecord]]]:
    """Preprocess image records and return shape-homogeneous batch tensors."""
    buckets: dict[tuple[int, int], list[tuple[torch.Tensor, PreprocessRecord]]] = {}
    for image_id, image in images:
        tensor, record = preprocess_retrieval_image(image, image_id=image_id, config=config)
        buckets.setdefault(record.final_hw, []).append((tensor, record))
    return {
        shape: (torch.stack([item[0] for item in items]), [item[1] for item in items])
        for shape, items in buckets.items()
    }
