from __future__ import annotations

import unittest

from PIL import Image

from cbir.config import PreprocessConfig
from cbir.data.transforms import preflight_image_dimensions, preprocess_retrieval_image


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PreprocessConfig(long_side=224, patch_size=14)

    def test_landscape_image_keeps_aspect_and_grid(self) -> None:
        image = Image.new("RGB", (400, 200))
        tensor, record = preprocess_retrieval_image(
            image,
            image_id="landscape",
            config=self.config,
        )
        self.assertEqual(record.final_hw, (112, 224))
        self.assertEqual(tuple(tensor.shape), (3, 112, 224))
        self.assertFalse(record.extreme_aspect_crop)

    def test_portrait_image_keeps_aspect_and_grid(self) -> None:
        image = Image.new("RGB", (200, 400))
        _, record = preprocess_retrieval_image(
            image,
            image_id="portrait",
            config=self.config,
        )
        self.assertEqual(record.final_hw, (224, 112))
        self.assertEqual(record.patch_grid_hw, (16, 8))

    def test_extreme_aspect_is_cropped_and_logged(self) -> None:
        resized, final, extreme = preflight_image_dimensions(1, 1000, self.config)
        self.assertTrue(extreme)
        self.assertEqual(final, (14, 224))
        self.assertGreater(resized[1], 224)


if __name__ == "__main__":
    unittest.main()

