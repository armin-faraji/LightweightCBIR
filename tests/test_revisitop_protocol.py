from __future__ import annotations

import unittest

from PIL import Image

from cbir.data.revisitop import RevisitGroundTruth, crop_revisit_query
from cbir.evaluation import evaluate_revisitop


class RevisitProtocolTests(unittest.TestCase):
    def test_easy_medium_hard_ignore_semantics(self) -> None:
        gnd = {
            "q": RevisitGroundTruth(
                query_id="q",
                easy=frozenset({"easy"}),
                hard=frozenset({"hard"}),
                junk=frozenset({"junk"}),
            )
        }
        report = evaluate_revisitop(
            [["junk", "easy", "hard", "other"]],
            ["q"],
            gnd,
            protocols=("easy", "medium", "hard"),
        )
        self.assertAlmostEqual(report.easy.map, 1.0)
        self.assertAlmostEqual(report.medium.map, 1.0)
        self.assertAlmostEqual(report.hard.map, 1.0)
        self.assertAlmostEqual(report.medium.mean_precision_at_10, 1.0)

    def test_query_crop_uses_xyxy(self) -> None:
        image = Image.new("RGB", (20, 10))
        crop = crop_revisit_query(image, (2, 3, 12, 8))
        self.assertEqual(crop.size, (10, 5))

    def test_precision_and_ap_follow_official_rank_correction(self) -> None:
        gnd = {
            "q": RevisitGroundTruth(
                query_id="q",
                easy=frozenset({"easy"}),
                hard=frozenset({"hard"}),
                junk=frozenset({"junk"}),
            )
        }
        report = evaluate_revisitop(
            [["junk", "other", "easy", "hard"]],
            ["q"],
            gnd,
        )
        # Medium positives are at post-junk zero-based ranks 1 and 2.
        self.assertAlmostEqual(report.medium.map, 5.0 / 12.0)
        self.assertAlmostEqual(report.medium.mean_precision_at_10, 2.0 / 3.0)
        # In Hard, Easy and Junk are ignored, leaving the one hard positive at rank 1.
        self.assertAlmostEqual(report.hard.map, 0.25)
        self.assertAlmostEqual(report.hard.mean_precision_at_10, 0.5)


if __name__ == "__main__":
    unittest.main()
