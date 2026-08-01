from __future__ import annotations

import unittest

import torch

from cbir.data.sfm import PairRecord
from cbir.training import ClusterUniquePairBatchSampler, symmetric_info_nce


class TrainingTests(unittest.TestCase):
    def test_sampler_has_unique_clusters_in_each_batch(self) -> None:
        pairs = tuple(
            PairRecord(f"q{index}", f"p{index}", index % 3, "train")
            for index in range(12)
        )
        sampler = ClusterUniquePairBatchSampler(pairs, batch_size=3, seed=3)
        for batch in sampler:
            clusters = [pairs[item.pair_index].cluster_id for item in batch]
            self.assertEqual(len(clusters), len(set(clusters)))
            self.assertGreaterEqual(len(batch), 2)

    def test_symmetric_infonce_is_finite(self) -> None:
        query = torch.eye(3)
        loss = symmetric_info_nce(query, query, temperature=0.1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
