from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from cbir.data.sfm import Sfm30kMetadata, canonicalize_train_pairs


class SfmMetadataTests(unittest.TestCase):
    def test_official_style_join_and_pair_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            cids_train = ["train000", "train001", "train002"]
            cids_val = ["val00000", "val00001"]
            raw = {
                "train": {
                    "cids": cids_train,
                    "cluster": [1, 1, 1],
                    "qidxs": [0, 1, 1, 2],
                    "pidxs": [1, 0, 1, 0],
                },
                "val": {
                    "cids": cids_val,
                    "cluster": [2, 2],
                    "qidxs": [0],
                    "pidxs": [1],
                },
            }
            pickle_path = root / "metadata.pkl"
            with pickle_path.open("wb") as handle:
                pickle.dump(raw, handle)
            mat_path = root / "selection.mat"
            savemat(
                mat_path,
                {
                    "cids": np.asarray([cids_train + cids_val], dtype=object),
                    "cluster": np.asarray([[1, 1, 1, 2, 2]], dtype=np.uint16),
                },
            )
            metadata = Sfm30kMetadata.from_official_files(pickle_path, mat_path)
            self.assertEqual(metadata.validate()["train_pairs"], 2)
            self.assertEqual(len(metadata.build_validation_cases()), 1)
            self.assertEqual(metadata.images["val00000"].image_locator, 0)


if __name__ == "__main__":
    unittest.main()

