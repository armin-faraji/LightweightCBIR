"""retrieval-SfM metadata handling and lazy image readers."""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    source: Literal["sfm"] = "sfm"
    split: Literal["train", "val"] | None = None
    cluster_id: int | None = None
    image_locator: str | int | None = None


@dataclass(frozen=True)
class PairRecord:
    query_id: str
    positive_id: str
    cluster_id: int
    split: Literal["train", "val"]


@dataclass(frozen=True)
class ValidationCase:
    query_id: str
    positive_id: str
    cluster_id: int
    ignored_ids: frozenset[str]


def cid_to_relative_path(cid: str) -> Path:
    """Map official retrieval-SfM CID to its original archive path."""
    if len(cid) < 6:
        raise ValueError(f"CID must contain at least six characters, got {cid!r}")
    return Path(cid[-2:]) / cid[-4:-2] / cid[-6:-4] / cid


def canonicalize_train_pairs(pairs: Iterable[PairRecord]) -> tuple[PairRecord, ...]:
    """Drop self-pairs and duplicate undirected train pairs deterministically."""
    seen: set[tuple[str, str, int]] = set()
    result: list[PairRecord] = []
    for pair in pairs:
        if pair.query_id == pair.positive_id:
            continue
        first, second = sorted((pair.query_id, pair.positive_id))
        key = (first, second, pair.cluster_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            PairRecord(
                query_id=first,
                positive_id=second,
                cluster_id=pair.cluster_id,
                split=pair.split,
            )
        )
    return tuple(result)


class Sfm30kMetadata:
    """Canonical-ID view of the 30k selection and official 120k pair metadata."""

    def __init__(
        self,
        *,
        images: Mapping[str, ImageRecord],
        train_pairs: Sequence[PairRecord],
        val_pairs: Sequence[PairRecord],
    ) -> None:
        self.images = dict(images)
        self.train_pairs = tuple(train_pairs)
        self.val_pairs = tuple(val_pairs)
        self._records_by_split: dict[str, tuple[ImageRecord, ...]] = {
            split: tuple(
                record for record in self.images.values() if record.split == split
            )
            for split in ("train", "val")
        }

    @classmethod
    def from_official_files(
        cls,
        metadata_pickle: Path,
        names_clusters_mat: Path,
    ) -> "Sfm30kMetadata":
        """Join 120k pair metadata to the official 30k CID/cluster selection."""
        with metadata_pickle.open("rb") as handle:
            raw = pickle.load(handle)
        if not isinstance(raw, Mapping) or not {"train", "val"}.issubset(raw):
            raise ValueError("expected official metadata pickle with train and val mappings")
        selection = _load_30k_selection(names_clusters_mat)
        selected_ids = set(selection)
        images: dict[str, ImageRecord] = {}
        split_payloads: dict[str, Mapping[str, Any]] = {}
        raw_split_by_cid: dict[str, str] = {}

        for split in ("train", "val"):
            payload = raw[split]
            if not isinstance(payload, Mapping):
                raise ValueError(f"{split} payload is not a mapping")
            _validate_official_split_payload(payload, split)
            cids = [str(item) for item in payload["cids"]]
            clusters = [int(item) for item in payload["cluster"]]
            split_payloads[split] = payload
            for cid in cids:
                prior_split = raw_split_by_cid.setdefault(cid, split)
                if prior_split != split:
                    raise ValueError(f"CID {cid} appears in both official splits")

        # MATLAB 30k image data are stored per split in the same order as the
        # 30k selection file, not at their original 120k row positions.
        selected_locator: dict[str, int] = {}
        split_offsets: dict[str, int] = {"train": 0, "val": 0}
        for cid in selection:
            split = raw_split_by_cid.get(cid)
            if split not in split_offsets:
                raise ValueError(f"selected CID {cid} is missing from official splits")
            selected_locator[cid] = split_offsets[split]
            split_offsets[split] += 1

        for split in ("train", "val"):
            payload = split_payloads[split]
            cids = [str(item) for item in payload["cids"]]
            clusters = [int(item) for item in payload["cluster"]]
            for index, (cid, cluster) in enumerate(zip(cids, clusters, strict=True)):
                if cid not in selected_ids:
                    continue
                selected_cluster = selection[cid]
                if selected_cluster != cluster:
                    raise ValueError(
                        f"cluster mismatch for CID {cid}: 30k={selected_cluster}, "
                        f"official {split}={cluster}"
                    )
                if cid in images:
                    raise ValueError(f"CID {cid} appears in multiple selected splits")
                images[cid] = ImageRecord(
                    image_id=cid,
                    split=split,  # type: ignore[arg-type]
                    cluster_id=cluster,
                    image_locator=selected_locator[cid],
                )

        if set(images) != selected_ids:
            missing = sorted(selected_ids - set(images))
            raise ValueError(
                f"{len(missing)} selected 30k CIDs did not resolve in 120k metadata; "
                f"first examples: {missing[:5]}"
            )

        pair_by_split: dict[str, list[PairRecord]] = {"train": [], "val": []}
        for split, payload in split_payloads.items():
            cids = [str(item) for item in payload["cids"]]
            clusters = [int(item) for item in payload["cluster"]]
            for query_index, positive_index in zip(
                payload["qidxs"],
                payload["pidxs"],
                strict=True,
            ):
                qidx, pidx = int(query_index), int(positive_index)
                if not (0 <= qidx < len(cids) and 0 <= pidx < len(cids)):
                    raise ValueError(f"{split} pair index outside cids range")
                query_id, positive_id = cids[qidx], cids[pidx]
                if query_id not in selected_ids or positive_id not in selected_ids:
                    continue
                if query_id not in images or positive_id not in images:
                    raise ValueError("selected pair endpoint has no selected image record")
                if clusters[qidx] != clusters[pidx]:
                    raise ValueError(
                        f"{split} official pair crosses clusters: {query_id}, {positive_id}"
                    )
                pair_by_split[split].append(
                    PairRecord(
                        query_id=query_id,
                        positive_id=positive_id,
                        cluster_id=clusters[qidx],
                        split=split,  # type: ignore[arg-type]
                    )
                )

        metadata = cls(
            images=images,
            train_pairs=canonicalize_train_pairs(pair_by_split["train"]),
            val_pairs=tuple(pair_by_split["val"]),
        )
        metadata.validate()
        return metadata

    def image_ids(self, split: Literal["train", "val"] | None = None) -> tuple[str, ...]:
        if split is None:
            return tuple(self.images)
        return tuple(record.image_id for record in self._records_by_split[split])

    def records(self, split: Literal["train", "val"] | None = None) -> tuple[ImageRecord, ...]:
        if split is None:
            return tuple(self.images.values())
        return self._records_by_split[split]

    def cluster_for(self, image_id: str) -> int:
        record = self.images[image_id]
        if record.cluster_id is None:
            raise KeyError(f"image {image_id} has no cluster")
        return record.cluster_id

    def build_validation_cases(self) -> tuple[ValidationCase, ...]:
        """Use only the designated official positive and ignore other same-cluster views."""
        by_query: dict[str, PairRecord] = {}
        for pair in self.val_pairs:
            existing = by_query.get(pair.query_id)
            if existing is not None and existing.positive_id != pair.positive_id:
                raise ValueError(
                    f"query {pair.query_id} has multiple designated positives; "
                    "define a policy explicitly rather than silently choosing one"
                )
            by_query[pair.query_id] = pair
        val_records = self._records_by_split["val"]
        by_cluster: dict[int, set[str]] = defaultdict(set)
        for record in val_records:
            assert record.cluster_id is not None
            by_cluster[record.cluster_id].add(record.image_id)
        cases = []
        for pair in by_query.values():
            ignored = frozenset(
                image_id
                for image_id in by_cluster[pair.cluster_id]
                if image_id != pair.positive_id
            )
            cases.append(
                ValidationCase(
                    query_id=pair.query_id,
                    positive_id=pair.positive_id,
                    cluster_id=pair.cluster_id,
                    ignored_ids=ignored,
                )
            )
        return tuple(cases)

    def validate(self) -> dict[str, int]:
        if len(self.images) != len(set(self.images)):
            raise ValueError("duplicate image IDs")
        train_ids = set(self.image_ids("train"))
        val_ids = set(self.image_ids("val"))
        if train_ids & val_ids:
            raise ValueError("train and val image IDs overlap")
        train_clusters = {self.cluster_for(image_id) for image_id in train_ids}
        val_clusters = {self.cluster_for(image_id) for image_id in val_ids}
        if train_clusters & val_clusters:
            raise ValueError("train and val clusters overlap")
        for split, pairs, ids in (
            ("train", self.train_pairs, train_ids),
            ("val", self.val_pairs, val_ids),
        ):
            for pair in pairs:
                if pair.split != split:
                    raise ValueError(f"{split} list contains a {pair.split} pair")
                if pair.query_id not in ids or pair.positive_id not in ids:
                    raise ValueError(f"{split} pair endpoint is outside split")
                if pair.query_id == pair.positive_id:
                    raise ValueError(f"{split} contains a self pair")
                if (
                    self.cluster_for(pair.query_id) != pair.cluster_id
                    or self.cluster_for(pair.positive_id) != pair.cluster_id
                ):
                    raise ValueError(f"{split} pair cluster mismatch")
        return {
            "images": len(self.images),
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "train_clusters": len(train_clusters),
            "val_clusters": len(val_clusters),
            "train_pairs": len(self.train_pairs),
            "val_pairs": len(self.val_pairs),
        }


def _validate_official_split_payload(payload: Mapping[str, Any], split: str) -> None:
    needed = {"cids", "qidxs", "pidxs", "cluster"}
    if not needed.issubset(payload):
        raise ValueError(f"{split} metadata is missing {sorted(needed - set(payload))}")
    if len(payload["cids"]) != len(payload["cluster"]):
        raise ValueError(f"{split} cids and cluster lengths differ")
    if len(payload["qidxs"]) != len(payload["pidxs"]):
        raise ValueError(f"{split} qidxs and pidxs lengths differ")


def _load_30k_selection(path: Path) -> dict[str, int]:
    try:
        from scipy.io import loadmat
    except ImportError as error:
        raise ImportError("scipy is required to load 30k MATLAB selection metadata") from error
    raw = loadmat(path)
    if "cids" not in raw or "cluster" not in raw:
        raise ValueError("30k selection MAT must contain cids and cluster")
    cids_raw = np.asarray(raw["cids"]).reshape(-1)
    clusters_raw = np.asarray(raw["cluster"]).reshape(-1)
    if len(cids_raw) != len(clusters_raw):
        raise ValueError("30k selection cids and cluster lengths differ")
    selection: dict[str, int] = {}
    for cid_raw, cluster_raw in zip(cids_raw, clusters_raw, strict=True):
        cid = _unwrap_matlab_string(cid_raw)
        if cid in selection:
            raise ValueError(f"duplicate CID in 30k selection: {cid}")
        selection[cid] = int(cluster_raw)
    return selection


def _unwrap_matlab_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.size == 1:
        return str(array.reshape(-1)[0])
    if array.dtype.kind in {"U", "S"}:
        return "".join(array.reshape(-1).tolist())
    raise ValueError(f"cannot decode MATLAB CID value with shape {array.shape}")


class SfmImageDirectoryReader:
    """Read selected SfM images from the official original-image directory layout."""

    def __init__(self, image_root: Path) -> None:
        self.image_root = Path(image_root)

    def path_for(self, image_id: str) -> Path:
        path = self.image_root / cid_to_relative_path(image_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing SfM image for CID {image_id}: {path}")
        return path

    def read(self, image_id: str) -> Image.Image:
        with self.path_for(image_id).open("rb") as handle:
            image = Image.open(handle)
            image.load()
        return image


class SfmMatImageReader:
    """Lazy reader for the large v7.3 MAT image database, using HDF5 references."""

    _CANDIDATE_PATHS = (
        "{split}/data",
        "db/{split}/data",
        "db_{split}/data",
    )

    def __init__(self, mat_path: Path) -> None:
        try:
            import h5py
        except ImportError as error:
            raise ImportError(
                "h5py is required for streaming retrieval-SfM-30k.mat"
            ) from error
        self._h5py = h5py
        self.mat_path = Path(mat_path)
        self._file = h5py.File(self.mat_path, "r")
        self._datasets = {
            split: self._find_split_dataset(split) for split in ("train", "val")
        }
        # The cell-reference arrays are tiny relative to image pixels. Read
        # them once so individual image reads never scan the full split.
        self._references = {
            split: np.asarray(dataset).reshape(-1)
            for split, dataset in self._datasets.items()
        }

    def _find_split_dataset(self, split: str) -> Any:
        for pattern in self._CANDIDATE_PATHS:
            path = pattern.format(split=split)
            if path in self._file:
                return self._file[path]
        raise KeyError(
            f"could not find {split} image-cell dataset in {self.mat_path}. "
            "Inspect the MAT layout and update SfmMatImageReader candidate paths."
        )

    def read(self, split: Literal["train", "val"], index: int) -> Image.Image:
        references = self._references[split]
        if index < 0 or index >= len(references):
            raise IndexError(f"{split} image index {index} out of range")
        reference = references[index]
        if not isinstance(reference, self._h5py.Reference):
            raise TypeError(
                "expected a MATLAB v7.3 cell reference. This file layout is not "
                "the supported streaming image database format."
            )
        array = np.asarray(self._file[reference])
        return _matlab_array_to_pil(array)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "SfmMatImageReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _matlab_array_to_pil(array: np.ndarray) -> Image.Image:
    """Convert common MATLAB v7.3 image array layouts to Pillow RGB/RGBA data."""
    if array.ndim == 2:
        array = array.T
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (2, 1, 0))
    elif array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"unsupported MATLAB image array shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)
