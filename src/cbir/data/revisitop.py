"""Revisited Oxford/Paris metadata, query-box crops, and image paths."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image


@dataclass(frozen=True)
class RevisitQuery:
    query_id: str
    source_image_id: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class RevisitGroundTruth:
    query_id: str
    easy: frozenset[str]
    hard: frozenset[str]
    junk: frozenset[str]


class RevisitOPDataset:
    """One pinned ROxford/RParis release with full database and cropped queries."""

    def __init__(
        self,
        *,
        name: str,
        image_root: Path,
        database_ids: Sequence[str],
        queries: Sequence[RevisitQuery],
        ground_truth: Mapping[str, RevisitGroundTruth],
    ) -> None:
        self.name = name
        self.image_root = Path(image_root)
        self.database_ids = tuple(database_ids)
        self.queries = tuple(queries)
        self.ground_truth = dict(ground_truth)
        self.validate()

    @classmethod
    def from_ground_truth_pickle(
        cls,
        *,
        name: str,
        ground_truth_path: Path,
        image_root: Path,
    ) -> "RevisitOPDataset":
        with ground_truth_path.open("rb") as handle:
            raw = pickle.load(handle)
        needed = {"imlist", "qimlist", "gnd"}
        if not isinstance(raw, Mapping) or not needed.issubset(raw):
            raise ValueError("expected RevisitOP pickle with imlist, qimlist, and gnd")
        database_ids = tuple(str(item) for item in raw["imlist"])
        query_ids = tuple(str(item) for item in raw["qimlist"])
        entries = raw["gnd"]
        if len(query_ids) != len(entries):
            raise ValueError("Revisit query names and ground-truth entries differ in length")
        queries: list[RevisitQuery] = []
        ground_truth: dict[str, RevisitGroundTruth] = {}
        for query_id, entry in zip(query_ids, entries, strict=True):
            if "bbx" not in entry:
                raise ValueError(f"query {query_id} has no bounding box")
            bbox = tuple(float(value) for value in entry["bbx"])
            if len(bbox) != 4:
                raise ValueError(f"query {query_id} bounding box must have four values")
            queries.append(
                RevisitQuery(
                    query_id=query_id,
                    source_image_id=query_id,
                    bbox_xyxy=bbox,  # type: ignore[arg-type]
                )
            )
            ground_truth[query_id] = RevisitGroundTruth(
                query_id=query_id,
                easy=frozenset(str(value) for value in entry.get("easy", [])),
                hard=frozenset(str(value) for value in entry.get("hard", [])),
                junk=frozenset(str(value) for value in entry.get("junk", [])),
            )
        return cls(
            name=name,
            image_root=image_root,
            database_ids=database_ids,
            queries=queries,
            ground_truth=ground_truth,
        )

    def image_path(self, image_id: str) -> Path:
        candidates = (
            self.image_root / f"{image_id}.jpg",
            self.image_root / image_id,
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"could not find RevisitOP image {image_id} below {self.image_root}"
        )

    def read_database_image(self, image_id: str) -> Image.Image:
        return _load_pil(self.image_path(image_id))

    def read_query_image(self, query: RevisitQuery) -> Image.Image:
        return _load_pil(self.image_path(query.source_image_id))

    def crop_query(self, query: RevisitQuery) -> Image.Image:
        return crop_revisit_query(self.read_query_image(query), query.bbox_xyxy)

    def validate(self) -> None:
        if len(self.database_ids) != len(set(self.database_ids)):
            raise ValueError("Revisit database IDs are not unique")
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Revisit query IDs are not unique")
        if set(query_ids) != set(self.ground_truth):
            raise ValueError("query and ground-truth ID sets differ")
        database = set(self.database_ids)
        for query in self.queries:
            if len(query.bbox_xyxy) != 4:
                raise ValueError(f"invalid bbox for {query.query_id}")
            ground_truth = self.ground_truth[query.query_id]
            known = ground_truth.easy | ground_truth.hard | ground_truth.junk
            if not known.issubset(database):
                missing = sorted(known - database)
                raise ValueError(
                    f"ground truth for {query.query_id} refers to unknown images: "
                    f"{missing[:5]}"
                )


def crop_revisit_query(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
) -> Image.Image:
    """Crop official [left, top, right, bottom] query region before preprocessing."""
    left, top, right, bottom = bbox_xyxy
    if not (right > left and bottom > top):
        raise ValueError(f"invalid RevisitOP bbox {bbox_xyxy}")
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        raise ValueError(
            f"bbox {bbox_xyxy} lies outside image size {(image.width, image.height)}"
        )
    return image.crop((left, top, right, bottom))


def _load_pil(path: Path) -> Image.Image:
    with path.open("rb") as handle:
        image = Image.open(handle)
        image.load()
    return image
