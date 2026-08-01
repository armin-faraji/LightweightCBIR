#!/usr/bin/env python3
"""Download/validate retrieval-SfM metadata and one explicitly chosen image source."""

from __future__ import annotations

import argparse
from pathlib import Path

from cbir.config import load_project_config
from cbir.data.download import (
    SfmUrls,
    download_with_resume,
    extract_selected_sfm_images,
)
from cbir.data.sfm import Sfm30kMetadata
from cbir.utils import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/development.yaml"))
    parser.add_argument(
        "--image-source",
        choices=("none", "mat", "archive"),
        default="mat",
        help="Use compact 30k MAT (default) or explicitly download/extract 120k archive.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override parent directory implied by the SfM metadata path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    data_dir = args.data_dir or config.sfm.metadata_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    urls = SfmUrls()

    metadata_path = data_dir / config.sfm.metadata_path.name
    names_path = data_dir / (
        config.sfm.names_clusters_path.name
        if config.sfm.names_clusters_path is not None
        else "retrieval-SfM-30k-imagenames-clusterids.mat"
    )
    download_with_resume(urls.metadata_pickle, metadata_path)
    download_with_resume(urls.names_clusters_mat, names_path)
    metadata = Sfm30kMetadata.from_official_files(metadata_path, names_path)
    summary = metadata.validate()

    image_details: dict[str, str] = {"image_source": args.image_source}
    if args.image_source == "mat":
        image_mat_path = data_dir / (
            config.sfm.image_mat_path.name
            if config.sfm.image_mat_path is not None
            else "retrieval-SfM-30k.mat"
        )
        download_with_resume(urls.image_mat, image_mat_path)
        image_details["image_mat_path"] = str(image_mat_path)
    elif args.image_source == "archive":
        archive_path = data_dir / "ims.tar.gz"
        image_root = config.sfm.image_root or data_dir / "ims30k"
        download_with_resume(urls.original_images_archive, archive_path)
        extract_selected_sfm_images(archive_path, image_root, metadata.image_ids())
        image_details["image_root"] = str(image_root)
        image_details["archive_path"] = str(archive_path)

    report_path = data_dir / "sfm30k_preparation_report.json"
    atomic_write_json(
        report_path,
        {
            "metadata_path": str(metadata_path),
            "names_clusters_path": str(names_path),
            **image_details,
            "counts": summary,
        },
    )
    print(f"Prepared SfM metadata: {summary}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

