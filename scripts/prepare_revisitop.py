#!/usr/bin/env python3
"""Stage or download ROxford5k/RParis6k in the layout used by evaluation.

Examples:

    # Restore a trusted Google Drive copy to the fresh Colab runtime.
    python scripts/prepare_revisitop.py \
        --output-root /content/cbir_data/revisitop \
        --source-root /content/drive/MyDrive/lightweight-cbir/datasets/revisitop \
        --mode stage

    # Download locally on first use, then validate/publish it to Drive.  On later
    # fresh runtimes the same command stages the Drive copy back to /content.
    python scripts/prepare_revisitop.py \
        --output-root /content/cbir_data/revisitop \
        --source-root /content/drive/MyDrive/lightweight-cbir/datasets/revisitop \
        --mode auto \
        --publish-root /content/drive/MyDrive/lightweight-cbir/datasets/revisitop

The output contains ``roxford5k/`` and ``rparis6k/`` directories.  Dataset
directories are published only after their annotation and all required images pass
validation; an interrupted preparation can be rerun safely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cbir.data.revisitop_prepare import (
    REVISITOP_DATASETS,
    prepare_revisitop_datasets,
    publish_revisitop_datasets,
)
from cbir.utils import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory to receive roxford5k/ and rparis6k/.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=REVISITOP_DATASETS,
        default=list(REVISITOP_DATASETS),
        help="One or both benchmark datasets (default: both).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help=(
            "Existing trusted root with <root>/<dataset>/ or "
            "<root>/datasets/<dataset>/; useful for a Drive-to-runtime stage."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "stage", "download"),
        default="auto",
        help="Stage an existing copy, download official sources, or choose automatically.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tgz archives below <output-root>/.downloads after validation.",
    )
    parser.add_argument(
        "--publish-root",
        type=Path,
        default=None,
        help=(
            "Optional persistent parent (for example Google Drive) to receive a "
            "validated local-to-Drive copy after preparation."
        ),
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help=(
            "Replace corrupt module-owned downloaded .pkl/.tgz files and retry; "
            "never alters a user-supplied --source-root."
        ),
    )
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Skip JPEG decode validation and retain only structural file checks.",
    )
    parser.add_argument(
        "--allow-nonofficial-counts",
        action="store_true",
        help="Allow synthetic/nonofficial database/query counts (testing only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_revisitop_datasets(
        args.output_root,
        datasets=args.datasets,
        source_root=args.source_root,
        mode=args.mode,
        keep_archives=args.keep_archives,
        verify_images=not args.skip_image_verification,
        repair=args.repair,
        enforce_official_counts=not args.allow_nonofficial_counts,
    )
    if args.publish_root is not None:
        published = publish_revisitop_datasets(
            args.output_root,
            args.publish_root,
            datasets=args.datasets,
            verify_images=not args.skip_image_verification,
            enforce_official_counts=not args.allow_nonofficial_counts,
        )
        report["persistent_publish"] = {
            "root": str(args.publish_root),
            "datasets": published["datasets"],
        }
        atomic_write_json(args.output_root / "revisitop_preparation_report.json", report)
    for dataset, details in report["datasets"].items():
        print(
            f"{dataset}: {details['status']} "
            f"({details.get('database_image_count', '?')} database images, "
            f"{details.get('query_count', '?')} queries)"
        )
    print(f"Wrote preparation report: {args.output_root / 'revisitop_preparation_report.json'}")
    if args.publish_root is not None:
        print(f"Validated persistent copy: {args.publish_root}")


if __name__ == "__main__":
    main()
