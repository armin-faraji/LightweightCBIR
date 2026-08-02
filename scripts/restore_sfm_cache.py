#!/usr/bin/env python3
"""Restore/validate the deterministic full SfM feature cache for a fresh runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

from cbir.config import load_project_config
from cbir.workflow import restore_complete_sfm_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/colab.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    location = restore_complete_sfm_cache(load_project_config(args.config))
    print(location.local_dir)


if __name__ == "__main__":
    main()
