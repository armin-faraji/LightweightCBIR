# Lightweight multi-level CBIR

This repository implements the course project **Content-Based Image Retrieval Using Multi-Level Deep Embeddings**.

The system freezes official DINOv2 ViT-S/14 with four registers, caches all-layer CLS/local aggregates, trains a small Reliability-Gated Multi-Level Global-Local Fusion (RGMF) head on retrieval-SfM-30k pairs, and evaluates a locked model on Revisited Oxford/Paris.

## Package layout

- <code>src/cbir/backbone.py</code>: frozen official DINOv2 adapter and intermediate-token checks.
- <code>src/cbir/features.py</code>: CLS-guided local pooling, entropy diagnostics, feature extraction, and pooling-temperature pilot.
- <code>src/cbir/cache.py</code>: fingerprinted sharded cache with atomic writes, validation, and optional Google Drive mirroring.
- <code>src/cbir/data/</code>: SfM metadata joins/readers, RevisitOP query crops, deterministic preprocessing, and safe download helpers.
- <code>src/cbir/fusion.py</code>: uniform/static/reliability-gated RGMF head.
- <code>src/cbir/training.py</code>: cluster-safe cached-pair batching and symmetric InfoNCE.
- <code>src/cbir/evaluation.py</code>: SfM development retrieval and official RevisitOP semantics.
- <code>notebooks/</code>: access check, cache/pilot, tuning, and final evaluation notebooks.

## Setup

In Colab or a local virtual environment:

    pip install -e .

Run the completed <code>notebooks/01_backbone_access_check.ipynb</code> first. It selected official Torch Hub <code>dinov2_vits14_reg</code>; DINOv3 was inaccessible due checkpoint gating.

## Reproducible run order

1. Edit <code>configs/extraction_sfm30k.yaml</code> to match local/Drive paths and pin the DINOv2 revision before any full cache.
2. Download/validate SfM data. The compact 30k MAT route is default; the 120k archive is explicit and much larger.

       python scripts/prepare_sfm30k.py --config configs/extraction_sfm30k.yaml --image-source mat

3. Use <code>notebooks/02_data_and_feature_cache.ipynb</code> to run square/rectangular token smoke tests and the 1–2k SfM-only pooling-temperature pilot. Record one locked <code>pooling.temperature</code>.
4. Test a small cache, then build/resume the full cache:

       python scripts/extract_features.py --config configs/extraction_sfm30k.yaml --limit 500
       python scripts/extract_features.py --config configs/extraction_sfm30k.yaml

5. Run the SfM-only baseline/ablation tuning notebook or train a configured head:

       python scripts/train_head.py \
         --config configs/tuning.yaml \
         --cache-dir /content/cbir_cache/REPLACE_WITH_COMPLETED_CACHE_FOLDER

6. Lock a configuration using SfM validation only. Then evaluate each locked final checkpoint on RevisitOP:

       python scripts/evaluate_revisitop.py \
         --config configs/final.yaml \
         --checkpoint outputs/training/REPLACE_WITH_LOCKED_RUN/best.pt \
         --revisit-root /content/revisitop/datasets \
         --dataset roxford5k

Run the same command for <code>rparis6k</code>. Queries are bbox-cropped; database images remain full. The evaluator reports Medium and Hard mAP/mP@10.

## Tests

    PYTHONPATH=src python -m unittest discover -s tests -v

The unit suite validates preprocessing, token shapes/register exclusion through a fake DINO model, cache round trips, RGMF math, SfM joins, cluster-safe batching, and RevisitOP ignore semantics. It does not download datasets or checkpoints.

## Important guardrails

- Do not mix cache settings: model revision, preprocessing, pooling temperature, or token convention changes require a new feature cache.
- Train only the fusion head. The DINO backbone remains frozen.
- Build InfoNCE batches with at most one pair from each SfM cluster.
- Do not describe pair-only SfM validation as mAP.
- Never choose a layer, temperature, seed, or checkpoint using RevisitOP labels.
