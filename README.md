# Lightweight multi-level CBIR

This repository implements the course project **Content-Based Image Retrieval
Using Multi-Level Deep Embeddings**.  It freezes DINOv2 ViT-S/14 with registers,
caches all-layer global/local aggregates, trains a small Reliability-Gated
Multi-Level Global-Local Fusion (RGMF) head on retrieval-SfM-30k pairs, and
evaluates a locked model on Revisited Oxford and Paris.

## Repository layout

- `src/cbir/`: importable implementation: backbone, pooling, cache, fusion,
  training, evaluation, artifact, and hosted-runtime helpers.
- `scripts/`: reproducible data preparation, extraction, cache restoration,
  training, and evaluation entry points.
- `configs/`: complete runtime profiles.  `colab.yaml` is the one canonical
  configuration used by all cloud stages.
- `notebooks/`: the notebook workflow.  Notebooks remain in this directory;
  they do not need to move to the repository root.
- `outputs/`: disposable local notebook artifacts, ignored by Git.

## Colab workflow

Each active notebook is deliberately self-contained because opening another
Colab notebook must be treated as a new runtime.  Open them in separate browser
tabs through URLs of this form after pushing the repository to GitHub:

```text
https://colab.research.google.com/github/armin-faraji/LightweightCBIR/blob/main/notebooks/02_data_and_feature_cache.ipynb
https://colab.research.google.com/github/armin-faraji/LightweightCBIR/blob/main/notebooks/03_hyperparameter_tuning.ipynb
https://colab.research.google.com/github/armin-faraji/LightweightCBIR/blob/main/notebooks/04_final_runs_and_revisitop.ipynb
```

The first cell of each notebook mounts Drive, clones a project revision to
`/content/lightweight-cbir`, installs a fresh local package copy, selects the repository as
the current working directory, and records the runtime environment. It defaults
to this repository's public HTTPS URL and `main`; override `CBIR_REPO_URL` and
`CBIR_REPO_REVISION` for a fork, a private clone URL, or a pinned commit. Do
**not** set `PYTHONPATH`; editable installation makes `cbir` importable. Do
**not** use `!cd`; it affects only one temporary shell. The notebooks use Python
`os.chdir()` and `subprocess.run(..., cwd=PROJECT_ROOT)` instead.

The intended order is:

1. `01_backbone_access_check.ipynb` is a historical DINO-access diagnostic;
   its conclusion is already recorded: use DINOv2 ViT-S/14 with registers.
2. `02_data_and_feature_cache.ipynb` runs smoke checks, stages SfM-30k,
   performs the pooling-temperature pilot, and builds/resumes the feature cache.
3. `03_hyperparameter_tuning.ipynb` restores the completed cache, runs
   SfM-only ablations, and writes a selected-checkpoint lock file.
4. `04_final_runs_and_revisitop.ipynb` restores that lock, stages RevisitOP,
   and performs the final held-out evaluation.

Each active cloud-stage notebook (02–04) writes report-ready artifacts under
`outputs/<notebook-number>/<run-id>/`, including figures, metrics, configuration,
and runtime provenance.  Its final **Publish outputs to Drive** cell validates
and publishes them to:

```text
MyDrive/lightweight-cbir/notebook_outputs/<notebook-number>/<run-id>/
```

Feature-cache shards are different from ordinary outputs: they are incrementally
published to Drive during extraction, so a Colab disconnection loses at most the
currently active shard. Notebook 03 also copies every newly improved checkpoint
to Drive before continuing training. Run a notebook's final publish cell only
after creating all artifacts for that run; it writes an immutable, validated
artifact bundle rather than overwriting an earlier publication.

### Persistent layout

```text
MyDrive/lightweight-cbir/
├── datasets/
│   ├── sfm30k/
│   └── revisitop/
├── checkpoints/
│   └── 03/
├── feature_caches/
├── notebook_outputs/
├── locked/
└── torch_hub/
```

Use `configs/colab.yaml` for all three stages.  Once the pooling pilot selects a
temperature, set it there *before* full extraction.  Keep the same backbone,
preprocessing, and pooling fields for cache extraction, training, and final
evaluation.  For the final reproducible run, replace `revision: main` with the
tested immutable DINOv2 Git revision.

## Kaggle workflow

`configs/kaggle.yaml` uses `/kaggle/working` for runtime files.  Attach a source
snapshot and datasets as Kaggle inputs, copy/stage them into `/kaggle/working`,
and publish each completed cache/output directory as a notebook output or Kaggle
Dataset version.  Kaggle inputs are read-only and Google Drive mirroring is a
Colab-specific feature.

## Local workflow

Create a virtual environment, install the package, and use `configs/development.yaml`
or a copied local profile:

```bash
python -m pip install -e ".[dev]"
python scripts/prepare_sfm30k.py --config configs/development.yaml --image-source mat
python scripts/extract_features.py --config configs/development.yaml --backbone-batch-size 8
```

After full cache extraction, training needs only the feature cache and the two
SfM metadata files; it does not need the 6 GB MAT image file.  The final RevisitOP
run separately requires its dataset images and ground truth.

## Command-line stages

```bash
python scripts/restore_sfm_cache.py --config configs/colab.yaml
python scripts/train_head.py --config configs/colab.yaml --cache-dir <validated-local-cache>
python scripts/prepare_revisitop.py \
  --output-root /content/cbir_data/revisitop \
  --source-root /content/drive/MyDrive/lightweight-cbir/datasets/revisitop \
  --mode auto \
  --publish-root /content/drive/MyDrive/lightweight-cbir/datasets/revisitop \
  --repair
python scripts/evaluate_revisitop.py --config configs/colab.yaml --checkpoint <locked.pt> --revisit-root <root> --dataset roxford5k
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite validates preprocessing, register-token exclusion, pooling, cache
integrity/resumption, artifact publishing, fusion math, SfM joins, cluster-safe
batching, and RevisitOP protocol semantics.  It does not download a dataset or
checkpoint.

## Guardrails

- The DINO backbone remains frozen; only the fusion head is trained.
- Do not mix cache settings: model revision, preprocessing, pooling temperature,
  and token convention changes require a new cache.
- Build InfoNCE batches with at most one pair from each SfM cluster.
- Do not call pair-only SfM validation “mAP.”
- Do not choose layers, temperature, seed, or checkpoint using RevisitOP labels.
