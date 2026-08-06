# Lightweight CBIR

Course project for content-based image retrieval with multi-level DINOv2
embeddings. The backbone is frozen DINOv2 ViT-S/14 with registers. Small heads
are trained on retrieval-SfM-30k and evaluated once on Revisited Oxford and
Paris.

## Setup

If you prefer using Conda, run the following bash commands from the repository root:

```bash
conda env create -f environment.yml
conda activate lightweight-cbir
conda env config vars set PYTHONPATH="$PWD/src"
conda deactivate
conda activate lightweight-cbir
jupyter lab                     # Your prefred Jupyter compatible IDE
```

Or if you prefer using pip + venv, run the following instead from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate               # Linux/macOS
# source .venv\Scripts\activate         # Windows (Command Prompt)
# source .venv\Scripts\Activate.ps1     # Windows (PowerShell)
pip install --upgrade pip
pip install -e .
pip install -e ".[dev]"
jupyter lab                     # Your prefred Jupyter compatible IDE
```

`environment.yml` creates the local CUDA environment; `pyproject.toml` lists
the Python dependencies. The `PYTHONPATH` setting makes `src/cbir` importable
from every notebook without installing packages inside a notebook.

The environment targets CUDA 12.1. If your machine needs another CUDA build,
adjust `pytorch-cuda` in `environment.yml` before creating the environment.
On WSL, first confirm that `nvidia-smi` works in Ubuntu.

## Local data layout

Put downloaded archives and SfM source files here. They are ignored by Git.

```text
data/
├── raw/
│   ├── sfm30k/
│   │   ├── retrieval-SfM-120k.pkl
│   │   ├── retrieval-SfM-30k-imagenames-clusterids.mat
│   │   └── retrieval-SfM-30k.mat
│   └── revisitop/
│       └── archives/
│           ├── oxbuild_images-v1.tgz
│           ├── paris_1-v1.tgz
│           └── paris_2-v1.tgz
├── models/
│   └── torch_hub/  # created by Notebook 01
└── cache/
    ├── sfm30k/
    └── revisitop/
```

The notebooks reuse local SfM source files and RevisitOP archives when present,
downloading only missing files. SfM extraction writes 1,000-image feature
shards while decoding at most 128 source images at once. Later notebooks use
the feature caches instead of decoding source images again.

## Notebook order

1. `01_sfm_feature_cache.ipynb` downloads DINOv2 ViT-S/14 with registers,
   runs the pooling-temperature pilot, and builds the all-layer SfM-30k cache.
2. `02_layer_set_selection.ipynb` compares layer sets and fusion methods at
   128-D on the SfM train/validation protocol.
3. `03_descriptor_dimension_selection.ipynb` compares 64-D, 128-D, 256-D,
   and 384-D descriptors for the selected layer set and method.
4. `04_revisitop_feature_cache.ipynb` prepares frozen RevisitOP features for
   whichever final fusion method was selected.
5. `05_final_training_and_evaluation.ipynb` retrains the selected head on
   merged SfM pairs for the selected epoch count, then evaluates ROxford5k and
   RParis6k.

The first notebook keeps the pooling-temperature pilot because it documents
the choice of \(\tau_p = 0.025\). It is a diagnostic, not a RevisitOP tuning
step.

`BACKBONE_BATCH_SIZE` controls frozen feature extraction. `TRAIN_BATCH_SIZE`
controls InfoNCE batches and is part of an experiment specification; changing
it means running that experiment again. Notebook 01 also exposes
`DECODED_IMAGE_CHUNK_SIZE` for host-RAM control; it does not change features.

## Outputs and reuse

```text
outputs/
├── selections/
│   ├── layer_set_selection.json
│   └── final_model_selection.json
├── 01_sfm_feature_cache/{figures,results.json}
├── 02_layer_set_selection/
│   ├── checkpoints/
│   ├── figures/
│   └── results.json
├── 03_descriptor_dimension_selection/{checkpoints,figures,results.json}
├── 04_revisitop_feature_cache/{figures,results.json}
└── 05_final_training_and_evaluation/{checkpoints,figures,results.json}
```

Notebook 02, 03, and 05 use `results.json` to reuse matching experiments.
Those records keep complete per-epoch histories and settings; checkpoints live
beside the results that produced them. Notebook 01 and 04 use `results.json`
as concise run reports. Their cache manifests control feature-cache reuse.

Feature-cache manifests record the extraction settings. Rebuild a cache
explicitly when the backbone, preprocessing, or pooling temperature changes.
Raw source files and archives are never deleted automatically.

## Experiment protocol

- DINOv2 remains frozen; only descriptor heads are trained.
- The all-layer SfM cache uses long-side 224 preprocessing and
  \(\tau_p = 0.025\).
- Notebook 02 compares final CLS, multi-level CLS concatenation, Uniform layer
  weighting, Static layer weighting, and Dynamic layer weighting. Dynamic
  weighting uses Layer-wise Entropy-Modulated Gating.
- SfM InfoNCE batches include at most one positive pair per reconstruction
  cluster.
- RevisitOP is held out: do not use it to select temperature, layers,
  dimension, method, seed, or epoch.

Historical runs remain recorded in `EXPERIMENT_LOG.md`. Local runs are
documented separately so the older 256-D layer-selection study is not confused
with the streamlined 128-D study.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests do not download data or model weights.
