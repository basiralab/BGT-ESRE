# BGT-ESRE

Reference implementation of the **Brain Graph Transformer with EcoSpace Rotary Encoding (BGT-ESRE)** for binary classification of functional and morphological brain graphs from the paper titled *Brain Economy-Aligned Graph Transformer*. Users supply their own connectivity matrices, atlas coordinates, and regional time series or similarity profiles through one dataset-independent format.

## Quick start

Use Python 3.10 or newer in a virtual environment. Install a PyTorch build suited to your hardware if needed, then run from this repository:

```bash
python -m pip install --upgrade pip
python -m pip install .
bgt-esre smoke --out runs/smoke
```

The CPU smoke workflow generates one small synthetic cohort, trains three folds, evaluates selected checkpoints, predicts without labels, and exports held-out attention. It downloads no data. Synthetic scores check software behavior and do not reproduce the paper's experimental results.

For a longer example:

```bash
bgt-esre synthetic --out synthetic.npz
bgt-esre train --data synthetic.npz --config configs/train.yaml --out runs/example
bgt-esre evaluate --data synthetic.npz --checkpoint runs/example/fold_0/best.pt --out runs/evaluation
bgt-esre predict --data synthetic.npz --checkpoint runs/example/fold_0/best.pt --out runs/prediction
```

Use a new or empty output directory for each command. `--device auto` selects CUDA when available and CPU otherwise; `--device cpu` is also supported. `python -m bgt_esre` is equivalent to `bgt-esre`.

## Your data

Create one `.npz` file with arrays in the same subject and atlas order:

| Array | Shape | Meaning |
|---|---|---|
| `connectivity` | `(S, N, N)` | Finite symmetric connectivity matrices |
| `coords` | `(N, 3)` | Shared atlas centroids in **millimetres** |
| `features` | `(S, N, T)` | BOLD time series; required only in `bold` mode |
| `labels` | `(S,)` | Integers 0/1; required for training and evaluation |
| `subject_ids` | `(S,)` | Recommended unique string identifiers |
| `groups` | `(S,)` | Optional group IDs kept together across all partitions |

`S`, `N`, and `T` are the numbers of subjects, regions, and time points. Arrays must be numeric or strings, not pickled objects. Class `1` is the positive class for AUC and sensitivity. Prediction does not require labels.

```python
import numpy as np

np.savez_compressed(
    "my_data.npz",
    connectivity=connectivity,  # (S, N, N)
    coords=coords_mm,           # (N, 3)
    features=bold,              # (S, N, T); omit in profile mode
    labels=labels,              # (S,); omit for prediction
    subject_ids=subject_ids,    # (S,); unique strings
)
```

```bash
bgt-esre train --data my_data.npz --config configs/train.yaml --out runs/my_experiment
```

Set `model.input_mode` in the YAML:

- **`bold`** uses the grouped temporal CNN. Signed FC is allowed; the graph retains the strongest `graph.density` fraction of positive pairs. Each subject/ROI time series is independently z-scored by default; disable `standardize_bold` to retain supplied features unchanged.
- **`profile`** linearly projects rows of the supplied nonnegative similarity matrix and uses full off-diagonal support. Omit `features`. Density is unused, and the supplied similarities are not transformed again.

Models accept any atlas size with at least two regions. A trained checkpoint requires the same atlas and ordering at inference. Time-series length can change between input files, but subjects within one file must share a length. Preprocess variable scan lengths before packaging them; unmasked padding changes the signal.

Neuroimaging preprocessing and harmonization belong upstream. Convert raw morphological dissimilarities to similarities before using profile mode. Any learned preprocessing must use training data only and covariates available at inference, without held-out target labels.

## Ecospace pairs and ablations

Choose any two distinct measures: `ediff`, `erout`, `comm`, `ecc`, `ebc`.

```bash
bgt-esre train --data my_data.npz --config configs/train.yaml --measures ebc comm --out runs/pair
bgt-esre train --data my_data.npz --config configs/train.yaml --all-pairs --out runs/all_pairs
```

The pair sweep computes the five metrics once per subject, uses identical saved partitions for all ten pairs, and reports every result. Selecting pairs or tuning hyperparameters must happen within development data, not using outer test scores.

For ablations, copy your full-model YAML and change the relevant setting:

| Ablation | Setting |
|---|---|
| Additive edge-specific Q/K encoding | `model.attention: additive` |
| Coordinates from raw connectivity | `graph.weighting: connectivity` |
| Inverse mean atlas-distance decay | `graph.distance_decay: inverse_mean_distance` |

Pass `--splits runs/my_experiment/splits.json` to reuse a full-model run's partitions. The model, encoders, classifier, and training implementation are shared across configurations.

## Evaluation and interpretation

By default, five stratified outer test folds each reserve 20% of the remaining subjects for validation. Validation AUC controls checkpoint selection, scheduling, and early stopping. Only the selected checkpoint is evaluated on the test fold. Groups, when supplied, never cross partitions. Saved splits are validated against the input's values and ordering; every partition must contain both classes.

Each run saves its resolved configuration, environment, exact splits, epoch history, selected checkpoints, held-out predictions, and ACC/AUC/sensitivity/specificity. `metrics.json` reports test-fold mean and sample standard deviation (`ddof=1`) separately from pooled out-of-fold metrics. Undefined external-data metrics are saved as `null`. `evaluate` defaults to saved test subjects; use `--split all` explicitly for an independent labeled dataset.

```bash
bgt-esre attention --data my_data.npz --run runs/my_experiment --out runs/attention
```

Attention export uses saved outer test assignments and selected checkpoints. Its CSV preserves `receiver <- sender` direction, last-layer head-averaged attention, class means conditional on edge presence, and prevalence. An ecospace plot is created when edges pass the default 50% prevalence filter.

See [method details](docs/METHOD.md) for equations, numerical conventions, and library usage, and [development instructions](docs/DEVELOPMENT.md) for tests, formatting, and the pinned dependency environment.
