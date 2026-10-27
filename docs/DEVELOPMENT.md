# Development

Install development tools and the package in editable mode:

```bash
python -m pip install -e '.[dev]' -c requirements-tested.txt
python -m ruff check src tests
python -m ruff format --check src tests
python -m unittest discover -s tests -v
bgt-esre smoke --out runs/smoke
```

`pyproject.toml` is the dependency source of truth. `requirements-tested.txt` is an optional constraints file recording the validated Python 3.10 environment. For a regular pinned install, use `python -m pip install . -c requirements-tested.txt`.

Tests compare metrics with first-passage linear systems, shortest-path examples, and the matrix exponential. They verify rotations, explicit versus sparse attention, directional equivariance of attention, gate learning, independent ROI encoders, isolated nodes, and both input modes. Pipeline tests use the one synthetic generator to verify split isolation, grouping, input fingerprints, checkpoint metrics, unlabeled prediction, all ten pairs, and held-out attention.

GitHub Actions installs the package as a wheel with CPU PyTorch and runs lint, format checks, tests, and the smoke workflow on Python 3.10 and 3.11. Successful local checks do not imply a remote workflow has run.

Record the resolved configuration, environment, splits, predictions, and selected checkpoints with each experiment. Source-checkout runs record their Git revision and dirty state; installed wheels report these as unavailable. Seeds cover Python, NumPy, PyTorch, and loader order. Bitwise identity across hardware, BLAS versions, and CUDA scatter kernels is not guaranteed.

Graph preprocessing includes dense matrix operations and weighted shortest paths; large atlases or cohorts can be expensive. Single runs compute requested measures only. Pair sweeps compute all five once per subject and reuse them.
