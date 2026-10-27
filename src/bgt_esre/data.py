"""Safe NumPy interchange format; no dataset-specific adapters or locations."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    connectivity: np.ndarray
    coords: np.ndarray
    features: np.ndarray | None = None
    labels: np.ndarray | None = None
    subject_ids: np.ndarray | None = None
    groups: np.ndarray | None = None

    def __post_init__(self):
        self.connectivity = np.asarray(self.connectivity, dtype=np.float64)
        c = self.connectivity
        if c.ndim != 3 or c.shape[0] < 1 or c.shape[1] != c.shape[2] or c.shape[1] < 2:
            raise ValueError(
                "connectivity must have shape (subjects, regions, regions), with at least two regions"
            )
        if not np.isfinite(c).all() or not np.allclose(
            c, c.transpose(0, 2, 1), atol=1e-6, rtol=1e-6
        ):
            raise ValueError("connectivity must be finite and symmetric")
        self.coords = np.asarray(self.coords, dtype=np.float64)
        if self.coords.shape != (c.shape[1], 3) or not np.isfinite(self.coords).all():
            raise ValueError(
                "coords must be a finite (regions, 3) array in millimetres"
            )
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float32)
            if (
                self.features.ndim != 3
                or self.features.shape[:2] != c.shape[:2]
                or self.features.shape[2] < 1
                or not np.isfinite(self.features).all()
            ):
                raise ValueError(
                    "features must be finite with shape (subjects, regions, timepoints)"
                )
        if self.labels is not None:
            labels = np.asarray(self.labels)
            if labels.shape != (len(c),) or not np.isin(labels, [0, 1]).all():
                raise ValueError(
                    "labels must have shape (subjects,) with binary values 0/1"
                )
            self.labels = labels.astype(np.int64)
        if self.subject_ids is None:
            self.subject_ids = np.array([f"subject_{i:06d}" for i in range(len(c))])
        self.subject_ids = np.asarray(self.subject_ids).astype(str)
        if self.subject_ids.shape != (len(c),) or len(set(self.subject_ids)) != len(c):
            raise ValueError(
                "subject_ids must contain one unique identifier per subject"
            )
        if self.groups is not None:
            self.groups = np.asarray(self.groups).astype(str)
            if self.groups.shape != (len(c),):
                raise ValueError("groups must have shape (subjects,)")

    def __len__(self):
        return self.connectivity.shape[0]

    def require_labels(self):
        if self.labels is None:
            raise ValueError("This operation requires labels; prediction does not")

    def fingerprint(self):
        """Bind saved splits to values and order, not just array positions."""
        digest = hashlib.sha256()
        for value in (
            self.connectivity,
            self.coords,
            self.features,
            self.labels,
            self.subject_ids,
            self.groups,
        ):
            if value is None:
                digest.update(b"none")
            else:
                array = np.ascontiguousarray(value)
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()


def load_dataset(path: str | Path) -> Dataset:
    with np.load(path, allow_pickle=False) as archive:
        if not {"connectivity", "coords"} <= set(archive.files):
            raise ValueError("Input NPZ must contain connectivity and coords")
        names = (
            "connectivity",
            "coords",
            "features",
            "labels",
            "subject_ids",
            "groups",
        )
        return Dataset(**{name: archive[name] for name in names if name in archive})


def make_synthetic(path: str | Path, subjects=80, regions=32, timepoints=128, seed=7):
    """One toy cohort: smooth temporal signals, spatial modules, and weak class effects."""
    if subjects < 12 or regions < 4 or timepoints < 16:
        raise ValueError(
            "Synthetic data require >=12 subjects, >=4 regions, >=16 timepoints"
        )
    rng = np.random.default_rng(seed)
    modules = np.arange(regions) % 4
    centers = np.array([[-35, -35, 0], [35, -35, 0], [-35, 35, 25], [35, 35, 25]])
    coords = centers[modules] + rng.normal(0, 12, (regions, 3))
    labels = np.arange(subjects) % 2
    rng.shuffle(labels)
    signals = np.empty((subjects, regions, timepoints), dtype=np.float32)
    for s, label in enumerate(labels):
        latent = rng.normal(size=(5, timepoints))
        for t in range(1, timepoints):
            latent[:, t] = 0.7 * latent[:, t - 1] + 0.7 * latent[:, t]
        coupling = 0.55 + 0.12 * label * (modules == 0)
        values = (
            latent[modules] * coupling[:, None]
            + 0.6 * latent[4]
            + rng.normal(0, 0.6, (regions, timepoints))
        )
        signals[s] = (values - values.mean(1, keepdims=True)) / values.std(
            1, keepdims=True
        )
    connectivity = np.stack([np.corrcoef(signal) for signal in signals])
    payload = dict(
        features=signals,
        connectivity=connectivity,
        coords=coords,
        labels=labels,
        subject_ids=np.array([f"synthetic_{i:04d}" for i in range(subjects)]),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    np.savez_compressed(path, **payload)
    return Dataset(**payload)
