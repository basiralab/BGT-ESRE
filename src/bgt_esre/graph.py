"""Economy adjacency and min-max/log edge attributes, shared by all commands."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from .config import Config
from .data import Dataset
from .measures import FUNCTIONS


def support(connectivity, density, full=False):
    n = len(connectivity)
    if full:
        return ~np.eye(n, dtype=bool)
    rows, cols = np.triu_indices(n, 1)
    positive = np.flatnonzero(connectivity[rows, cols] > 0)
    adjacency = np.zeros((n, n), dtype=bool)
    if positive.size:
        # Exact ceil(density * number_of_positive_pairs); atlas indices break ties.
        order = np.lexsort(
            (
                cols[positive],
                rows[positive],
                -connectivity[rows[positive], cols[positive]],
            )
        )
        keep = positive[order[: math.ceil(density * positive.size)]]
        adjacency[rows[keep], cols[keep]] = True
        adjacency |= adjacency.T
    return adjacency


def economy_adjacency(connectivity, coords, config, full=False):
    connectivity = (connectivity + connectivity.T) / 2
    positive = np.maximum(connectivity, 0).copy()
    np.fill_diagonal(positive, 0)
    adjacency = support(positive, config.density, full)
    weighted = positive * adjacency
    distance = cdist(coords, coords)
    decay = config.distance_decay
    if decay == "inverse_mean_distance":
        mean_distance = float(distance[np.triu_indices(len(distance), 1)].mean())
        if mean_distance <= 0:
            raise ValueError(
                "inverse_mean_distance requires distinct atlas coordinates"
            )
        decay = 1 / mean_distance
    if config.weighting == "economy":
        weighted *= np.exp(-float(decay) * distance)
    return adjacency, weighted


def normalize_log(values, eps):
    if values.size == 0:
        return values.astype(np.float32)
    if not np.isfinite(values).all() or np.any(values < -1e-10):
        raise ValueError(
            "Topological measure contains invalid values; inspect connectivity and coordinates"
        )
    values = np.maximum(values, 0)
    minimum, maximum = values.min(), values.max()
    normalized = (
        (values - minimum) / (maximum - minimum)
        if maximum > minimum
        else np.zeros_like(values)
    )
    return np.log(np.clip(normalized, 0, 1) + eps).astype(np.float32)


@dataclass
class PreparedGraph:
    features: torch.Tensor
    edge_index: torch.Tensor
    log_measures: dict[str, torch.Tensor]
    label: int | None

    def data(self, pair):
        attributes = dict(
            x=self.features,
            edge_index=self.edge_index,
            phi=torch.stack([self.log_measures[name] for name in pair], dim=-1),
            num_nodes=len(self.features),
        )
        if self.label is not None:
            attributes["y"] = torch.tensor([self.label], dtype=torch.long)
        return Data(**attributes)


def prepare_graphs(dataset: Dataset, config: Config, measures=None):
    config.validate()
    full = config.model.input_mode == "profile"
    if full and np.any(dataset.connectivity < 0):
        raise ValueError(
            "profile mode requires nonnegative similarities, not signed correlations or dissimilarities"
        )
    if not full and dataset.features is None:
        raise ValueError(
            "bold mode requires features with shape (subjects, regions, timepoints)"
        )
    names = tuple(dict.fromkeys(measures or config.graph.measures))
    results = []
    for index, connectivity in enumerate(dataset.connectivity):
        adjacency, w = economy_adjacency(
            connectivity, dataset.coords, config.graph, full
        )
        sender, receiver = np.where(adjacency)
        attributes = {}
        for name in names:
            matrix = FUNCTIONS[name](w, config.graph.eps)
            # The paper's phi_ij has receiver i and sender j; PyG edge_index
            # stores sender first. This transpose matters for diffusion.
            values = matrix[receiver, sender]
            attributes[name] = torch.from_numpy(normalize_log(values, config.graph.eps))
        x = (
            connectivity.astype(np.float32).copy()
            if full
            else dataset.features[index].copy()
        )
        if not full and config.model.standardize_bold:
            mean = x.mean(axis=1, keepdims=True, dtype=np.float64)
            scale = x.std(axis=1, keepdims=True, dtype=np.float64)
            x = ((x - mean) / np.maximum(scale, config.graph.eps)).astype(np.float32)
        results.append(
            PreparedGraph(
                torch.from_numpy(x),
                torch.from_numpy(np.stack([sender, receiver])).long(),
                attributes,
                None if dataset.labels is None else int(dataset.labels[index]),
            )
        )
    return results
