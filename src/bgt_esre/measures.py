"""Appendix E metrics. Every matrix uses row i / column j indexing.

Diffusion entry (i,j) is the inverse MFPT from i to j. It is directional,
even when the weighted adjacency is symmetric. No metric is log-scaled here.
"""

import networkx as nx
import numpy as np
from scipy.linalg import expm, solve
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def diffusion(w, eps):
    result = np.zeros_like(w)
    count, labels = connected_components(csr_matrix(w > 0), directed=False)
    for component in range(count):
        indices = np.flatnonzero(labels == component)
        if len(indices) < 2:
            continue
        a = w[np.ix_(indices, indices)]
        strength = a.sum(axis=1)
        # These divisions are exact on positive-strength components: adding
        # epsilon here would make P substochastic and change the random walk.
        p = a / strength[:, None]
        pi = strength / strength.sum()
        z = solve(np.eye(len(a)) - p + pi[None, :], np.eye(len(a)))
        times = (np.diag(z)[None, :] - z) / pi[None, :]
        off_diagonal = ~np.eye(len(a), dtype=bool)
        if not np.isfinite(times).all() or np.any(times[off_diagonal] <= 0):
            raise ValueError(
                "Unstable diffusion MFPT; inspect the graph's weight dynamic range"
            )
        values = np.zeros_like(a)
        values[off_diagonal] = 1 / (times[off_diagonal] + eps)
        result[np.ix_(indices, indices)] = values
    return result


def routing(w, eps):
    rows, cols = np.nonzero(w > 0)
    lengths = csr_matrix((1 / (w[rows, cols] + eps), (rows, cols)), shape=w.shape)
    distances = shortest_path(lengths, directed=False, method="FW")
    result = np.zeros_like(w)
    mask = np.isfinite(distances) & (distances > 0)
    result[mask] = 1 / (distances[mask] + eps)
    return result


def communicability(w, eps):
    result = expm(w)
    np.fill_diagonal(result, 0)
    return result


def clustering(w, eps):
    strength = w.sum(axis=1)
    result = (w @ w.T) / (np.minimum(strength[:, None], strength[None, :]) + eps)
    np.fill_diagonal(result, 0)
    return result


def betweenness(w, eps):
    graph = nx.from_numpy_array(w)
    for i, j in graph.edges:
        graph[i][j]["length"] = 1 / (w[i, j] + eps)
    # NetworkX counts unordered pairs on undirected graphs. Appendix E's
    # sum over s != t counts ordered pairs, hence the factor two.
    values = nx.edge_betweenness_centrality(graph, normalized=False, weight="length")
    result = np.zeros_like(w)
    for (i, j), value in values.items():
        result[i, j] = result[j, i] = 2 * value
    return result


FUNCTIONS = {
    "ediff": diffusion,
    "erout": routing,
    "comm": communicability,
    "ecc": clustering,
    "ebc": betweenness,
}
