"""Equation-level checks, including the failure cases found in the audit."""

import math
import unittest

import numpy as np
import torch
from scipy.linalg import expm
from scipy.spatial.distance import cdist
from torch_geometric.data import Batch

from bgt_esre.config import MEASURES, Config
from bgt_esre.data import Dataset
from bgt_esre.graph import economy_adjacency, normalize_log, prepare_graphs, support
from bgt_esre.measures import (
    betweenness,
    clustering,
    communicability,
    diffusion,
    routing,
)
from bgt_esre.model import BGTESRE, ESREAttention, TemporalEncoder, rotate

torch.set_num_threads(1)


def arrays():
    w = np.array([[0.0, 1.0, 0.2], [1.0, 0.0, 0.4], [0.2, 0.4, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 15.0, 0.0]])
    return w, coords


class MeasureTests(unittest.TestCase):
    def test_economy_adjacency_and_distance_ablation(self):
        w, coords = arrays()
        c = Config().graph
        a, actual = economy_adjacency(w, coords, c, full=True)
        np.testing.assert_allclose(actual, w * np.exp(-0.1 * cdist(coords, coords)))
        self.assertEqual(a.sum(), 6)
        c.distance_decay = "inverse_mean_distance"
        _, actual = economy_adjacency(w, coords, c, full=True)
        distance = cdist(coords, coords)
        np.testing.assert_allclose(
            actual, w * np.exp(-distance / distance[np.triu_indices(3, 1)].mean())
        )
        c.weighting = "connectivity"
        _, actual = economy_adjacency(w, coords, c, full=True)
        np.testing.assert_array_equal(actual, w)

    def test_threshold_uses_positive_pairs_and_exact_deterministic_ties(self):
        tied = np.ones((5, 5)) - np.eye(5)
        a = support(tied, 0.15)
        self.assertEqual(a.sum(), 4)  # ceil(.15 * 10) undirected edges
        np.testing.assert_array_equal(a, a.T)
        self.assertFalse(np.diag(a).any())
        negative = -tied
        self.assertFalse(support(negative, 0.15).any())

    def test_diffusion_is_directional_and_solves_hitting_time_equations(self):
        w, _ = arrays()
        eps = 1e-8
        values = diffusion(w, eps)
        self.assertFalse(np.allclose(values, values.T))
        p = w / w.sum(1, keepdims=True)
        for target in range(3):
            others = np.array([i for i in range(3) if i != target])
            times = np.linalg.solve(np.eye(2) - p[np.ix_(others, others)], np.ones(2))
            np.testing.assert_allclose(values[others, target], 1 / (times + eps))

    def test_diffusion_components_and_isolates(self):
        w = np.zeros((5, 5))
        w[0, 1] = w[1, 0] = 1
        w[2, 3] = w[3, 2] = 0.5
        actual = diffusion(w, 1e-8)
        np.testing.assert_allclose(actual[w > 0], 1 / (1 + 1e-8))
        self.assertTrue(np.all(actual[w == 0] == 0))

    def test_routing_respects_absent_edges_and_indirect_paths(self):
        w = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        actual = routing(w, 1e-8)
        self.assertAlmostEqual(actual[0, 2], 1 / 3, places=7)
        self.assertEqual(actual[0, 3], 0)
        self.assertTrue(np.all(np.diag(actual) == 0))

    def test_clustering_matches_appendix_equation(self):
        w, _ = arrays()
        expected = np.zeros_like(w)
        for i in range(3):
            for j in range(3):
                if i != j:
                    expected[i, j] = sum(w[i, k] * w[j, k] for k in range(3)) / (
                        min(w[i].sum(), w[j].sum()) + 1e-8
                    )
        np.testing.assert_allclose(clustering(w, 1e-8), expected)

    def test_communicability_is_raw_matrix_exponential(self):
        w, _ = arrays()
        expected = expm(w)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(communicability(w, 1e-8), expected)

    def test_clustering_zero_direct_weight_can_have_common_neighbors(self):
        # Profile support is complete even for zero-similarity pairs. The
        # Appendix E numerator does not multiply by the direct edge weight.
        w = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertAlmostEqual(clustering(w, 1e-8)[0, 2], 1 / (1 + 1e-8))

    def test_betweenness_counts_equal_shortest_paths(self):
        cycle = np.array(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
            ]
        )
        actual = betweenness(cycle, 1e-8)
        np.testing.assert_allclose(actual[cycle > 0], 4.0)
        self.assertTrue(np.all(actual[cycle == 0] == 0))

    def test_minmax_then_log_and_constant_policy(self):
        eps = 1e-8
        np.testing.assert_allclose(
            normalize_log(np.array([2.0, 4.0, 6.0]), eps),
            np.log(np.array([0.0, 0.5, 1.0]) + eps),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            normalize_log(np.ones(3), eps), np.log(eps), rtol=1e-6
        )
        self.assertEqual(normalize_log(np.array([]), eps).shape, (0,))
        with self.assertRaises(ValueError):
            normalize_log(np.array([np.nan]), eps)

    def test_directional_attribute_matches_receiver_sender(self):
        w, coords = arrays()
        config = Config()
        config.graph.density = 1
        config.graph.weighting = "connectivity"
        dataset = Dataset(w[None], coords, features=np.ones((1, 3, 16)))
        graph = prepare_graphs(dataset, config)[0]
        sender, receiver = graph.edge_index.numpy()
        expected = normalize_log(
            diffusion(w, config.graph.eps)[receiver, sender], config.graph.eps
        )
        np.testing.assert_allclose(graph.log_measures["ediff"].numpy(), expected)
        self.assertNotEqual(
            float(graph.log_measures["ediff"][0]), float(graph.log_measures["ediff"][2])
        )

    def test_profile_is_fully_connected_and_does_not_transform_similarities(self):
        w, coords = arrays()
        similarity = w + np.eye(3)
        config = Config()
        config.model.input_mode = "profile"
        data = prepare_graphs(Dataset(similarity[None], coords), config)[0].data(
            config.graph.measures
        )
        self.assertEqual(data.edge_index.shape[1], 6)
        np.testing.assert_allclose(data.x, similarity)
        with self.assertRaises(ValueError):
            prepare_graphs(Dataset((-w)[None], coords), config)

    def test_five_metrics_define_ten_pairs(self):
        from itertools import combinations

        w, coords = arrays()
        config = Config()
        config.model.input_mode = "profile"
        prepared = prepare_graphs(Dataset(w[None], coords), config, MEASURES)[0]
        pairs = list(combinations(MEASURES, 2))
        self.assertEqual(len(pairs), 10)
        for pair in pairs:
            self.assertEqual(tuple(prepared.data(pair).phi.shape), (6, 2))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)

    def test_rotation_identity_norm_and_equal_sign_cancellation(self):
        q, k = (
            torch.randn(5, 2, 8, dtype=torch.float64),
            torch.randn(5, 2, 8, dtype=torch.float64),
        )
        angle = torch.randn(5, 2, 4, dtype=torch.float64)
        actual = (rotate(q, angle) * rotate(k, -angle)).sum(-1)
        expected = (q * rotate(k, -2 * angle)).sum(-1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(rotate(q, angle).norm(dim=-1), q.norm(dim=-1))
        torch.testing.assert_close(
            (rotate(q, angle) * rotate(k, angle)).sum(-1), (q * k).sum(-1)
        )

    def test_sparse_attention_matches_explicit_receiver_loop(self):
        attention = ESREAttention(8, 2).double()
        attention.gamma.data.fill_(0.3)
        x = torch.randn(4, 8, dtype=torch.float64)
        edge = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 1, 1, 2]])
        phi = torch.randn(5, 2, dtype=torch.float64)
        actual, alpha = attention(x, edge, phi)
        q, k, v = [
            layer(x).reshape(4, 2, 4)
            for layer in (attention.q, attention.k, attention.v)
        ]
        expected = torch.zeros_like(q)
        for receiver in range(4):
            selected = torch.where(edge[1] == receiver)[0]
            if not len(selected):
                continue
            angles = torch.einsum("ec,hkc->ehk", phi[selected], attention.angle)
            logits = (
                rotate(q[receiver].expand(len(selected), -1, -1), angles)
                * rotate(k[edge[0, selected]], -angles)
            ).sum(-1) / 2
            weights = logits.softmax(0)
            correction = torch.einsum(
                "ec,hdc->ehd", phi[selected], attention.value_projection
            )
            expected[receiver] = (
                weights[..., None]
                * (v[edge[0, selected]] + math.tanh(0.3) * correction)
            ).sum(0)
            torch.testing.assert_close(
                alpha[selected].sum(0), torch.ones(2, dtype=torch.float64)
            )
        torch.testing.assert_close(actual, attention.output(expected.flatten(1)))

    def test_zero_gate_opens_without_initial_value_projection_gradient(self):
        attention = ESREAttention(8, 2)
        edge = torch.tensor([[0, 1, 2, 0], [1, 2, 1, 2]])
        x, phi = torch.randn(3, 8), torch.randn(4, 2)
        output, _ = attention(x, edge, phi)
        output.square().sum().backward()
        self.assertEqual(float(attention.gamma), 0)
        self.assertGreater(float(attention.gamma.grad.abs()), 0)
        self.assertEqual(float(attention.value_projection.grad.abs().sum()), 0)
        optimizer = torch.optim.SGD(attention.parameters(), lr=0.01)
        optimizer.step()
        optimizer.zero_grad()
        attention(x, edge, phi)[0].square().sum().backward()
        self.assertGreater(float(attention.value_projection.grad.abs().sum()), 0)

    def test_attention_equivariance_with_directional_edges(self):
        layer = ESREAttention(8, 2).eval()
        x, phi = torch.randn(4, 8), torch.randn(6, 2)
        edge = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 1, 0, 3, 0]])
        old_to_new = torch.tensor([2, 0, 3, 1])
        permuted = torch.empty_like(x)
        permuted[old_to_new] = x
        actual, _ = layer(permuted, old_to_new[edge], phi)
        expected, _ = layer(x, edge, phi)
        torch.testing.assert_close(actual[old_to_new], expected)

    def test_temporal_encoder_has_no_cross_region_mixing(self):
        encoder = TemporalEncoder(3, 4, (3, 3), 2, 0).eval()
        x = torch.randn(2, 3, 24)
        changed = x.clone()
        changed[:, 0] += torch.randn(2, 24) * 3
        torch.testing.assert_close(encoder(x)[:, 1:], encoder(changed)[:, 1:])

    def test_modes_ablations_and_unlabeled_forward(self):
        w, coords = arrays()
        for mode in ("bold", "profile"):
            for mechanism in ("rotary", "additive"):
                config = Config()
                config.model.input_mode, config.model.attention = mode, mechanism
                (
                    config.model.hidden_dim,
                    config.model.num_heads,
                    config.model.num_layers,
                ) = 8, 2, 1
                dataset = Dataset(np.stack([w, w]), coords, np.ones((2, 3, 16)))
                graphs = prepare_graphs(dataset, config)
                batch = Batch.from_data_list(
                    [g.data(config.graph.measures) for g in graphs]
                )
                model = BGTESRE(config, 3)
                logits = model(batch)["logits"]
                self.assertEqual(tuple(logits.shape), (2, 2))
                logits.square().sum().backward()

    def test_empty_support_and_isolated_nodes_remain_finite(self):
        layer = ESREAttention(8, 2)
        result, alpha = layer(
            torch.randn(3, 8), torch.empty(2, 0, dtype=torch.long), torch.empty(0, 2)
        )
        torch.testing.assert_close(result, torch.zeros_like(result))
        self.assertEqual(tuple(alpha.shape), (0, 2))


if __name__ == "__main__":
    unittest.main()
