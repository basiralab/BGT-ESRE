"""BGT-ESRE: independent encoders, edge rotations, gated values, mean readout."""

import math

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax

from .config import Config


class TemporalEncoder(nn.Module):
    """Every convolution uses groups=N; each ROI owns its temporal filters."""

    def __init__(self, regions, dim, kernels, downsample, dropout):
        super().__init__()
        self.regions, self.dim = regions, dim
        self.downsample = (
            nn.AvgPool1d(downsample, ceil_mode=True)
            if downsample > 1
            else nn.Identity()
        )

        def block(channels_in, kernel):
            return nn.Sequential(
                nn.Conv1d(
                    channels_in,
                    regions * dim,
                    kernel,
                    padding=kernel // 2,
                    groups=regions,
                    bias=False,
                ),
                nn.BatchNorm1d(regions * dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.projection = block(regions, kernels[0])
        self.residual = nn.ModuleList(
            [block(regions * dim, kernel) for kernel in kernels[1:]]
        )

    def forward(self, x):
        x = self.projection(self.downsample(x))
        for block in self.residual:
            x = torch.nn.functional.gelu(x + block(x))
        return x.mean(dim=-1).reshape(len(x), self.regions, self.dim)


def rotate(x, angle):
    """Independent 2x2 rotations of adjacent feature pairs, without matrices."""
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    a, b = pairs.unbind(-1)
    cosine, sine = angle.cos(), angle.sin()
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1).flatten(
        -2
    )


class ESREAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, mechanism="rotary"):
        super().__init__()
        if dim % heads or (dim // heads) % 2:
            raise ValueError("Each attention head must have an even dimension")
        if mechanism not in ("rotary", "additive"):
            raise ValueError("mechanism must be rotary or additive")
        self.heads, self.head_dim, self.mechanism = heads, dim // heads, mechanism
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        if mechanism == "rotary":
            self.angle = nn.Parameter(torch.randn(heads, self.head_dim // 2, 2) * 0.1)
        else:
            self.add_q = nn.Parameter(torch.randn(heads, self.head_dim, 2) * 0.1)
            self.add_k = nn.Parameter(torch.randn(heads, self.head_dim, 2) * 0.1)
        self.value_projection = nn.Parameter(torch.randn(heads, self.head_dim, 2) * 0.1)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, phi):
        sender, receiver = edge_index
        q = self.q(x).reshape(len(x), self.heads, self.head_dim)[receiver]
        k = self.k(x).reshape(len(x), self.heads, self.head_dim)[sender]
        v = self.v(x).reshape(len(x), self.heads, self.head_dim)[sender]
        if self.mechanism == "rotary":
            angle = torch.einsum("ec,hkc->ehk", phi, self.angle)
            q, k = rotate(q, angle), rotate(k, -angle)
        else:
            q = q + torch.einsum("ec,hdc->ehd", phi, self.add_q)
            k = k + torch.einsum("ec,hdc->ehd", phi, self.add_k)
        scores = (q * k).sum(-1) / math.sqrt(self.head_dim)
        alpha = softmax(scores, receiver, num_nodes=len(x))
        correction = torch.einsum("ec,hdc->ehd", phi, self.value_projection)
        messages = self.dropout(alpha).unsqueeze(-1) * (
            v + self.gamma.tanh() * correction
        )
        aggregated = x.new_zeros(len(x), self.heads, self.head_dim)
        aggregated.index_add_(0, receiver, messages)
        return self.output(aggregated.flatten(1)), alpha


class ESRELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.norm1, self.norm2 = nn.LayerNorm(c.hidden_dim), nn.LayerNorm(c.hidden_dim)
        self.attention = ESREAttention(
            c.hidden_dim, c.num_heads, c.dropout, c.attention
        )
        self.ffn = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim * c.ffn_multiplier),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim * c.ffn_multiplier, c.hidden_dim),
            nn.Dropout(c.dropout),
        )

    def forward(self, x, edge_index, phi):
        z, alpha = self.attention(self.norm1(x), edge_index, phi)
        x = x + z
        return x + self.ffn(self.norm2(x)), alpha


class BGTESRE(nn.Module):
    def __init__(self, config: Config, num_regions: int):
        super().__init__()
        config.validate()
        self.config, self.num_regions = config, num_regions
        c = config.model
        self.encoder = (
            TemporalEncoder(
                num_regions,
                c.hidden_dim,
                c.cnn_kernels,
                c.temporal_downsample,
                c.dropout,
            )
            if c.input_mode == "bold"
            else nn.Linear(num_regions, c.hidden_dim, bias=False)
        )
        self.layers = nn.ModuleList([ESRELayer(c) for _ in range(c.num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, 2),
        )

    def forward(self, data, return_attention=False):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(len(data.x), dtype=torch.long, device=data.x.device)
        graphs = int(batch.max()) + 1
        if not torch.all(torch.bincount(batch, minlength=graphs) == self.num_regions):
            raise ValueError(
                "Every input graph must use the model's atlas region count and ordering"
            )
        if self.config.model.input_mode == "bold":
            h = self.encoder(data.x.reshape(graphs, self.num_regions, -1)).reshape(
                -1, self.config.model.hidden_dim
            )
        else:
            h = self.encoder(data.x)
        for layer in self.layers:
            h, alpha = layer(h, data.edge_index, data.phi)
        pooled = global_mean_pool(h, batch)
        result = {"logits": self.classifier(pooled), "embedding": pooled}
        if return_attention:
            result["attention"] = alpha
        return result
