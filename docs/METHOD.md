# Method-to-code correspondence

| Manuscript operation | Implementation |
|---|---|
| Positive proportional FC graph / full MC support | `graph.support`, `graph.economy_adjacency` |
| Economy adjacency | `graph.economy_adjacency` |
| Appendix E's five metrics | `measures.py` |
| Per-subject min–max followed by log | `graph.normalize_log` |
| Receiver/sender edge attributes | `graph.prepare_graphs` |
| Grouped independent BOLD CNN / profile projection | `model.TemporalEncoder`, `model.BGTESRE.encoder` |
| Head-specific angle projection and opposite rotations | `model.ESREAttention`, `model.rotate` |
| Zero-initialized learned value correction | `model.ESREAttention.gamma`, `value_projection` |
| Pre-norm residual block, GELU FFN | `model.ESRELayer` |
| Mean pooling and two-layer classifier | `model.BGTESRE` |
| Standard cross entropy | `training.train_cv` |
| Validation selection, independent outer test | `splits.py`, `training.train_cv` |

## Fixed numerical conventions

Distances use shared atlas centroids in millimetres; the default decay is 0.1 per millimetre. The inverse-mean-distance ablation uses **all unique, non-diagonal atlas pairs**, rather than subject-dependent retained edges.

FC support retains `ceil(density * positive_pair_count)` unique undirected pairs and stores both directions, with no self-loops. Equal weights are resolved by atlas indices, deterministically. Strict index-independent equivariance of thresholding therefore assumes no ties at the cutoff. Full profile support includes off-diagonal zero-similarity pairs too; such pairs have zero economy weight but remain attention edges.

Each metric is computed in float64. Normalization uses its values on every supported directed edge of one subject. Constant-valued coordinates map to zero before the log; an empty graph has empty attributes. Only tiny negative roundoff in otherwise nonnegative measures is clipped. Non-finite values and materially negative metrics raise errors, rather than being silently replaced.

Diffusion uses the stated inverse mean first-passage time. For each connected component, rows of `P` sum exactly to one, and the stationary distribution is normalized within that component. The fundamental-matrix solve is performed per component. Isolated nodes and unreachable pairs have zero diffusion efficiency; self-passage entries are excluded. This component-wise extension is needed because the single stationary fundamental-matrix formula assumes a connected walk. No spanning tree or artificial self-loop changes graph support.

Routing treats absent economy weights as absent edges, not finite paths of length `1/eps`. Supported positive weights have length `1/(W_ij + eps)`; unreachable pairs have zero efficiency. Communicability is the matrix exponential of W, without a separate max normalization. Its diagonal is excluded. Clustering is exactly the Appendix E common-neighbor product divided by minimum strength plus epsilon. Betweenness counts **all** equal shortest paths and uses ordered source-target pairs (twice the undirected raw NetworkX value). Its global scale cancels during min–max normalization, but raw semantics remain explicit.

`edge_index[0]` is the sender and `edge_index[1]` is the receiver. The attention formula uses `phi[receiver, sender]`. In the diffusion matrix, entry `(i,j)` is inverse passage time from i to j; an attention edge `j -> i` consequently reads that matrix at `(i,j)`, following the manuscript's indexing literally. Reverse edges generally receive different diffusion coordinates. Neither attention nor its exported data is forcibly symmetrized.

The gamma parameter is one scalar **per layer**, shared across heads. At gamma=0 the loss gradient of the value projection is zero in the first backward pass, while gamma itself can receive a gradient and open the branch. This is expected; a nonzero initial gate is unnecessary and would change the stated method.

## Architecture and reproducibility

The attention block uses Q/K/V projections of `LN(h)` and a pre-norm GELU FFN. The additive ablation injects the edge coordinate into Q and K at every layer, retaining value augmentation. All configurations use mean pooling, a two-layer GELU/dropout classifier, and ordinary cross entropy.

The YAML exposes defaults not fixed by the core equations: CNN kernels (7,5,3), fourfold temporal average pooling, residual CNN blocks with per-channel BatchNorm, twofold FFN expansion, dropout 0.1, and classifier hidden width equal to model dimension. Report the resolved configuration with experimental results.

The graph-attention layers are permutation equivariant when directed attributes are relabeled consistently. Symmetry is not required: row-normalized transitions and MFPT-based diffusion are generally asymmetric for symmetric weighted graphs. Independent ROI-specific temporal filters and profile input columns are tied to atlas identity, so arbitrary permutation equivariance is not claimed for the complete encoder with unchanged parameters.

Evaluation uses independent outer test folds and inner validation partitions. Variation over folds differs from variation over repeated seeds. Learned upstream harmonization must be fitted on training data and use only covariates available at inference. Input similarities are accepted unchanged.

Synthetic checks verify the implementation; reproducing real-data results requires matching data preparation, resolved configurations, and evaluation conventions. Checkpoints from the earlier dataset-specific implementation are incompatible.

## Library usage

```python
from bgt_esre import BGTESRE, Config
from bgt_esre.data import load_dataset
from bgt_esre.graph import prepare_graphs
from torch_geometric.data import Batch

config = Config.load("configs/train.yaml")
dataset = load_dataset("my_data.npz")
graphs = prepare_graphs(dataset, config)
model = BGTESRE(config, num_regions=dataset.coords.shape[0]).eval()
batch = Batch.from_data_list([graph.data(config.graph.measures) for graph in graphs[:2]])
logits = model(batch)["logits"]  # forward() does not require labels
```
