"""GNN encoders on music structure graphs, plus the CNN mel-spectrogram baseline.

Task 2 (spec section 4.2). GraphSAGE update as written in the spec:

    h_i^(l+1) = sigma( W^(l) . CONCAT[ h_i^(l) , MEAN_{j in N(i)} h_j^(l) ] )

which is exactly PyG's ``SAGEConv``. ``GATConv`` is offered as the alternative
the spec also names. Readout is mean pooling:

    g = (1/|V|) * sum_{i in V} h_i^(L),    y^ = sigmoid(W g + b)

:class:`GNNEncoder` returns *both* the pooled graph vector ``g`` and the final
node states ``h^(L)``. Tasks 3 and 4 need ``g``; the graph-coherence analysis and
the case studies need ``h^(L)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_add_pool, global_max_pool, global_mean_pool

from .utils import Config, get_logger

log = get_logger("gnn_model")


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class GNNEncoder(nn.Module):
    """L-layer GraphSAGE / GAT encoder with a mean-pool readout."""

    def __init__(self, cfg: Config, in_dim: int):
        super().__init__()
        g = cfg.model["gnn"]
        self.conv_type = g.get("conv", "sage")
        self.num_layers = int(g["num_layers"])
        self.hidden = int(g["hidden_dim"])
        self.readout_mode = g.get("readout", "mean")
        self.dropout = float(g.get("dropout", 0.3))
        use_bn = bool(g.get("batch_norm", True))
        heads = int(g.get("gat_heads", 4))

        # Project raw audio statistics to hidden width before message passing.
        # Without this the first conv has to both normalise a 328-dim feature
        # vector and aggregate, which trains poorly on ~450 graphs.
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self.hidden), nn.LayerNorm(self.hidden), nn.ReLU()
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            if self.conv_type == "gat":
                # concat=False keeps width at hidden so residuals line up
                self.convs.append(GATConv(self.hidden, self.hidden, heads=heads, concat=False,
                                          dropout=self.dropout))
            else:
                self.convs.append(SAGEConv(self.hidden, self.hidden, aggr="mean"))
            self.norms.append(nn.BatchNorm1d(self.hidden) if use_bn else nn.Identity())

        self.out_dim = self.hidden * (3 if self.readout_mode == "concat" else 1)
        log.info(
            "GNNEncoder: %s x%d, hidden=%d, in_dim=%d, readout=%s -> g dim %d",
            self.conv_type, self.num_layers, self.hidden, in_dim, self.readout_mode, self.out_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (g, h) where g is (B, out_dim) and h is (num_nodes, hidden)."""
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in                     # residual: keeps 3+ layers from oversmoothing
        return self.readout(h, batch), h

    def readout(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout_mode == "max":
            return global_max_pool(h, batch)
        if self.readout_mode == "sum":
            return global_add_pool(h, batch)
        if self.readout_mode == "concat":
            # mean carries average texture, max carries the salient segment,
            # sum carries length -- ablated in the report
            return torch.cat([
                global_mean_pool(h, batch), global_max_pool(h, batch), global_add_pool(h, batch)
            ], dim=1)
        return global_mean_pool(h, batch)     # spec default


# --------------------------------------------------------------------------- #
# Task 2 classifier
# --------------------------------------------------------------------------- #
class GNNClassifier(nn.Module):
    """GNN encoder + linear head. ``multilabel=False`` gives genre logits for
    softmax CE; ``True`` gives per-tag logits for BCE."""

    def __init__(self, cfg: Config, in_dim: int, num_classes: int, multilabel: bool = False):
        super().__init__()
        self.encoder = GNNEncoder(cfg, in_dim)
        self.multilabel = multilabel
        d = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Dropout(float(cfg.dotted("model.gnn.dropout", 0.3))),
            nn.Linear(d, d // 2), nn.ReLU(),
            nn.Dropout(float(cfg.dotted("model.gnn.dropout", 0.3))),
            nn.Linear(d // 2, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        g, _ = self.encoder(data.x, data.edge_index, data.batch,
                            getattr(data, "edge_weight", None))
        return self.head(g)

    def embed(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """(g, h) without the head -- used for t-SNE and coherence analysis."""
        return self.encoder(data.x, data.edge_index, data.batch,
                            getattr(data, "edge_weight", None))


# --------------------------------------------------------------------------- #
# Baseline B2: CNN on mel-spectrogram (no graph, no text)
# --------------------------------------------------------------------------- #
class MelCNN(nn.Module):
    """4-block VGG-style CNN over a (1, n_mels, T) log-mel patch.

    This is the spec's B2 baseline and the direct comparison for Task 2: it sees
    the same audio, at higher time resolution, without any graph structure. Its
    parameter count is reported next to the GNN's so the comparison is not
    confounded by capacity.
    """

    def __init__(self, cfg: Config, num_classes: int, n_mels: int | None = None):
        super().__init__()
        c = cfg.model["cnn_baseline"]
        chans = list(c["channels"])
        k = int(c.get("kernel_size", 3))
        p = float(c.get("dropout", 0.3))
        n_mels = int(n_mels or cfg.audio["n_mels"])

        blocks: list[nn.Module] = []
        prev = 1
        for ch in chans:
            blocks += [
                nn.Conv2d(prev, ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(ch), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(p * 0.5),
            ]
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(p), nn.Linear(prev, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)                     # (B, 1, n_mels, T)
        return self.head(self.pool(self.features(x)))


# --------------------------------------------------------------------------- #
# Baseline B4: hand-crafted features -> MLP
# --------------------------------------------------------------------------- #
class FeatureMLP(nn.Module):
    """Spec baseline B4. Operates on the mean-pooled node features of a graph,
    i.e. exactly the GNN's input with all structure discarded. The gap between
    this and the GNN isolates the contribution of message passing itself --
    a cleaner attribution than GNN-vs-CNN, which also changes the input
    representation."""

    def __init__(self, cfg: Config, in_dim: int, num_classes: int, hidden: int = 256):
        super().__init__()
        p = float(cfg.dotted("model.gnn.dropout", 0.3))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, data) -> torch.Tensor:
        pooled = global_mean_pool(data.x, data.batch)
        return self.net(pooled)
