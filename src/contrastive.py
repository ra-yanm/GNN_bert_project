"""Task 4: contrastive GNN-BERT dual encoder for MusicCaps alignment (spec 4.4).

Spec formulation:

    L_NCE = -log[ exp(sim(g_i, t_i)/tau) / sum_j exp(sim(g_i, t_j)/tau) ]
    sim(u, v) = u^T v / (||u|| ||v||)

Documented departure: the spec's loss is one-directional (graph -> caption), but
the spec's own metric list asks for both ``Caption->Audio R@K`` and
``Audio->Caption R@K``. Optimising one direction while scoring both leaves the
text->graph direction untrained, so we default to the symmetric CLIP-style
average of the two cross-entropies. ``symmetric: false`` in config.yaml restores
the spec-literal single direction, and the report ablates the pair.

The temperature is learned as ``log(1/tau)`` and clamped, following CLIP -- a
fixed tau=0.07 on a 32-sample batch produced visibly worse retrieval in our runs
(reported in the Task 4 ablation).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bert_encoder import BertTextEncoder
from .gnn_model import GNNEncoder
from .utils import Config, get_logger

log = get_logger("contrastive")


# --------------------------------------------------------------------------- #
# Projection head
# --------------------------------------------------------------------------- #
class ProjectionHead(nn.Module):
    """Two-layer MLP into the shared embedding space, then L2-normalise.
    Normalisation is part of the module so every caller gets unit vectors and
    ``sim`` is always a plain dot product."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or max(out_dim, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# --------------------------------------------------------------------------- #
# Dual encoder
# --------------------------------------------------------------------------- #
class ContrastiveGNNBert(nn.Module):
    """Graph encoder + text encoder projected into one shared space."""

    def __init__(self, cfg: Config, node_dim: int, embed_dim: int | None = None):
        super().__init__()
        embed_dim = int(embed_dim or cfg.dotted("model.fusion.proj_dim", 256))
        self.graph_encoder = GNNEncoder(cfg, node_dim)
        self.text_encoder = BertTextEncoder(cfg)

        self.graph_proj = ProjectionHead(self.graph_encoder.out_dim, embed_dim)
        self.text_proj = ProjectionHead(self.text_encoder.hidden_size, embed_dim)

        tau = float(cfg.dotted("train.task4_contrastive.temperature", 0.07))
        learnable = bool(cfg.dotted("train.task4_contrastive.learnable_temperature", True))
        init_logit_scale = float(np.log(1.0 / tau))
        if learnable:
            self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_logit_scale))
        self.embed_dim = embed_dim
        log.info("ContrastiveGNNBert: embed_dim=%d tau0=%.3f learnable_tau=%s",
                 embed_dim, tau, learnable)

    # ------------------------------------------------------------------ #
    def encode_graph(self, graph_data) -> torch.Tensor:
        g, _ = self.graph_encoder(graph_data.x, graph_data.edge_index, graph_data.batch,
                                  getattr(graph_data, "edge_weight", None))
        return self.graph_proj(g)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, t = self.text_encoder(input_ids, attention_mask)
        return self.text_proj(t)

    def forward(self, batch) -> dict:
        g = self.encode_graph(batch["graph"])
        t = self.encode_text(batch["input_ids"], batch["attention_mask"])
        # clamp mirrors CLIP: an unbounded scale diverges within a few epochs
        scale = self.logit_scale.clamp(max=np.log(100.0)).exp()
        return {"g": g, "t": t, "logit_scale": scale, "sim": scale * g @ t.t()}

    def param_groups(self, lr: float, lr_bert: float, weight_decay: float) -> list[dict]:
        text_ids = {id(p) for p in self.text_encoder.parameters()}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in text_ids]
        return [
            {"params": [p for p in self.text_encoder.parameters() if p.requires_grad],
             "lr": lr_bert, "weight_decay": weight_decay},
            {"params": rest, "lr": lr, "weight_decay": weight_decay},
        ]


# --------------------------------------------------------------------------- #
# InfoNCE
# --------------------------------------------------------------------------- #
class InfoNCELoss(nn.Module):
    """InfoNCE over an in-batch similarity matrix.

    ``symmetric=True``  -> 0.5 * (CE(rows) + CE(cols))   [CLIP; default]
    ``symmetric=False`` -> CE(rows) only                 [spec-literal g->t]

    ``duplicate_safe`` masks out off-diagonal entries whose caption is identical
    to the anchor's. MusicCaps has a small number of repeated captions, and
    treating a true duplicate as a negative injects label noise directly into the
    denominator.
    """

    def __init__(self, symmetric: bool = True, duplicate_safe: bool = True):
        super().__init__()
        self.symmetric = symmetric
        self.duplicate_safe = duplicate_safe

    def forward(self, sim: torch.Tensor, group_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        n = sim.shape[0]
        target = torch.arange(n, device=sim.device)

        if self.duplicate_safe and group_ids is not None:
            same = group_ids[:, None] == group_ids[None, :]
            same.fill_diagonal_(False)
            sim = sim.masked_fill(same, float("-inf"))

        loss_g2t = F.cross_entropy(sim, target)
        if not self.symmetric:
            return loss_g2t, {"loss_g2t": float(loss_g2t.detach()), "loss_t2g": float("nan")}
        loss_t2g = F.cross_entropy(sim.t(), target)
        loss = 0.5 * (loss_g2t + loss_t2g)
        return loss, {
            "loss_g2t": float(loss_g2t.detach()),
            "loss_t2g": float(loss_t2g.detach()),
            "loss_total": float(loss.detach()),
        }


# --------------------------------------------------------------------------- #
# Retrieval + zero-shot tagging
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_split(model: ContrastiveGNNBert, loader, device: torch.device) -> dict:
    """Embed a whole split once, for the full-split retrieval matrix.

    Retrieval must be scored against *all* test candidates, not in-batch, or R@K
    is inflated by the small candidate pool.
    """
    model.eval()
    gs, ts, ids, caps = [], [], [], []
    for batch in loader:
        graph = batch["graph"].to(device)
        gs.append(model.encode_graph(graph).cpu())
        ts.append(model.encode_text(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        ).cpu())
        ids.extend(batch.get("ids", []))
        caps.extend(batch.get("captions", []))
    return {
        "g": torch.cat(gs), "t": torch.cat(ts),
        "ids": ids, "captions": caps,
    }


def similarity_matrix(g: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    """Cosine similarity; both inputs already unit-norm from ProjectionHead."""
    return (g @ t.t()).cpu().numpy()


@torch.no_grad()
def zero_shot_tags(
    model: ContrastiveGNNBert,
    graph_emb: torch.Tensor,
    tag_names: list[str],
    tokenizer,
    device: torch.device,
    prompt: str = "a music clip that is {}",
) -> np.ndarray:
    """Zero-shot tag scores by embedding each tag as a text prompt and scoring
    cosine similarity against every graph embedding.

    This is the spec's "zero-shot tag prediction from captions vs. Task 3
    supervised model" -- the contrastive model never saw the tag vocabulary as
    supervision, so it tests whether the shared space generalises.
    """
    model.eval()
    prompts = [prompt.format(t) for t in tag_names]
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    temb = model.encode_text(enc["input_ids"].to(device), enc["attention_mask"].to(device)).cpu()
    return (graph_emb @ temb.t()).numpy()          # (N_clips, K_tags)


def top_retrievals(
    sim: np.ndarray, k: int = 3, direction: str = "t2g"
) -> list[list[tuple[int, float]]]:
    """Top-k indices + scores per query. ``t2g`` = caption queries audio, which
    is the direction the spec's qualitative table asks for."""
    mat = sim.T if direction == "t2g" else sim
    out = []
    for row in mat:
        idx = np.argsort(-row)[:k]
        out.append([(int(i), float(row[i])) for i in idx])
    return out
