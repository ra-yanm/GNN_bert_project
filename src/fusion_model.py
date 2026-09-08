"""GNN-BERT fusion for multi-context understanding (spec section 4.3).

Spec formulation:

    A = softmax( Q K^T / sqrt(d) ),   Q = g W_Q,  K = H_text W_K
    z = CONCAT( g , A H_text ),        y^ = sigmoid(W z)
    L = L_tags + alpha*||v - v^||^2 + beta*||a - a^||^2

Two deliberate, documented departures from the literal formulas:

1. **A value projection is added.** The spec writes ``A H_text`` -- attention
   weights applied to the raw token states. We use ``A (H_text W_V)``, i.e.
   standard scaled dot-product attention. Without W_V the fused vector is forced
   to live in BERT's un-adapted output space, and the fusion layer has no way to
   reshape what it reads out.

2. **Node-level queries are available.** With ``Q = g W_Q``, ``g`` is a *single*
   vector, so ``A`` is 1xL: the "cross-attention" degenerates into one learned
   weighted average over text tokens. ``query_mode="nodes"`` instead uses the
   final node states H_graph as queries, giving a genuine |V|xL attention map --
   which is also what makes the Task 3 case studies interpretable, since each
   segment gets its own distribution over caption tokens. Both modes are
   implemented and ablated; ``query_mode="graph"`` reproduces the spec exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from .bert_encoder import BertTextEncoder
from .gnn_model import GNNEncoder
from .utils import Config, get_logger

log = get_logger("fusion_model")

FUSION_MODES = ("bert_only", "gnn_only", "concat", "gated", "cross_attention")


# --------------------------------------------------------------------------- #
# Cross-attention block
# --------------------------------------------------------------------------- #
class CrossAttentionFusion(nn.Module):
    """Multi-head attention from graph queries onto text keys/values."""

    def __init__(self, graph_dim: int, text_dim: int, proj_dim: int,
                 heads: int = 4, dropout: float = 0.2, query_mode: str = "graph"):
        super().__init__()
        assert proj_dim % heads == 0, "proj_dim must divide evenly by attn_heads"
        self.h = heads
        self.dk = proj_dim // heads
        self.proj_dim = proj_dim
        self.query_mode = query_mode

        self.W_Q = nn.Linear(graph_dim, proj_dim)
        self.W_K = nn.Linear(text_dim, proj_dim)
        self.W_V = nn.Linear(text_dim, proj_dim)      # absent from the spec; see module docstring
        self.out = nn.Linear(proj_dim, proj_dim)
        self.norm_q = nn.LayerNorm(graph_dim)
        self.norm_kv = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None    # cached for case studies

    def forward(
        self,
        q_src: torch.Tensor,          # (B, Nq, graph_dim)
        H_text: torch.Tensor,         # (B, L, text_dim)
        text_mask: torch.Tensor,      # (B, L) 1 = real token
        q_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Nq, _ = q_src.shape
        L = H_text.shape[1]

        Q = self.W_Q(self.norm_q(q_src)).view(B, Nq, self.h, self.dk).transpose(1, 2)
        Hn = self.norm_kv(H_text)
        K = self.W_K(Hn).view(B, L, self.h, self.dk).transpose(1, 2)
        V = self.W_V(Hn).view(B, L, self.h, self.dk).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.dk ** 0.5)          # (B,h,Nq,L)
        scores = scores.masked_fill(~text_mask[:, None, None, :].bool(), float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        self.last_attn = attn.detach()

        ctx = (attn @ V).transpose(1, 2).reshape(B, Nq, self.proj_dim)  # (B,Nq,proj)
        ctx = self.out(ctx)

        # Pool queries down to one vector per graph, ignoring padded nodes
        if q_mask is not None:
            m = q_mask.unsqueeze(-1).float()
            return (ctx * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return ctx.mean(dim=1)


# --------------------------------------------------------------------------- #
# Full Task 3 model
# --------------------------------------------------------------------------- #
class GNNBertFusion(nn.Module):
    """Fuses a graph encoder and a text encoder into tag logits plus optional
    valence/arousal regression heads.

    ``mode`` selects the ablation:
      ``bert_only``       text branch only (equivalent to Task 1's model)
      ``gnn_only``        graph branch only (equivalent to Task 2's model)
      ``concat``          z = [g ; t]                (early fusion)
      ``gated``           z = [g ; t ; gate*g + (1-gate)*t_proj]
      ``cross_attention`` z = [g ; CrossAttn(g, H_text)]   (spec's recommendation)
    """

    def __init__(
        self,
        cfg: Config,
        node_dim: int,
        num_tags: int,
        mode: str | None = None,
        with_emotion: bool = True,
    ):
        super().__init__()
        f = cfg.model["fusion"]
        self.mode = mode or f.get("mode", "cross_attention")
        if self.mode not in FUSION_MODES:
            raise ValueError(f"mode must be one of {FUSION_MODES}, got {self.mode!r}")
        self.with_emotion = with_emotion
        self.proj_dim = int(f.get("proj_dim", 256))
        drop = float(f.get("dropout", 0.2))
        self.query_mode = f.get("query_mode", "nodes")

        self.use_text = self.mode != "gnn_only"
        self.use_graph = self.mode != "bert_only"

        text_dim = 0
        if self.use_text:
            self.text_encoder = BertTextEncoder(cfg)
            text_dim = self.text_encoder.hidden_size
            self.text_proj = nn.Linear(text_dim, self.proj_dim)

        graph_dim = 0
        if self.use_graph:
            self.graph_encoder = GNNEncoder(cfg, node_dim)
            graph_dim = self.graph_encoder.out_dim
            self.graph_proj = nn.Linear(graph_dim, self.proj_dim)

        # --- decide z's width per mode --------------------------------------
        if self.mode == "cross_attention":
            self.cross = CrossAttentionFusion(
                graph_dim=self.graph_encoder.hidden if self.query_mode == "nodes" else graph_dim,
                text_dim=text_dim,
                proj_dim=self.proj_dim,
                heads=int(f.get("attn_heads", 4)),
                dropout=drop,
                query_mode=self.query_mode,
            )
            z_dim = self.proj_dim * 2                       # [g_proj ; attended_text]
        elif self.mode == "concat":
            z_dim = self.proj_dim * 2
        elif self.mode == "gated":
            self.gate = nn.Sequential(nn.Linear(self.proj_dim * 2, self.proj_dim), nn.Sigmoid())
            z_dim = self.proj_dim * 3
        else:                                              # single-branch ablations
            z_dim = self.proj_dim

        self.z_dim = z_dim
        self.norm_z = nn.LayerNorm(z_dim)
        self.dropout = nn.Dropout(drop)
        self.tag_head = nn.Sequential(
            nn.Linear(z_dim, z_dim // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(z_dim // 2, num_tags),
        )
        if with_emotion:
            # separate small heads so the two targets do not share a bottleneck
            self.valence_head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            self.arousal_head = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        log.info("GNNBertFusion mode=%s query_mode=%s z_dim=%d tags=%d emotion=%s",
                 self.mode, self.query_mode, z_dim, num_tags, with_emotion)

    # ------------------------------------------------------------------ #
    def forward(self, batch) -> dict:
        g_proj = t_proj = None
        H_text = text_mask = None
        h_nodes = None

        if self.use_text:
            H_text, t = self.text_encoder(batch["input_ids"], batch["attention_mask"])
            text_mask = batch["attention_mask"]
            t_proj = self.text_proj(t)

        if self.use_graph:
            gd = batch["graph"]
            g, h_nodes = self.graph_encoder(gd.x, gd.edge_index, gd.batch,
                                            getattr(gd, "edge_weight", None))
            g_proj = self.graph_proj(g)

        # --- build z ---------------------------------------------------
        if self.mode == "bert_only":
            z = t_proj
        elif self.mode == "gnn_only":
            z = g_proj
        elif self.mode == "concat":
            z = torch.cat([g_proj, t_proj], dim=1)
        elif self.mode == "gated":
            cat = torch.cat([g_proj, t_proj], dim=1)
            gate = self.gate(cat)
            z = torch.cat([g_proj, t_proj, gate * g_proj + (1 - gate) * t_proj], dim=1)
        else:                                      # cross_attention
            if self.query_mode == "nodes":
                # (B, Nmax, hidden) + validity mask, so padded nodes are ignored
                dense_h, node_mask = to_dense_batch(h_nodes, batch["graph"].batch)
                attended = self.cross(dense_h, H_text, text_mask, q_mask=node_mask)
            else:                                  # spec-literal: single graph query
                attended = self.cross(g_proj.unsqueeze(1), H_text, text_mask)
            z = torch.cat([g_proj, attended], dim=1)

        z = self.dropout(self.norm_z(z))
        out = {"tag_logits": self.tag_head(z), "z": z}
        if self.with_emotion:
            out["valence"] = self.valence_head(z).squeeze(-1)
            out["arousal"] = self.arousal_head(z).squeeze(-1)
        if self.mode == "cross_attention":
            out["attn"] = self.cross.last_attn
        return out

    # ------------------------------------------------------------------ #
    def param_groups(self, lr_bert: float, lr_head: float, weight_decay: float) -> list[dict]:
        """BERT gets the small LR; graph encoder, projections and heads get the
        large one (they are trained from scratch)."""
        groups: list[dict] = []
        if self.use_text:
            decay, no_decay = [], []
            for n, p in self.text_encoder.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if (n.endswith("bias") or "orm" in n) else decay).append(p)
            groups += [
                {"params": decay, "lr": lr_bert, "weight_decay": weight_decay},
                {"params": no_decay, "lr": lr_bert, "weight_decay": 0.0},
            ]
        text_ids = {id(p) for p in (self.text_encoder.parameters() if self.use_text else [])}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in text_ids]
        groups.append({"params": rest, "lr": lr_head, "weight_decay": weight_decay})
        return groups


# --------------------------------------------------------------------------- #
# Multi-task loss (spec section 4.3)
# --------------------------------------------------------------------------- #
class MultiTaskLoss(nn.Module):
    """L = L_tags + alpha*MSE(valence) + beta*MSE(arousal).

    Every term is masked per sample, because Task 3 trains on a union of corpora
    with complementary annotation: MusicCaps rows carry tags but no
    valence/arousal, DEAM rows carry valence/arousal but no tags. An unmasked
    version would read each corpus's *missing* labels as negatives -- teaching the
    model that DEAM audio has no tags and that MusicCaps audio is neutral-valence
    -- which is a labelling error, not a supervision signal.

    ``target["y_mask"]`` is a (B,) bool selecting rows with real tag labels;
    ``target["valence_mask"]`` / ``["arousal_mask"]`` do the same for emotion.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha, self.beta = float(alpha), float(beta)
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, out: dict, target: dict) -> tuple[torch.Tensor, dict]:
        logits, y = out["tag_logits"], target["y"]
        y_mask = target.get("y_mask")
        if y_mask is not None:
            m = y_mask.bool()
            logits, y = logits[m], y[m]

        parts: dict[str, float] = {}
        if logits.shape[0] > 0:
            l_tags = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=self.pos_weight, reduction="mean"
            )
            parts["loss_tags"] = float(l_tags.detach())
        else:                          # batch happened to contain no tagged rows
            l_tags = out["tag_logits"].sum() * 0.0
        total = l_tags

        for key, weight in (("valence", self.alpha), ("arousal", self.beta)):
            if key in out and target.get(key) is not None and weight > 0:
                mask = target.get(f"{key}_mask")
                pred, gold = out[key], target[key]
                if mask is not None:
                    if mask.sum() == 0:
                        continue
                    pred, gold = pred[mask.bool()], gold[mask.bool()]
                term = F.mse_loss(pred, gold)
                total = total + weight * term
                parts[f"loss_{key}"] = float(term.detach())

        parts["loss_total"] = float(total.detach())
        return total, parts
