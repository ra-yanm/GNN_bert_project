"""BERT text encoder and the Task 1 multi-label tag classifier (spec section 4.1).

Task 1 model, exactly as specified:

    t     = BERT_CLS(X_text)
    y^_k  = sigmoid(w_k^T t + b_k)
    L     = -(1/K) * sum_k [ y_k log y^_k + (1 - y_k) log(1 - y^_k) ]

:class:`BertTextEncoder` is reused unchanged by Tasks 3 and 4 -- it exposes both
the pooled CLS vector ``t`` and the full token sequence ``H_text``, which is what
the Task 3 cross-attention block attends over.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .utils import Config, get_logger

log = get_logger("bert_encoder")


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class BertTextEncoder(nn.Module):
    """Wraps a HuggingFace encoder and returns (H_text, t).

    ``H_text`` : (B, L, d) token states -- the K/V source for cross-attention.
    ``t``      : (B, d) pooled sentence vector -- the spec's CLS vector.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        tcfg = cfg.text
        self.model_name = tcfg["model_name"]
        self.pooling = tcfg.get("pooling", "cls")
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = int(AutoConfig.from_pretrained(self.model_name).hidden_size)

        if tcfg.get("freeze_bert", False):
            for p in self.encoder.parameters():
                p.requires_grad = False
            log.info("BERT fully frozen (%s)", self.model_name)
        else:
            n_freeze = int(tcfg.get("freeze_first_n_layers", 0) or 0)
            if n_freeze > 0:
                self._freeze_lower(n_freeze)

    def _freeze_lower(self, n: int) -> None:
        """Freeze embeddings + the first n transformer blocks. Cheap way to cut
        CPU backward cost while still adapting the upper layers."""
        emb = getattr(self.encoder, "embeddings", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad = False
        layers = self._layers()
        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False
        log.info("froze embeddings + first %d/%d encoder blocks", n, len(layers))

    def _layers(self) -> nn.ModuleList:
        """DistilBERT nests blocks at .transformer.layer, BERT at .encoder.layer."""
        enc = self.encoder
        if hasattr(enc, "transformer") and hasattr(enc.transformer, "layer"):
            return enc.transformer.layer
        if hasattr(enc, "encoder") and hasattr(enc.encoder, "layer"):
            return enc.encoder.layer
        return nn.ModuleList()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = out.last_hidden_state                              # (B, L, d)
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).float()
            t = (H * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            t = H[:, 0]                                        # CLS
        return H, t


# --------------------------------------------------------------------------- #
# Task 1 classifier
# --------------------------------------------------------------------------- #
class BertTagClassifier(nn.Module):
    """BERT CLS -> linear head -> K tag logits (spec section 4.1)."""

    def __init__(self, cfg: Config, num_tags: int):
        super().__init__()
        self.encoder = BertTextEncoder(cfg)
        d = self.encoder.hidden_size
        self.dropout = nn.Dropout(cfg.dotted("model.fusion.dropout", 0.2))
        self.head = nn.Linear(d, num_tags)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)
        self.num_tags = num_tags

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, t = self.encoder(input_ids, attention_mask)
        return self.head(self.dropout(t))                      # logits (B, K)

    def param_groups(self, lr_bert: float, lr_head: float, weight_decay: float) -> list[dict]:
        """Discriminative learning rates: the pretrained encoder moves slowly,
        the randomly-initialised head moves fast. No weight decay on biases or
        LayerNorm, which is standard for BERT fine-tuning."""
        decay, no_decay = [], []
        for n, p in self.encoder.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (n.endswith("bias") or "LayerNorm" in n or "layer_norm" in n) else decay).append(p)
        return [
            {"params": decay, "lr": lr_bert, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr_bert, "weight_decay": 0.0},
            {"params": list(self.head.parameters()), "lr": lr_head, "weight_decay": weight_decay},
        ]


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
class MaskedBCELoss(nn.Module):
    """BCE-with-logits averaged over tags, i.e. the spec's (1/K) * sum_k [...].

    ``pos_weight`` is supported because the MusicCaps top-50 tag distribution is
    steep (the rarest tag appears in well under 2% of clips); without it the
    model collapses to predicting all-zeros and macro-F1 sits near 0.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="mean"
        )


def compute_pos_weight(y: np.ndarray, cap: float = 10.0) -> torch.Tensor:
    """(#neg / #pos) per tag, clipped. The cap matters: an uncapped weight on a
    tag with 5 positives out of 2000 is 399, which makes that tag's gradient
    dominate the batch and destabilises training."""
    y = np.asarray(y, dtype=np.float64)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class TextTagDataset(Dataset):
    """Tokenised captions + multi-hot tag targets.

    ``text_col`` selects the Task 1 variant: ``caption`` (naive) or
    ``caption_masked`` (tag surface forms removed -- the honest task).
    Tokenisation is done once up front; on CPU this measurably beats
    re-tokenising every epoch.
    """

    def __init__(self, df, cfg: Config, text_col: str = "caption"):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text["model_name"])
        texts = df[text_col].astype(str).tolist()
        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=int(cfg.text["max_length"]),
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.y = torch.tensor(np.stack(df["y"].to_list()), dtype=torch.float32)
        self.texts = texts
        self.ids = df["ytid"].tolist() if "ytid" in df.columns else list(range(len(df)))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "y": self.y[i],
            "idx": i,
        }


def truncation_report(dataset: TextTagDataset, cfg: Config) -> dict:
    """How many inputs actually hit max_length. Worth reporting: if most
    captions are truncated, ``text.max_length`` is the wrong knob to have set."""
    lens = dataset.attention_mask.sum(dim=1)
    max_len = int(cfg.text["max_length"])
    return {
        "max_length": max_len,
        "token_len_mean": float(lens.float().mean()),
        "token_len_p95": float(torch.quantile(lens.float(), 0.95)),
        "token_len_max": int(lens.max()),
        "frac_truncated": float((lens >= max_len).float().mean()),
    }
