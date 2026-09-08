"""Shared training-loop machinery for all four tasks.

Kept separate from the task scripts so that the optimiser schedule, history
tracking, and text-batch handling are provably identical across tasks -- if the
Task 3 ablations used a different warmup than Task 1, the comparison in the
report would not be a comparison of architectures.

Contents:
  * :func:`trim_text_batch` -- dynamic padding, the single biggest CPU speedup.
  * :func:`build_optimizer` -- AdamW + linear-warmup/cosine-decay schedule.
  * :class:`History`        -- per-epoch metric log, dumped for the learning curves.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .utils import get_logger, human_time

log = get_logger("trainer")


# --------------------------------------------------------------------------- #
# Text batching
# --------------------------------------------------------------------------- #
def trim_text_batch(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, multiple_of: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut a batch padded to ``max_length`` down to its longest real sequence.

    Datasets here tokenise once up front with ``padding="max_length"`` (192), but
    MusicCaps captions average well under that. Transformer cost is linear in
    sequence length for the FFN and quadratic for attention, so trimming per batch
    is a large saving on CPU with *identical* outputs -- the removed columns are
    pure padding and are masked out anyway.

    Rounded up to a multiple of 8 to keep the GEMM shapes friendly.
    """
    real = int(attention_mask.sum(dim=1).max().item())
    real = max(multiple_of, min(input_ids.shape[1], int(math.ceil(real / multiple_of) * multiple_of)))
    if real >= input_ids.shape[1]:
        return input_ids, attention_mask
    return input_ids[:, :real].contiguous(), attention_mask[:, :real].contiguous()


# --------------------------------------------------------------------------- #
# Optimiser + schedule
# --------------------------------------------------------------------------- #
def build_optimizer(
    param_groups: list[dict] | torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    total_steps: int = 1000,
    warmup_ratio: float = 0.1,
    schedule: str = "cosine",
) -> tuple[AdamW, LambdaLR]:
    """AdamW with linear warmup then cosine decay (or constant).

    Warmup matters for the BERT-bearing tasks: a cold linear head produces large
    gradients that, applied at full LR to a pretrained encoder on step 1, undo
    the pretraining. ``LambdaLR`` scales each group's own ``lr``, so the
    discriminative rates set in ``param_groups`` are preserved.
    """
    if isinstance(param_groups, torch.nn.Module):
        param_groups = [{"params": [p for p in param_groups.parameters() if p.requires_grad],
                         "lr": lr, "weight_decay": weight_decay}]
    opt = AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    warmup = max(1, int(warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        if schedule != "cosine":
            return 1.0
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    return opt, LambdaLR(opt, lr_lambda)


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #
@dataclass
class History:
    """Per-epoch metric log. Saved into results/metrics.json and read straight
    back by the plotting code, so every learning curve in the report comes from
    the actual run rather than being redrawn by hand."""

    rows: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def log_epoch(self, epoch: int, **kwargs) -> None:
        row = {"epoch": epoch, "elapsed_s": round(time.time() - self.t0, 1)}
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in kwargs.items()})
        self.rows.append(row)

    def series(self, key: str) -> list[float]:
        return [r[key] for r in self.rows if key in r]

    def to_list(self) -> list[dict]:
        return self.rows

    def summary_line(self, epoch: int, keys: tuple[str, ...]) -> str:
        row = self.rows[-1]
        bits = [f"{k}={row[k]:.4f}" for k in keys if k in row and isinstance(row[k], float)]
        return f"epoch {epoch:3d} | " + " | ".join(bits) + f" | {human_time(row['elapsed_s'])}"


# --------------------------------------------------------------------------- #
def clip_and_step(
    model: torch.nn.Module, opt: AdamW, sched: LambdaLR, max_norm: float = 1.0
) -> float:
    """Clip gradients, step, zero. Returns the pre-clip grad norm, which is worth
    watching: a spike is the first visible sign of a diverging temperature or an
    uncapped pos_weight."""
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm
    )
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    return float(norm)


def describe_model(model: torch.nn.Module, name: str) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("%s: %.2fM trainable / %.2fM total params", name, trainable / 1e6, total / 1e6)
    return {"name": name, "params_trainable": int(trainable), "params_total": int(total)}
