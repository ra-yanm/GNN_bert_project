"""Evaluation metrics -- spec section 6, implemented exactly as written there.

Covers:
  * per-tag precision / recall / F1, Macro-F1 and Micro-F1
  * AUC-PR (mean average precision over tags)
  * MAE and R^2 for the DEAM valence/arousal regression
  * S_graph, the optional graph-coherence score
  * R@K retrieval recall for the Task 4 contrastive model

Threshold handling: multi-label F1 depends strongly on the decision threshold.
:func:`tune_threshold` picks it on the *validation* split only and the chosen
value is then frozen for test, so no test information leaks into the metric.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)

from .utils import get_logger

log = get_logger("metrics")


# --------------------------------------------------------------------------- #
# Multi-label tag metrics
# --------------------------------------------------------------------------- #
def multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.3,
    tag_names: list[str] | None = None,
    topk_report: int = 10,
) -> dict:
    """Full multi-label report for a (N, K) probability matrix.

    AUC-PR is computed only over tags that have at least one positive in
    ``y_true`` -- average_precision_score is undefined for an all-negative
    column, and silently averaging in a 0.0 there would understate the model.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.float64)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    samples_f1 = f1_score(y_true, y_pred, average="samples", zero_division=0)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
        labels=np.arange(y_true.shape[1]),
    )

    valid = y_true.sum(axis=0) > 0
    aucpr_per_tag = np.full(y_true.shape[1], np.nan)
    auroc_per_tag = np.full(y_true.shape[1], np.nan)
    for k in np.flatnonzero(valid):
        aucpr_per_tag[k] = average_precision_score(y_true[:, k], y_prob[:, k])
        if 0 < y_true[:, k].sum() < y_true.shape[0]:
            auroc_per_tag[k] = roc_auc_score(y_true[:, k], y_prob[:, k])

    out = {
        "threshold": float(threshold),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "samples_f1": float(samples_f1),
        "macro_precision": float(prec[valid].mean()) if valid.any() else 0.0,
        "macro_recall": float(rec[valid].mean()) if valid.any() else 0.0,
        "auc_pr": float(np.nanmean(aucpr_per_tag)),
        "auc_roc": float(np.nanmean(auroc_per_tag)),
        "n_samples": int(y_true.shape[0]),
        "n_tags": int(y_true.shape[1]),
        "n_tags_with_positives": int(valid.sum()),
    }

    if tag_names is not None:
        order = np.argsort(-np.nan_to_num(aucpr_per_tag, nan=-1))
        out["per_tag"] = {
            tag_names[k]: {
                "precision": round(float(prec[k]), 4),
                "recall": round(float(rec[k]), 4),
                "f1": round(float(f1[k]), 4),
                "auc_pr": (None if np.isnan(aucpr_per_tag[k]) else round(float(aucpr_per_tag[k]), 4)),
                "support": int(support[k]),
            }
            for k in order[:topk_report]
        }
    return out


def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Pick the global threshold maximising macro-F1. Call on validation only.
    Returns (best_threshold, best_macro_f1)."""
    grid = np.arange(0.05, 0.71, 0.025) if grid is None else grid
    best_t, best_f1 = 0.3, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def single_label_metrics(
    y_true: np.ndarray, logits: np.ndarray, class_names: list[str] | None = None
) -> dict:
    """Accuracy / macro-F1 / confusion matrix for single-label genre tasks."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(logits).argmax(axis=1)
    K = int(max(y_true.max(), y_pred.max())) + 1

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=np.arange(K)
    )
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    out = {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(y_true.size),
    }
    # one-vs-rest AUC-PR so the number is comparable with the multi-label tasks
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    onehot = np.eye(K)[y_true]
    valid = onehot.sum(axis=0) > 0
    out["auc_pr"] = float(np.mean([
        average_precision_score(onehot[:, k], probs[:, k]) for k in np.flatnonzero(valid)
    ]))
    if class_names:
        out["per_class"] = {
            class_names[k]: {
                "precision": round(float(prec[k]), 4),
                "recall": round(float(rec[k]), 4),
                "f1": round(float(f1[k]), 4),
                "support": int(support[k]),
            }
            for k in range(min(K, len(class_names)))
        }
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Emotion regression (spec section 6, "Emotion regression (DEAM)")
# --------------------------------------------------------------------------- #
def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, name: str = "valence", scale: tuple[float, float] | None = None
) -> dict:
    """MAE / RMSE / R^2 / Pearson r. If ``scale=(mu, sd)`` is given, both arrays
    are de-standardised first so MAE is reported on DEAM's original 1-9 scale --
    which is the only version comparable to published numbers."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if scale is not None:
        mu, sd = scale
        y_true = y_true * sd + mu
        y_pred = y_pred * sd + mu
    err = y_true - y_pred
    r = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 and y_pred.std() > 0 else 0.0
    return {
        f"mae_{name}": float(np.abs(err).mean()),
        f"rmse_{name}": float(np.sqrt((err ** 2).mean())),
        f"r2_{name}": float(r2_score(y_true, y_pred)),
        f"pearson_{name}": r,
    }


# --------------------------------------------------------------------------- #
# Retrieval (spec section 4.4)
# --------------------------------------------------------------------------- #
def recall_at_k(
    sim: np.ndarray, ks: tuple[int, ...] = (1, 5, 10), ground_truth: np.ndarray | None = None
) -> dict:
    """R@K in both directions from a (N_graph, N_text) similarity matrix.

    Row i is assumed to pair with column i unless ``ground_truth`` gives the
    correct column index per row. Also returns median rank and MRR, which are
    more stable than R@1 on a few-hundred-item test set.
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    gt = np.arange(n) if ground_truth is None else np.asarray(ground_truth)

    out: dict = {}
    for direction, mat, truth in (("g2t", sim, gt), ("t2g", sim.T, gt)):
        # rank of the correct item = how many scored strictly higher
        correct = mat[np.arange(n), truth]
        ranks = (mat > correct[:, None]).sum(axis=1) + 1
        for k in ks:
            out[f"R@{k}_{direction}"] = float((ranks <= k).mean())
        out[f"medr_{direction}"] = float(np.median(ranks))
        out[f"mrr_{direction}"] = float((1.0 / ranks).mean())

    for k in ks:                     # symmetric average, the headline number
        out[f"R@{k}"] = float(0.5 * (out[f"R@{k}_g2t"] + out[f"R@{k}_t2g"]))
    out["n_candidates"] = int(n)
    return out


# --------------------------------------------------------------------------- #
# Graph coherence (spec section 6, optional analysis)
# --------------------------------------------------------------------------- #
def graph_coherence(
    h: torch.Tensor, edge_index: torch.Tensor, tau: float = 0.5,
    edge_attr: torch.Tensor | None = None,
) -> dict:
    """S_graph = (1/|E|) * sum_{(i,j) in E} 1[cos(h_i, h_j) > tau].

    Implemented as the spec defines it, plus two diagnostics the bare score
    needs to be interpretable:

    ``s_random``  the same statistic over an equal number of *random* node pairs.
                  S_graph alone rises toward 1.0 as a deep GNN oversmooths, so it
                  only means something relative to this control.
    ``s_temporal`` / ``s_similarity``
                  the score restricted to each edge kind, which is what actually
                  tests whether structure-derived edges connect coherent regions.
    """
    hn = torch.nn.functional.normalize(h, dim=1)
    src, dst = edge_index[0], edge_index[1]
    cos = (hn[src] * hn[dst]).sum(dim=1)

    out = {"s_graph": float((cos > tau).float().mean()), "num_edges": int(cos.numel()),
           "mean_edge_cos": float(cos.mean())}

    n = hn.shape[0]
    if n > 2:
        g = torch.Generator().manual_seed(425)
        ri = torch.randint(0, n, (cos.numel(),), generator=g)
        rj = torch.randint(0, n, (cos.numel(),), generator=g)
        keep = ri != rj
        rcos = (hn[ri[keep]] * hn[rj[keep]]).sum(dim=1)
        out["s_random"] = float((rcos > tau).float().mean())
        out["lift_over_random"] = out["s_graph"] - out["s_random"]

    if edge_attr is not None and edge_attr.shape[1] >= 3:
        for name, col in (("s_temporal", 1), ("s_similarity", 2)):
            m = edge_attr[:, col] > 0.5
            out[name] = float((cos[m] > tau).float().mean()) if m.any() else None
    return out


# --------------------------------------------------------------------------- #
# Baseline predictors (spec section 8, B1)
# --------------------------------------------------------------------------- #
def random_baseline(y_true: np.ndarray, seed: int = 425) -> np.ndarray:
    """B1a: uniform random scores per tag."""
    rng = np.random.default_rng(seed)
    return rng.random(y_true.shape)


def prior_baseline(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """B1b: predict every tag at its training prevalence. Stronger than random
    and the correct floor for an imbalanced multi-label problem."""
    prior = np.asarray(y_train, dtype=np.float64).mean(axis=0)
    return np.tile(prior, (n_test, 1))


def majority_baseline(y_train: np.ndarray, n_test: int, n_classes: int) -> np.ndarray:
    """B1c: single-label majority class, as logits."""
    counts = np.bincount(np.asarray(y_train).astype(int), minlength=n_classes)
    logits = np.zeros((n_test, n_classes))
    logits[:, int(counts.argmax())] = 10.0
    return logits
