"""Turn trained runs into the report's tables and figures.

Reads ``results/metrics.json`` (written by train.py) and ``results/preds/*.npz``
(the raw test-set predictions), so nothing is retrained here and no number is
recomputed from a different checkpoint than the one that produced it.

    python scripts/evaluate.py                # everything that is available
    python scripts/evaluate.py --task 2       # one task
    python scripts/evaluate.py --skip-tsne    # t-SNE is the slow part

Outputs
-------
results/plots/*.png          every figure
results/tables/*.md, *.tex   every table, markdown for the README, LaTeX for report/
results/retrieval_examples/  Task 4 qualitative retrievals (json + markdown)
results/metrics.json         extended with the analyses computed here
                             (graph coherence, dataset statistics)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                # no display on this box
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import seaborn as sns                                 # noqa: E402
import torch                                          # noqa: E402
from sklearn.metrics import precision_recall_curve     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import GTZAN_GENRES                                        # noqa: E402
from src.graph_dataset import (                                                   # noqa: E402
    GraphDataset, collate_graphs, filter_split, load_cached_graphs,
)
from src.metrics import graph_coherence, multilabel_metrics                       # noqa: E402
from src.utils import (                                                           # noqa: E402
    ROOT, get_device, load_config, load_metrics, resolve, save_metrics, set_seed, setup_logging,
)

log = logging.getLogger("evaluate")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
PALETTE = sns.color_palette("colorblind")
DPI = 160


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def plots_dir() -> Path:
    d = ROOT / "results" / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def tables_dir() -> Path:
    d = ROOT / "results" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_preds(run: str) -> dict | None:
    p = ROOT / "results" / "preds" / f"{run}.npz"
    if not p.exists():
        return None
    return dict(np.load(p, allow_pickle=True))


def savefig(fig, name: str) -> Path:
    out = plots_dir() / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  figure -> results/plots/%s.png", name)
    return out


def write_table(name: str, header: list[str], rows: list[list], caption: str = "",
                colspec: str = "") -> None:
    """Emit the same table twice: markdown for the README, LaTeX (booktabs) for
    the report. Writing both from one source keeps them from drifting apart.

    ``colspec`` overrides the LaTeX column specifier. The default -- ``l`` then ``r``
    for every remaining column -- is what the numeric tables want, but ``r`` is a
    rigid column: it cannot line-break, so a table holding free text silently runs
    past the right margin instead of wrapping. Such a table has to name ``p{}``
    columns explicitly. Markdown is unaffected either way; it wraps on its own.
    """
    def fmt(v):
        if isinstance(v, float):
            return "--" if not np.isfinite(v) else f"{v:.4f}"
        return str(v)

    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    (tables_dir() / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&")            # noqa: E731
    spec = colspec or ("l" + "r" * (len(header) - 1))
    tex = [r"\begin{table}[t]", r"\centering",
           r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    tex += [" & ".join([esc(r[0])] + [fmt(v) for v in r[1:]]) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        tex += [rf"\caption{{{esc(caption)}}}", rf"\label{{tab:{name}}}"]
    tex += [r"\end{table}"]
    (tables_dir() / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    log.info("  table   -> results/tables/%s.{md,tex}", name)


def curve(ax, hist: list[dict], key: str, label: str, color=None) -> None:
    xs = [r["epoch"] for r in hist if key in r]
    ys = [r[key] for r in hist if key in r]
    if xs:
        ax.plot(xs, ys, marker="o", ms=3, label=label, color=color)


# =========================================================================== #
# TASK 1
# =========================================================================== #
def eval_task1(cfg, m: dict, args) -> dict:
    t1 = m.get("task1")
    if not t1:
        log.warning("no task1 metrics -- skipping")
        return {}
    log.info("Task 1 figures and tables")

    # ---- table: model vs baselines --------------------------------------
    rows = []
    for key, name in (("B1_random", "B1 random scores"),
                      ("B1_prior", "B1 training prior"),
                      ("B0_lexical_caption", "B0 lexical match (raw caption)"),
                      ("B0_lexical_caption_masked", "B0 lexical match (masked caption)"),
                      ("task1_bert_naive", "DistilBERT (raw caption)"),
                      ("task1_bert_masked", "DistilBERT (masked caption)")):
        d = t1.get(key)
        if not d:
            continue
        r = d.get("test", d)
        rows.append([name, r.get("macro_f1", float("nan")), r.get("micro_f1", float("nan")),
                     r.get("samples_f1", float("nan")), r.get("auc_pr", float("nan")),
                     r.get("threshold", float("nan"))])
    write_table("task1_results", ["Model", "Macro-F1", "Micro-F1", "Samples-F1", "AUC-PR", "Thr"],
                rows, "Task 1: caption to multi-label tag prediction on the MusicCaps test split. "
                      "The masked variant has every tag surface form removed from the caption.")

    # ---- learning curves -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, (variant, color) in enumerate((("naive", PALETTE[0]), ("masked", PALETTE[1]))):
        d = t1.get(f"task1_bert_{variant}")
        if not d:
            continue
        curve(axes[0], d["history"], "train_loss", variant, color)
        curve(axes[1], d["history"], "val_macro_f1", variant, color)
    axes[0].set(xlabel="epoch", ylabel="training loss", title="Task 1 training loss")
    axes[1].set(xlabel="epoch", ylabel="val macro-F1", title="Task 1 validation macro-F1")
    for a in axes:
        a.legend(title="caption")
    savefig(fig, "task1_learning_curves")

    # ---- per-tag F1, and where the lexical shortcut lives ----------------
    out: dict = {}
    pn, pm = load_preds("task1_bert_naive"), load_preds("task1_bert_masked")
    if pm is not None:
        tags = [str(t) for t in pm["tag_names"]]
        thr_m = t1["task1_bert_masked"]["threshold_from_val"]
        f1_m = _per_tag_f1(pm["y_true"], pm["y_prob"], thr_m)
        support = pm["y_true"].sum(axis=0)
        order = np.argsort(-support)[:25]

        if pn is not None:
            thr_n = t1["task1_bert_naive"]["threshold_from_val"]
            f1_n = _per_tag_f1(pn["y_true"], pn["y_prob"], thr_n)
            out["per_tag_shortcut"] = {
                tags[i]: round(float(f1_n[i] - f1_m[i]), 4) for i in order[:15]
            }
        else:
            f1_n = None

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(order))
        if f1_n is None:
            ax.bar(x, f1_m[order], 0.7, label="masked caption", color=PALETTE[1])
        else:
            ax.bar(x - 0.2, f1_n[order], 0.4, label="raw caption", color=PALETTE[0])
            ax.bar(x + 0.2, f1_m[order], 0.4, label="masked caption", color=PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels([tags[i] for i in order], rotation=60, ha="right", fontsize=7)
        ax.set(ylabel="per-tag F1", title="Task 1 per-tag F1, 25 most frequent tags")
        ax.legend()
        savefig(fig, "task1_per_tag_f1")

        # ---- support vs difficulty ----------------------------------------
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.scatter(support, f1_m, s=18, color=PALETTE[1], label="masked caption")
        if f1_n is not None:
            ax.scatter(support, f1_n, s=18, color=PALETTE[0], alpha=0.6, label="raw caption")
        r = float(np.corrcoef(np.log1p(support), f1_m)[0, 1])
        ax.set(xscale="log", xlabel="tag support in the test split (log)", ylabel="per-tag F1",
               title=f"Task 1 F1 vs tag frequency (r = {r:.2f} on log support)")
        ax.legend(fontsize=7)
        savefig(fig, "task1_f1_vs_support")
        out["f1_log_support_correlation"] = round(r, 4)

        # ---- precision-recall curves for the most frequent tags ----------
        fig, ax = plt.subplots(figsize=(5, 4))
        for j, i in enumerate(order[:6]):
            pr, rc, _ = precision_recall_curve(pm["y_true"][:, i], pm["y_prob"][:, i])
            ax.plot(rc, pr, label=f"{tags[i]} (n={int(support[i])})", color=PALETTE[j % len(PALETTE)])
        ax.set(xlabel="recall", ylabel="precision",
               title="Task 1 precision-recall (masked captions)")
        ax.legend(fontsize=7)
        savefig(fig, "task1_precision_recall")

        # ---- threshold sensitivity ---------------------------------------
        grid = np.arange(0.05, 0.96, 0.025)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for d, lbl, color in ((pn, "raw caption", PALETTE[0]), (pm, "masked caption", PALETTE[1])):
            if d is None:
                continue
            f1s = [multilabel_metrics(d["y_true"], d["y_prob"], t)["macro_f1"] for t in grid]
            ax.plot(grid, f1s, label=lbl, color=color)
        ax.axvline(thr_m, ls=":", color="grey", lw=1)
        ax.set(xlabel="decision threshold", ylabel="test macro-F1",
               title="Task 1 threshold sensitivity")
        ax.legend()
        savefig(fig, "task1_threshold_sensitivity")

    if "lexical_shortcut_macro_f1_gap" in t1:
        out["lexical_shortcut_macro_f1_gap"] = t1["lexical_shortcut_macro_f1_gap"]
    return out


def _per_tag_f1(y: np.ndarray, p: np.ndarray, thr: float) -> np.ndarray:
    pred = (p >= thr).astype(float)
    tp = (pred * y).sum(0)
    return np.divide(2 * tp, pred.sum(0) + y.sum(0), out=np.zeros(y.shape[1]),
                     where=(pred.sum(0) + y.sum(0)) > 0)


# =========================================================================== #
# TASK 2
# =========================================================================== #
TASK2_LABELS = {
    "B1_majority": "B1 majority class",
    "B2_mel_cnn": "B2 CNN on log-mel",
    "B4_feature_mlp": "B4 mean-pooled features + MLP",
    "gnn_sage_segment": "GraphSAGE, segment graph (ours)",
    "gnn_gat_segment": "GAT, segment graph",
    "gnn_sage_chord": "GraphSAGE, chord-transition graph",
    "gnn_sage_temporal_only": "GraphSAGE, temporal edges only",
    "gnn_sage_similarity_only": "GraphSAGE, similarity edges only",
    "gnn_sage_no_edges": "GraphSAGE, self-loops only",
}


def eval_task2(cfg, m: dict, args) -> dict:
    t2 = m.get("task2")
    if not t2:
        log.warning("no task2 metrics -- skipping")
        return {}
    log.info("Task 2 figures and tables")

    rows = []
    for key, name in TASK2_LABELS.items():
        d = t2.get(key)
        if not d or "skipped" in d:
            continue
        r = d.get("test", d)
        rows.append([name, r.get("accuracy", float("nan")), r.get("macro_f1", float("nan")),
                     r.get("auc_pr", float("nan")),
                     d.get("model", {}).get("params_trainable", 0) / 1e6,
                     d.get("best_epoch", -1)])
    write_table("task2_results",
                ["Model", "Accuracy", "Macro-F1", "AUC-PR", "Params (M)", "Best epoch"], rows,
                "Task 2: GTZAN genre classification on the fault-filtered test split "
                "(290 tracks). Parameter counts are trainable parameters.")

    # ---- ablation bar chart ---------------------------------------------
    # B1_majority is recorded flat (it has no training history), the rest nest
    # their scores under "test"; ``.get("test", d)`` handles both.
    keys = [k for k in TASK2_LABELS if k in t2 and "skipped" not in t2[k]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    vals = [t2[k].get("test", t2[k])["macro_f1"] for k in keys]
    colors = [PALETTE[2] if k.startswith("B") else
              (PALETTE[0] if k == "gnn_sage_segment" else PALETTE[1]) for k in keys]
    ax.barh(range(len(keys)), vals, color=colors)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([TASK2_LABELS[k] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="test macro-F1", title="Task 2: models, baselines and ablations")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
    savefig(fig, "task2_ablations")

    # ---- learning curves -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for j, k in enumerate([k for k in keys if "history" in t2[k]]):
        curve(axes[0], t2[k]["history"], "train_loss", TASK2_LABELS[k], PALETTE[j % len(PALETTE)])
        curve(axes[1], t2[k]["history"], "val_macro_f1", TASK2_LABELS[k], PALETTE[j % len(PALETTE)])
    axes[0].set(xlabel="epoch", ylabel="training loss", title="Task 2 training loss")
    axes[1].set(xlabel="epoch", ylabel="val macro-F1", title="Task 2 validation macro-F1")
    axes[1].legend(fontsize=6, loc="lower right")
    savefig(fig, "task2_learning_curves")

    # ---- confusion matrix ------------------------------------------------
    best = max((k for k in keys if k.startswith("gnn")),
               key=lambda k: t2[k]["test"]["macro_f1"], default=None)
    if best:
        cm = np.array(t2[best]["test"]["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        sns.heatmap(norm, annot=cm.astype(int), fmt="d", cmap="Blues", cbar_kws={"label": "row-normalised"},
                    xticklabels=GTZAN_GENRES, yticklabels=GTZAN_GENRES, ax=ax, annot_kws={"size": 7})
        ax.set(xlabel="predicted", ylabel="true",
               title=f"Task 2 confusion matrix ({TASK2_LABELS[best]})")
        savefig(fig, "task2_confusion_matrix")

        # ---- per-class F1 ------------------------------------------------
        pc = t2[best]["test"].get("per_class", {})
        if pc:
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            names = list(pc)
            ax.bar(names, [pc[n]["f1"] for n in names], color=PALETTE[0])
            ax.set(ylabel="F1", title=f"Task 2 per-genre F1 ({TASK2_LABELS[best]})")
            ax.tick_params(axis="x", rotation=45)
            savefig(fig, "task2_per_class_f1")

    # ---- t-SNE + graph coherence on the trained encoder ------------------
    out: dict = {}
    if best:
        out |= _task2_embedding_analysis(cfg, best, t2[best], skip_tsne=args.skip_tsne)
    return out


def _task2_embedding_analysis(cfg, run: str, entry: dict, skip_tsne: bool = False) -> dict:
    """t-SNE of the pooled graph vectors, plus S_graph on the trained node states.

    S_graph has to be measured on *trained* node embeddings to mean anything, so
    it is computed here rather than in train.py: the checkpoint is reloaded and run
    over the test graphs.
    """
    from sklearn.manifold import TSNE

    from src.gnn_model import GNNClassifier

    ck = ROOT / "results" / "checkpoints" / f"task2_{run}.pt"
    if not ck.exists():
        log.warning("checkpoint %s missing -- skipping embedding analysis", ck.name)
        return {}

    kind = "chord" if "chord" in run else "segment"
    cfg["model"]["gnn"]["conv"] = "gat" if "gat" in run else "sage"
    graphs = filter_split(load_cached_graphs(cfg, "gtzan", kind), "test")
    device = get_device(cfg)
    model = GNNClassifier(cfg, int(graphs[0].x.shape[1]), len(GTZAN_GENRES)).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    model.eval()

    from torch.utils.data import DataLoader
    loader = DataLoader(GraphDataset(graphs), batch_size=32, collate_fn=collate_graphs)

    G, Y, cached = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            g, h = model.embed(batch)
            G.append(g.cpu().numpy())
            Y.append(batch.y.view(-1).cpu().numpy())
            cached.append((h.cpu(), batch.edge_index.cpu(), batch.edge_attr.cpu()))
    G, Y = np.concatenate(G), np.concatenate(Y)

    # S_graph as the spec writes it, swept over tau.
    #
    # At the configured tau=0.5 the score saturates: post-ReLU node states sit in
    # the positive orthant, so nearly every pair -- connected or not -- has cosine
    # above 0.5 and even the random control scores >0.95. The absolute number is
    # therefore uninformative on its own, and the honest reading is the *lift* over
    # the random control and the tau at which the two curves separate most. Both
    # are reported.
    taus = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
    tau_cfg = float(cfg.dotted("eval.graph_coherence_tau", 0.5))
    if tau_cfg not in taus:
        taus = sorted(taus + [tau_cfg])
    sweep: dict[float, dict] = {}
    for tau in taus:
        rows = [graph_coherence(h, ei, tau=tau, edge_attr=ea) for h, ei, ea in cached]
        sweep[tau] = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
                      for k in ("s_graph", "s_random", "mean_edge_cos", "s_temporal", "s_similarity")
                      if any(r.get(k) is not None for r in rows)}
        sweep[tau]["lift_over_random"] = sweep[tau]["s_graph"] - sweep[tau]["s_random"]

    agg = dict(sweep[tau_cfg])
    agg["tau"] = tau_cfg
    agg["num_test_graphs"] = int(len(graphs))
    best_tau = max(sweep, key=lambda t: sweep[t]["lift_over_random"])
    agg["most_discriminative_tau"] = float(best_tau)
    agg["lift_at_most_discriminative_tau"] = sweep[best_tau]["lift_over_random"]
    agg["sweep"] = {str(t): sweep[t] for t in taus}
    log.info("tau=%.3f: S_graph=%.4f vs s_random=%.4f (lift %+.4f) | mean edge cos %.4f",
             tau_cfg, agg["s_graph"], agg["s_random"], agg["lift_over_random"],
             agg.get("mean_edge_cos", float("nan")))
    log.info("most discriminative tau=%.3f: S_graph=%.4f vs s_random=%.4f (lift %+.4f)",
             best_tau, sweep[best_tau]["s_graph"], sweep[best_tau]["s_random"],
             sweep[best_tau]["lift_over_random"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(taus, [sweep[t]["s_graph"] for t in taus], marker="o", ms=4,
                 label="$S_{graph}$ (real edges)", color=PALETTE[0])
    axes[0].plot(taus, [sweep[t]["s_random"] for t in taus], marker="s", ms=4,
                 label="random node pairs", color=PALETTE[3])
    axes[0].axvline(tau_cfg, ls=":", color="grey", lw=1)
    axes[0].set(xlabel=r"threshold $\tau$", ylabel=r"fraction of pairs with $\cos > \tau$",
                title="Graph coherence and its control")
    axes[0].legend(fontsize=7)
    names = [k for k in ("s_graph", "s_random", "s_temporal", "s_similarity")
             if k in sweep[best_tau]]
    axes[1].bar(names, [sweep[best_tau][k] for k in names],
                color=[PALETTE[0] if k == "s_graph" else PALETTE[3] if k == "s_random"
                       else PALETTE[1] for k in names])
    axes[1].set(ylabel=f"fraction with cos > {best_tau}",
                title=rf"By edge kind at $\tau$ = {best_tau}")
    axes[1].tick_params(axis="x", rotation=20)
    savefig(fig, "graph_coherence")
    write_table("graph_coherence",
                ["tau", "S_graph", "Random pairs", "Lift", "Temporal edges", "Similarity edges"],
                [[t, sweep[t]["s_graph"], sweep[t]["s_random"], sweep[t]["lift_over_random"],
                  sweep[t].get("s_temporal", float("nan")),
                  sweep[t].get("s_similarity", float("nan"))] for t in taus],
                f"Graph coherence S_graph over {agg['num_test_graphs']} GTZAN test graphs, swept "
                f"over tau. Mean cosine on real edges is "
                f"{agg.get('mean_edge_cos', float('nan')):.4f}. The score saturates at low tau "
                f"because post-ReLU node states are nearly collinear, so the lift over the "
                f"random-pair control is the interpretable quantity.")

    perp = min(float(cfg.dotted("eval.tsne.perplexity", 30)), max(5.0, (len(G) - 1) / 3))
    if skip_tsne:
        return {"graph_coherence": agg}
    emb = TSNE(n_components=2, perplexity=perp, metric=cfg.dotted("eval.tsne.metric", "cosine"),
               init="pca", random_state=cfg.get("seed", 425)).fit_transform(G)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for gi, genre in enumerate(GTZAN_GENRES):
        s = Y == gi
        ax.scatter(emb[s, 0], emb[s, 1], s=14, label=genre, alpha=0.8,
                   color=sns.color_palette("tab10")[gi])
    ax.set(title=f"t-SNE of pooled graph embeddings g (perplexity {perp:g})",
           xlabel="dim 1", ylabel="dim 2")
    ax.legend(fontsize=6, ncol=2, markerscale=0.8)
    savefig(fig, "task2_tsne_graph_embeddings")

    return {"graph_coherence": agg, "tsne_perplexity": float(perp)}


# =========================================================================== #
# TASK 3
# =========================================================================== #
def eval_task3(cfg, m: dict, args) -> dict:
    t3 = m.get("task3")
    if not t3:
        log.warning("no task3 metrics -- skipping")
        return {}
    log.info("Task 3 figures and tables")
    modes = [k[len("task3_"):] for k in t3 if k.startswith("task3_")]

    rows = []
    for mode in modes:
        d = t3[f"task3_{mode}"]
        tg, em = d["test_tags"], d["test_emotion"]
        rows.append([mode, tg["macro_f1"], tg["micro_f1"], tg["auc_pr"],
                     em.get("mae_valence", float("nan")), em.get("r2_valence", float("nan")),
                     em.get("mae_arousal", float("nan")), em.get("r2_arousal", float("nan")),
                     d["model"]["params_trainable"] / 1e6])
    write_table("task3_results",
                ["Fusion", "Macro-F1", "Micro-F1", "AUC-PR", "MAE val", "R2 val",
                 "MAE aro", "R2 aro", "Params (M)"], rows,
                "Task 3: multi-context fusion. Tags are scored on the MusicCaps test split, "
                "valence/arousal on the DEAM test split (MAE on the original 1-9 scale).")

    # ---- ablation comparison --------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    x = np.arange(len(modes))
    axes[0].bar(x, [t3[f"task3_{k}"]["test_tags"]["macro_f1"] for k in modes], color=PALETTE[0])
    axes[0].set(ylabel="macro-F1", title="Tags (MusicCaps test)")
    axes[1].bar(x, [t3[f"task3_{k}"]["test_emotion"].get("mae_valence", np.nan) for k in modes],
                color=PALETTE[1])
    axes[1].set(ylabel="MAE (1-9 scale)", title="Valence (DEAM test)")
    axes[2].bar(x, [t3[f"task3_{k}"]["test_emotion"].get("mae_arousal", np.nan) for k in modes],
                color=PALETTE[2])
    axes[2].set(ylabel="MAE (1-9 scale)", title="Arousal (DEAM test)")
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(modes, rotation=25, ha="right", fontsize=8)
    fig.suptitle("Task 3 fusion ablation", y=1.02)
    savefig(fig, "task3_ablations")

    # ---- learning curves -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for j, mode in enumerate(modes):
        h = t3[f"task3_{mode}"]["history"]
        curve(axes[0], h, "train_loss", mode, PALETTE[j % len(PALETTE)])
        curve(axes[1], h, "val_loss", mode, PALETTE[j % len(PALETTE)])
    axes[0].set(xlabel="epoch", ylabel="training loss", title="Task 3 training loss")
    axes[1].set(xlabel="epoch", ylabel="validation loss", title="Task 3 validation loss")
    axes[1].legend(fontsize=7)
    savefig(fig, "task3_learning_curves")

    # ---- predicted vs true emotion, and t-SNE of z ----------------------
    best = max(modes, key=lambda k: t3[f"task3_{k}"]["test_tags"]["macro_f1"])
    p = load_preds(f"task3_{best}")
    out: dict = {"best_fusion_by_macro_f1": best}
    if p is None:
        return out

    vs, as_ = t3.get("valence_scale"), t3.get("arousal_scale")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, key, scale, name in ((axes[0], "valence", vs, "valence"),
                                 (axes[1], "arousal", as_, "arousal")):
        mask = p[f"{key}_mask"].astype(bool)
        if not mask.any():
            continue
        yt, yp = p[key][mask], p[f"{key}_pred"][mask]
        if scale:
            yt, yp = yt * scale[1] + scale[0], yp * scale[1] + scale[0]
        ax.scatter(yt, yp, s=12, alpha=0.5, color=PALETTE[0])
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=1)
        r = float(np.corrcoef(yt, yp)[0, 1])
        ax.set(xlabel=f"true {name}", ylabel=f"predicted {name}",
               title=f"{name} (r = {r:.3f})")
    fig.suptitle(f"Task 3 emotion regression on DEAM test ({best})", y=1.03)
    savefig(fig, "task3_emotion_scatter")

    if not args.skip_tsne and "z" in p:
        from sklearn.manifold import TSNE
        z, ymask = p["z"], p["y_mask"].astype(bool)
        perp = min(float(cfg.dotted("eval.tsne.perplexity", 30)), max(5.0, (len(z) - 1) / 3))
        emb = TSNE(n_components=2, perplexity=perp, metric="cosine", init="pca",
                   random_state=cfg.get("seed", 425)).fit_transform(z)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(emb[ymask, 0], emb[ymask, 1], s=12, alpha=0.7, label="MusicCaps (tags)",
                   color=PALETTE[0])
        ax.scatter(emb[~ymask, 0], emb[~ymask, 1], s=12, alpha=0.7, label="DEAM (valence/arousal)",
                   color=PALETTE[1])
        ax.set(title=f"t-SNE of the fused representation z ({best})")
        ax.legend(fontsize=7)
        savefig(fig, "task3_tsne_fused_z")

        # colour the same layout by predicted arousal: if fusion learned anything
        # transferable, the DEAM points should show a gradient
        amask = p["arousal_mask"].astype(bool)
        if amask.any():
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            sc = ax.scatter(emb[amask, 0], emb[amask, 1], s=14,
                            c=p["arousal"][amask], cmap="coolwarm")
            fig.colorbar(sc, ax=ax, label="true arousal (z-scored)")
            ax.set(title="Task 3 fused space coloured by true arousal")
            savefig(fig, "task3_tsne_by_arousal")

    # ---- case studies (spec: 3) ------------------------------------------
    out["case_studies"] = _task3_case_studies(cfg, t3, best, p)
    return out


def _task3_case_studies(cfg, t3: dict, mode: str, p: dict) -> list[dict]:
    """Pick the N most-confidently-tagged MusicCaps test clips and dump what the
    model predicted against the ground truth. Written to results/tables/ as
    markdown so the report can quote them verbatim."""
    n = int(cfg.dotted("eval.num_case_studies", 3))
    tags = [str(t) for t in p["tag_names"]]
    thr = t3[f"task3_{mode}"]["threshold_from_val"]
    ymask = p["y_mask"].astype(bool)
    idx = np.flatnonzero(ymask)
    if idx.size == 0:
        return []

    # rank by per-sample F1 against the frozen threshold, then take a spread:
    # best, median and worst, which is more informative than three easy wins
    pred = (p["y_prob"][idx] >= thr).astype(float)
    gold = p["y_true"][idx]
    tp = (pred * gold).sum(1)
    f1 = np.divide(2 * tp, pred.sum(1) + gold.sum(1), out=np.zeros(len(idx)),
                   where=(pred.sum(1) + gold.sum(1)) > 0)
    order = np.argsort(-f1)
    picks = [order[0], order[len(order) // 2], order[-1]][:n]

    rows, studies = [], []
    for rank, j in enumerate(picks):
        i = idx[j]
        top = np.argsort(-p["y_prob"][i])[:6]
        study = {
            "rank": ["best", "median", "worst"][min(rank, 2)],
            "sample_f1": round(float(f1[j]), 4),
            "true_tags": [tags[k] for k in np.flatnonzero(p["y_true"][i] > 0.5)],
            "predicted_tags": [tags[k] for k in np.flatnonzero(p["y_prob"][i] >= thr)],
            "top6_scores": {tags[k]: round(float(p["y_prob"][i][k]), 3) for k in top},
        }
        studies.append(study)
        rows.append([study["rank"], study["sample_f1"],
                     ", ".join(study["true_tags"][:6]),
                     ", ".join(study["predicted_tags"][:6])])
    write_table("task3_case_studies", ["Case", "Sample F1", "True tags", "Predicted tags"], rows,
                f"Task 3 case studies ({mode} fusion): best, median and worst-scoring "
                f"MusicCaps test clips at the validation-selected threshold {thr:.3f}.",
                # The two tag columns hold comma-separated lists up to six tags long -- the
                # median row alone runs to ~60 characters -- and an `r` column cannot break
                # a line, so the default spec pushed those rows 128pt past the right margin
                # and the text was cut off. p{} wraps them instead. 0.34\linewidth each fits
                # inside the 16.6cm text block with room to spare once the `l`, `r` and the
                # eight \tabcolsep gaps are paid for, and being a fraction rather than a
                # fixed width it survives a change of geometry. \raggedright because a
                # justified 160pt column full of unhyphenatable two-word tags ("amateur
                # recording") stretches the interword space badly; \arraybackslash puts back
                # the row-ending \\ that \raggedright redefines.
                colspec=r"lr>{\raggedright\arraybackslash}p{0.34\linewidth}"
                        r">{\raggedright\arraybackslash}p{0.34\linewidth}")
    (tables_dir() / "task3_case_studies.json").write_text(
        json.dumps(studies, indent=2), encoding="utf-8")
    return studies


# =========================================================================== #
# TASK 4
# =========================================================================== #
def eval_task4(cfg, m: dict, args) -> dict:
    t4 = m.get("task4")
    if not t4:
        log.warning("no task4 metrics -- skipping")
        return {}
    log.info("Task 4 figures and tables")
    r, rand = t4["test_retrieval"], t4["random_control_retrieval"]
    # The retrieval dicts hold both symmetric ("R@10") and directional ("R@10_g2t")
    # entries, so K has to be read from the symmetric keys only -- splitting on "@"
    # across all of them yields "10_g2t" and int() raises.
    ks = sorted(int(mm.group(1)) for k in rand if (mm := re.fullmatch(r"R@(\d+)", k)))

    write_table("task4_retrieval",
                ["Direction"] + [f"R@{k}" for k in ks] + ["Median rank", "MRR"],
                [["Audio -> caption"] + [r[f"R@{k}_g2t"] for k in ks] + [r["medr_g2t"], r["mrr_g2t"]],
                 ["Caption -> audio"] + [r[f"R@{k}_t2g"] for k in ks] + [r["medr_t2g"], r["mrr_t2g"]],
                 ["Symmetric mean"] + [r[f"R@{k}"] for k in ks] + [float("nan"), float("nan")],
                 ["Random control"] + [rand[f"R@{k}"] for k in ks] + [float("nan"), float("nan")]],
                f"Task 4: contrastive retrieval over all {r['n_candidates']} MusicCaps test "
                f"candidates (not in-batch). The random control scores a shuffled "
                f"similarity matrix of the same shape.")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(ks))
    w = 0.26
    axes[0].bar(x - w, [r[f"R@{k}_g2t"] for k in ks], w, label="audio -> caption", color=PALETTE[0])
    axes[0].bar(x, [r[f"R@{k}_t2g"] for k in ks], w, label="caption -> audio", color=PALETTE[1])
    axes[0].bar(x + w, [rand[f"R@{k}"] for k in ks], w, label="random control", color=PALETTE[7])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"R@{k}" for k in ks])
    axes[0].set(ylabel="recall", title=f"Task 4 retrieval ({r['n_candidates']} candidates)")
    axes[0].legend(fontsize=7)

    h = t4["history"]
    curve(axes[1], h, "train_loss", "InfoNCE loss", PALETTE[0])
    ax2 = axes[1].twinx()
    ax2.grid(False)
    xs = [row["epoch"] for row in h if "val_R10" in row]
    ax2.plot(xs, [row["val_R10"] for row in h if "val_R10" in row], marker="s", ms=3,
             color=PALETTE[1], label=f"val R@{ks[-1]}")
    axes[1].set(xlabel="epoch", ylabel="training loss", title="Task 4 training")
    ax2.set_ylabel(f"val R@{ks[-1]}")
    axes[1].legend(loc="upper right", fontsize=7)
    ax2.legend(loc="lower right", fontsize=7)
    savefig(fig, "task4_retrieval")

    out: dict = {}
    p = load_preds("task4_contrastive")
    if p is None:
        return out
    sim = p["sim"]

    # ---- similarity matrix: the diagonal should be visible --------------
    k = min(60, sim.shape[0])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(sim[:k, :k], cmap="magma", ax=ax, cbar_kws={"label": "cosine similarity"})
    ax.set(xlabel="caption index", ylabel="audio index",
           title=f"Task 4 similarity matrix (first {k} test clips)")
    savefig(fig, "task4_similarity_matrix")

    # ---- rank distribution ----------------------------------------------
    correct = sim[np.arange(sim.shape[0]), np.arange(sim.shape[0])]
    ranks = (sim > correct[:, None]).sum(1) + 1
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(ranks, bins=min(40, sim.shape[0]), color=PALETTE[0])
    ax.axvline(float(np.median(ranks)), ls="--", color=PALETTE[3],
               label=f"median rank {np.median(ranks):.0f}")
    ax.axvline(sim.shape[0] / 2, ls=":", color="grey", label="chance median")
    ax.set(xlabel="rank of the correct caption", ylabel="clips",
           title="Task 4 rank distribution (audio -> caption)")
    ax.legend(fontsize=7)
    savefig(fig, "task4_rank_distribution")

    # ---- zero-shot vs supervised ----------------------------------------
    zs = t4.get("zero_shot_tagging", {})
    sup = m.get("task3", {}).get(f"task3_{m.get('extra', {}).get('best', 'cross_attention')}")
    sup = sup or next((v for k, v in m.get("task3", {}).items()
                       if k.startswith("task3_") and isinstance(v, dict)), None)
    if zs and sup:
        rows = [["Task 4 zero-shot (no tag supervision)", zs["macro_f1"], zs["micro_f1"],
                 zs["auc_pr"]],
                [f"Task 3 supervised ({sup['mode']})", sup["test_tags"]["macro_f1"],
                 sup["test_tags"]["micro_f1"], sup["test_tags"]["auc_pr"]]]
        t1 = m.get("task1", {}).get("task1_bert_masked")
        if t1:
            rows.append(["Task 1 text-only (masked captions)", t1["test"]["macro_f1"],
                         t1["test"]["micro_f1"], t1["test"]["auc_pr"]])
        write_table("task4_zero_shot", ["Model", "Macro-F1", "Micro-F1", "AUC-PR"], rows,
                    "Zero-shot tagging from the contrastive space against the supervised "
                    "tag heads. The contrastive model never saw the tag vocabulary.")
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        ax.bar([r[0].split(" (")[0] for r in rows], [r[1] for r in rows],
               color=[PALETTE[1], PALETTE[0], PALETTE[2]][:len(rows)])
        ax.set(ylabel="macro-F1", title="Tag prediction: zero-shot vs supervised")
        ax.tick_params(axis="x", rotation=20, labelsize=7)
        savefig(fig, "task4_zero_shot_vs_supervised")

    # ---- qualitative retrieval examples (spec: 10) ----------------------
    out["retrieval_examples"] = _retrieval_examples(cfg, p, sim)
    return out


def _retrieval_examples(cfg, p: dict, sim: np.ndarray) -> list[dict]:
    n = int(cfg.dotted("eval.num_retrieval_examples", 10))
    caps = [str(c) for c in p["captions"]]
    ids = [str(i) for i in p["ids"]]
    correct = sim[np.arange(sim.shape[0]), np.arange(sim.shape[0])]
    ranks = (sim > correct[:, None]).sum(1) + 1

    # spread across the rank distribution rather than showing ten easy hits
    order = np.argsort(ranks)
    picks = np.unique(np.linspace(0, len(order) - 1, n).astype(int))
    examples, rows = [], []
    for j in picks:
        q = int(order[j])
        top = np.argsort(-sim[:, q])[:3]         # caption q queries the audio pool
        ex = {
            "query_caption": caps[q],
            "query_ytid": ids[q],
            "correct_rank_of_audio": int((sim[:, q] > sim[q, q]).sum() + 1),
            "top3_retrieved": [
                {"ytid": ids[int(i)], "score": round(float(sim[int(i), q]), 4),
                 "is_correct": bool(int(i) == q),
                 "its_caption": caps[int(i)][:160]} for i in top
            ],
        }
        examples.append(ex)
        rows.append([ex["query_caption"][:70] + "...", ex["correct_rank_of_audio"],
                     "yes" if ex["top3_retrieved"][0]["is_correct"] else "no",
                     round(ex["top3_retrieved"][0]["score"], 3)])

    d = resolve(load_config(), "retrieval_examples")
    (d / "task4_retrieval_examples.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8")
    write_table("task4_retrieval_examples",
                ["Caption query (truncated)", "Rank of correct audio", "Top-1 correct", "Top-1 score"],
                rows, f"{len(examples)} caption->audio retrievals sampled evenly across the "
                      f"rank distribution of the {sim.shape[0]}-clip test pool.",
                # Same rigid-`r` problem as task3_case_studies: the query column holds 70-odd
                # characters of caption, which set on one line ran 114pt past the right margin
                # and cut the text off. The three numeric columns are narrow in their data
                # ("7", "no", "0.6250") but wide in their headers, and between them plus the
                # eight \tabcolsep gaps they claim 259pt of the 472pt text block -- so the
                # query column gets the remaining ~213pt, taken here as 0.42\linewidth to keep
                # a margin. Every row wraps to two lines at any width that leaves room for the
                # other three, so nothing is gained by abbreviating the headers to buy space.
                colspec=r">{\raggedright\arraybackslash}p{0.42\linewidth}rrr")
    log.info("  %d retrieval examples -> results/retrieval_examples/", len(examples))
    return examples


# =========================================================================== #
# Dataset-level figures (spec section 10: dataset statistics)
# =========================================================================== #
def eval_datasets(cfg, args) -> dict:
    log.info("dataset figures")
    out: dict = {}

    # ---- graph statistics per corpus -------------------------------------
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for j, (ds, kind) in enumerate((("gtzan", "segment"), ("deam", "segment"),
                                    ("musiccaps", "segment"), ("gtzan", "chord"))):
        try:
            gs = load_cached_graphs(cfg, ds, kind)
        except FileNotFoundError:
            continue
        nodes = np.array([int(g.num_nodes) for g in gs])
        edges = np.array([int(g.edge_index.shape[1]) for g in gs])
        deg = edges / np.maximum(nodes, 1)
        rows.append([f"{ds} ({kind})", len(gs), float(nodes.mean()), float(edges.mean()),
                     float(deg.mean()), int(gs[0].x.shape[1])])
        if j < 3:
            axes[0].hist(nodes, bins=20, alpha=0.6, label=f"{ds}/{kind}")
            axes[1].hist(edges, bins=20, alpha=0.6, label=f"{ds}/{kind}")
            axes[2].hist(deg, bins=20, alpha=0.6, label=f"{ds}/{kind}")
    for ax, t in zip(axes, ("nodes per graph", "edges per graph", "average degree")):
        ax.set(xlabel=t, ylabel="graphs")
        ax.legend(fontsize=7)
    fig.suptitle("Music structure graph statistics", y=1.03)
    savefig(fig, "dataset_graph_statistics")
    write_table("dataset_graphs",
                ["Corpus (graph)", "Graphs", "Mean nodes", "Mean edges", "Mean degree", "Node dim"],
                rows, "Cached graph statistics per corpus.")
    out["graph_statistics"] = rows

    # ---- MusicCaps tag distribution + DEAM valence/arousal ---------------
    try:
        from src.data_loading import load_deam, load_musiccaps
        df, vocab = load_musiccaps(cfg)
        y = np.stack(df["y"].to_list())
        counts = y.sum(0)
        order = np.argsort(-counts)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        axes[0].bar(range(len(order)), counts[order], color=PALETTE[0])
        axes[0].set_xticks(range(0, len(order), 2))
        axes[0].set_xticklabels([vocab.tags[i] for i in order[::2]], rotation=75,
                                ha="right", fontsize=6)
        axes[0].set(ylabel="clips", title=f"MusicCaps top-{len(order)} tag frequency")
        axes[1].hist(y.sum(1), bins=range(1, int(y.sum(1).max()) + 2), color=PALETTE[1])
        axes[1].set(xlabel="tags per clip", ylabel="clips", title="Label cardinality")
        savefig(fig, "dataset_musiccaps_tags")
        out["musiccaps"] = {
            "n_clips": int(len(df)), "n_tags": int(y.shape[1]),
            "tags_per_clip_mean": float(y.sum(1).mean()),
            "rarest_tag_support": int(counts.min()), "commonest_tag_support": int(counts.max()),
            "audio_recovered": int(df["audio_exists"].sum()),
            "audio_recovery_rate": float(df["audio_exists"].mean()),
        }

        deam = load_deam(cfg)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for split, color in (("train", PALETTE[0]), ("val", PALETTE[1]), ("test", PALETTE[2])):
            s = deam["split"] == split
            axes[0].scatter(deam.loc[s, "valence"], deam.loc[s, "arousal"], s=10, alpha=0.6,
                            label=split, color=color)
        axes[0].axhline(5, ls=":", color="grey", lw=1)
        axes[0].axvline(5, ls=":", color="grey", lw=1)
        axes[0].set(xlabel="valence (1-9)", ylabel="arousal (1-9)",
                    title="DEAM annotations by split")
        axes[0].legend(fontsize=7)
        axes[1].hist(deam["valence"], bins=30, alpha=0.6, label="valence", color=PALETTE[0])
        axes[1].hist(deam["arousal"], bins=30, alpha=0.6, label="arousal", color=PALETTE[1])
        axes[1].set(xlabel="rating (1-9)", ylabel="songs", title="Marginal distributions")
        axes[1].legend(fontsize=7)
        savefig(fig, "dataset_deam_emotion")
        out["deam"] = {
            "n_songs": int(len(deam)),
            "valence_mean": float(deam["valence"].mean()), "valence_sd": float(deam["valence"].std()),
            "arousal_mean": float(deam["arousal"].mean()), "arousal_sd": float(deam["arousal"].std()),
            "valence_arousal_correlation": float(deam["valence"].corr(deam["arousal"])),
        }
    except (FileNotFoundError, KeyError) as exc:
        log.warning("dataset figures partially skipped: %s", exc)

    # ---- one worked example graph ----------------------------------------
    try:
        gs = load_cached_graphs(cfg, "gtzan", "segment")
        g = gs[0]
        A = np.zeros((int(g.num_nodes), int(g.num_nodes)))
        ei = g.edge_index.numpy()
        kinds = g.edge_attr.numpy() if g.edge_attr is not None else None
        for e in range(ei.shape[1]):
            A[ei[0, e], ei[1, e]] = 2.0 if (kinds is not None and kinds[e, 2] > 0.5) else 1.0
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        sns.heatmap(A, cmap=sns.color_palette(["white", PALETTE[0], PALETTE[3]], as_cmap=True),
                    cbar=False, ax=axes[0], square=True, linewidths=0.2, linecolor="#eee")
        axes[0].set(xlabel="segment j", ylabel="segment i",
                    title=f"Adjacency of {g.track_id}\n(blue = temporal, red = similarity)")
        sns.heatmap(g.x.numpy()[:, :64], cmap="viridis", ax=axes[1], cbar_kws={"label": "z-score"})
        axes[1].set(xlabel="first 64 of %d node features" % g.x.shape[1], ylabel="segment i",
                    title="Node feature matrix X")
        savefig(fig, "example_graph_structure")
    except (FileNotFoundError, IndexError) as exc:
        log.warning("example graph figure skipped: %s", exc)

    return out


# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["1", "2", "3", "4", "data", "all"])
    ap.add_argument("--skip-tsne", action="store_true", help="skip the slow manifold plots")
    args = ap.parse_args()

    cfg = load_config()
    setup_logging(logging.INFO, resolve(cfg, "results") / "evaluate.log")
    set_seed(cfg.get("seed", 425))
    m = load_metrics(cfg)
    if not m:
        raise SystemExit("results/metrics.json is empty -- run scripts/train.py first")
    log.info("metrics.json contains: %s", ", ".join(sorted(m)))

    extra: dict = {}
    todo = ["data", "1", "2", "3", "4"] if args.task == "all" else [args.task]
    for t in todo:
        try:
            if t == "data":
                extra["datasets"] = eval_datasets(cfg, args)
            elif t == "1":
                extra["task1_analysis"] = eval_task1(cfg, m, args)
            elif t == "2":
                extra["task2_analysis"] = eval_task2(cfg, m, args)
            elif t == "3":
                extra["task3_analysis"] = eval_task3(cfg, m, args)
            elif t == "4":
                extra["task4_analysis"] = eval_task4(cfg, m, args)
        except Exception as exc:                                        # noqa: BLE001
            log.exception("evaluation of %r failed: %s", t, exc)

    save_metrics(cfg, "analysis", {k: v for k, v in extra.items() if v})
    n_fig = len(list(plots_dir().glob("*.png")))
    n_tab = len(list(tables_dir().glob("*.md")))
    log.info("done: %d figures in results/plots/, %d tables in results/tables/", n_fig, n_tab)


if __name__ == "__main__":
    main()
