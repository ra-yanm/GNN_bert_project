"""Train every task in the project. One entry point, four tasks, shared plumbing.

    python scripts/train.py --task 1        # BERT caption -> tags (+ naive/masked, B0)
    python scripts/train.py --task 2        # GNN genre classification (+ B1/B2/B4, ablations)
    python scripts/train.py --task 3        # GNN-BERT fusion (4 fusion ablations + emotion)
    python scripts/train.py --task 4        # contrastive retrieval (+ zero-shot tagging)
    python scripts/train.py --task all

Every run appends to ``results/metrics.json`` under its own key and writes raw
predictions to ``results/preds/<run>.npz``, so ``scripts/evaluate.py`` can rebuild
every table and figure without retraining anything.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bert_encoder import (                                                    # noqa: E402
    BertTagClassifier, MaskedBCELoss, TextTagDataset, compute_pos_weight, truncation_report,
)
from src.contrastive import (                                                     # noqa: E402
    ContrastiveGNNBert, InfoNCELoss, encode_split, similarity_matrix, zero_shot_tags,
)
from src.data_loading import (                                                    # noqa: E402
    GTZAN_GENRES, lexical_match_baseline, load_deam, load_musiccaps,
)
from src.fusion_model import GNNBertFusion, MultiTaskLoss                         # noqa: E402
from src.gnn_model import FeatureMLP, GNNClassifier, MelCNN                       # noqa: E402
from src.graph_dataset import (                                                   # noqa: E402
    GraphDataset, GraphTextDataset, MelDataset, PairedGraphTextDataset, ablate_edges,
    collate_graph_text, collate_graphs, collate_paired, filter_split,
    load_cached_graphs, load_cached_mels,
)
from src.metrics import (                                                         # noqa: E402
    majority_baseline, multilabel_metrics, prior_baseline, random_baseline,
    recall_at_k, regression_metrics, single_label_metrics, tune_threshold,
)
from src.trainer import (                                                         # noqa: E402
    History, build_optimizer, clip_and_step, describe_model, trim_text_batch,
)
from src.utils import (                                                           # noqa: E402
    EarlyStopper, ROOT, get_device, human_time, load_config, resolve, save_metrics,
    set_seed, setup_logging,
)

log = logging.getLogger("train")


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def preds_dir() -> Path:
    d = ROOT / "results" / "preds"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt_dir() -> Path:
    d = ROOT / "results" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_preds(run: str, **arrays) -> Path:
    p = preds_dir() / f"{run}.npz"
    np.savez_compressed(p, **arrays)
    return p


def record(cfg, run: str, payload: dict) -> None:
    save_metrics(cfg, run, payload)
    log.info("[%s] recorded -> results/metrics.json", run)


def canonical_graph(g: Data) -> Data:
    """Rebuild a graph with exactly one key set: structure plus ``track_id``.

    ``Batch.from_data_list`` takes its key list from the *first* element of the
    batch, so mixing MusicCaps graphs (which carry ``y_multi``) with DEAM graphs
    (which carry ``valence_z``) raises a KeyError on whichever corpus does not
    happen to come first. Task 3 supplies its targets through the dataset tensors,
    so the graph itself only needs its topology.
    """
    d = Data(x=g.x, edge_index=g.edge_index)
    for k in ("edge_attr", "edge_weight"):
        v = getattr(g, k, None)
        if v is not None:
            setattr(d, k, v)
    d.track_id = str(g.track_id)
    return d


# =========================================================================== #
# TASK 1 -- BERT caption -> multi-label tags   (spec section 4.1)
# =========================================================================== #
def train_task1(cfg, args) -> dict:
    """t = BERT_CLS(X); y_k = sigmoid(w_k^T t + b_k); L = mean-over-tags BCE.

    Run twice: once on the raw caption (``naive``) and once with every tag surface
    form stripped (``masked``). The gap between them is the size of the lexical
    shortcut, and baseline B0 measures that shortcut directly.
    """
    tc = cfg.dotted("train.task1_bert")
    device = get_device(cfg)
    df, vocab = load_musiccaps(cfg, num_tags=int(tc["num_tags"]))
    tag_names = list(vocab.tags)

    # ---- Baseline B0: pure string matching ------------------------------
    out: dict = {}
    for col in ("caption", "caption_masked"):
        yt, yp = lexical_match_baseline(df[df.split == "test"], vocab, text_col=col)
        m = multilabel_metrics(yt, yp, threshold=0.5, tag_names=tag_names)
        out[f"B0_lexical_{col}"] = {k: v for k, v in m.items() if k != "per_tag"}
        log.info("B0 lexical match on %-15s: macro-F1 %.4f  micro-F1 %.4f",
                 col, m["macro_f1"], m["micro_f1"])

    # ---- Baselines B1: random and training-prior ------------------------
    y_tr = np.stack(df[df.split == "train"]["y"].to_list())
    y_te = np.stack(df[df.split == "test"]["y"].to_list())
    out["B1_random"] = {k: v for k, v in multilabel_metrics(
        y_te, random_baseline(y_te), threshold=0.5).items() if k != "per_tag"}
    out["B1_prior"] = {k: v for k, v in multilabel_metrics(
        y_te, prior_baseline(y_tr, len(y_te)), threshold=0.3).items() if k != "per_tag"}
    log.info("B1 random macro-F1 %.4f | B1 prior macro-F1 %.4f / AUC-PR %.4f",
             out["B1_random"]["macro_f1"], out["B1_prior"]["macro_f1"], out["B1_prior"]["auc_pr"])

    variants = ["masked", "naive"] if args.task1_variants == "both" else [args.task1_variants]
    for variant in variants:
        text_col = "caption" if variant == "naive" else "caption_masked"
        run = f"task1_bert_{variant}"
        log.info("=" * 78)
        log.info("TASK 1 [%s] -- text column %r", variant, text_col)
        set_seed(cfg.get("seed", 425))

        splits = {s: TextTagDataset(df[df.split == s].reset_index(drop=True), cfg, text_col)
                  for s in ("train", "val", "test")}
        log.info("truncation: %s", json.dumps(truncation_report(splits["train"], cfg)))

        bs = int(tc["batch_size"])
        loaders = {
            s: DataLoader(d, batch_size=bs, shuffle=(s == "train"),
                          num_workers=0, drop_last=(s == "train"))
            for s, d in splits.items()
        }

        model = BertTagClassifier(cfg, num_tags=len(tag_names)).to(device)
        info = describe_model(model, run)
        crit = MaskedBCELoss(compute_pos_weight(splits["train"].y.numpy())).to(device)

        epochs = args.epochs or int(tc["epochs"])
        steps = max(1, len(loaders["train"])) * epochs
        opt, sched = build_optimizer(
            model.param_groups(float(tc["lr_bert"]), float(tc["lr_head"]), float(tc["weight_decay"])),
            total_steps=steps, warmup_ratio=float(tc["warmup_ratio"]), schedule="cosine")

        hist, stopper = History(), EarlyStopper(patience=3, mode="max")
        for ep in range(1, epochs + 1):
            model.train()
            tot, nb = 0.0, 0
            for batch in loaders["train"]:
                ids, am = trim_text_batch(batch["input_ids"], batch["attention_mask"])
                loss = crit(model(ids.to(device), am.to(device)), batch["y"].to(device))
                loss.backward()
                clip_and_step(model, opt, sched, float(tc["max_grad_norm"]))
                tot += float(loss.detach()); nb += 1

            vy, vp = _infer_text(model, loaders["val"], device)
            thr, vf1 = tune_threshold(vy, vp)
            hist.log_epoch(ep, train_loss=tot / max(nb, 1), val_macro_f1=vf1, val_threshold=thr,
                           lr=opt.param_groups[0]["lr"])
            log.info(hist.summary_line(ep, ("train_loss", "val_macro_f1", "val_threshold")))
            if stopper.step(vf1, model, ep):
                log.info("early stop at epoch %d (best %d)", ep, stopper.best_epoch)
                break
        stopper.restore(model)

        # threshold is re-tuned on val at the restored checkpoint, then frozen
        vy, vp = _infer_text(model, loaders["val"], device)
        thr, _ = tune_threshold(vy, vp)
        ty, tp = _infer_text(model, loaders["test"], device)
        test = multilabel_metrics(ty, tp, threshold=thr, tag_names=tag_names, topk_report=15)
        val = multilabel_metrics(vy, vp, threshold=thr, tag_names=None)
        log.info("TASK 1 [%s] TEST macro-F1 %.4f | micro-F1 %.4f | AUC-PR %.4f (thr %.3f)",
                 variant, test["macro_f1"], test["micro_f1"], test["auc_pr"], thr)

        torch.save(model.state_dict(), ckpt_dir() / f"{run}.pt")
        save_preds(run, y_true=ty, y_prob=tp, tag_names=np.array(tag_names))
        out[run] = {
            "variant": variant, "text_col": text_col, "model": info,
            "threshold_from_val": thr, "best_epoch": stopper.best_epoch,
            "epochs_run": len(hist.rows), "history": hist.to_list(),
            "val": {k: v for k, v in val.items() if k != "per_tag"}, "test": test,
            "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        }

    if "task1_bert_naive" in out and "task1_bert_masked" in out:
        gap = out["task1_bert_naive"]["test"]["macro_f1"] - out["task1_bert_masked"]["test"]["macro_f1"]
        out["lexical_shortcut_macro_f1_gap"] = float(gap)
        log.info("lexical shortcut worth %.4f macro-F1 (naive - masked)", gap)

    record(cfg, "task1", out)
    return out


@torch.no_grad()
def _infer_text(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        ids, am = trim_text_batch(batch["input_ids"], batch["attention_mask"])
        logits = model(ids.to(device), am.to(device))
        ps.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["y"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


# =========================================================================== #
# TASK 2 -- GNN on music structure graphs   (spec section 4.2)
# =========================================================================== #
def train_task2(cfg, args) -> dict:
    """Genre classification from graphs, against the spec's baselines B1/B2/B4
    and three ablations (conv type, graph construction, edge class)."""
    device = get_device(cfg)
    out: dict = {}

    graphs = load_cached_graphs(cfg, "gtzan", "segment")
    sp = {s: filter_split(graphs, s) for s in ("train", "val", "test")}
    node_dim = int(graphs[0].x.shape[1])
    log.info("Task 2: %d/%d/%d graphs, node_dim=%d",
             len(sp["train"]), len(sp["val"]), len(sp["test"]), node_dim)

    y_train = np.array([int(g.y.item()) for g in sp["train"]])
    y_test = np.array([int(g.y.item()) for g in sp["test"]])

    # ---- B1: majority class ---------------------------------------------
    out["B1_majority"] = single_label_metrics(
        y_test, majority_baseline(y_train, len(y_test), len(GTZAN_GENRES)))
    out["B1_majority"].pop("per_class", None)
    log.info("B1 majority: acc %.4f macro-F1 %.4f",
             out["B1_majority"]["accuracy"], out["B1_majority"]["macro_f1"])

    # ---- graph models + B4 ----------------------------------------------
    configs = [
        ("gnn_sage_segment", dict(conv="sage", kind="segment", edges="all")),
        ("gnn_gat_segment", dict(conv="gat", kind="segment", edges="all")),
        ("gnn_sage_chord", dict(conv="sage", kind="chord", edges="all")),
        ("gnn_sage_temporal_only", dict(conv="sage", kind="segment", edges="temporal")),
        ("gnn_sage_similarity_only", dict(conv="sage", kind="segment", edges="similarity")),
        ("gnn_sage_no_edges", dict(conv="sage", kind="segment", edges="none")),
        ("B4_feature_mlp", dict(conv=None, kind="segment", edges="all")),
    ]
    if args.quick:
        configs = [configs[0], configs[-1]]

    for run, spec in configs:
        set_seed(cfg.get("seed", 425))
        gs = (load_cached_graphs(cfg, "gtzan", "chord") if spec["kind"] == "chord" else graphs)
        s = {k: filter_split(gs, k) for k in ("train", "val", "test")}
        if spec["edges"] != "all":
            s = {k: ablate_edges(v, spec["edges"]) for k, v in s.items()}
        nd = int(s["train"][0].x.shape[1])

        if spec["conv"] is None:
            model = FeatureMLP(cfg, nd, len(GTZAN_GENRES)).to(device)
        else:
            cfg["model"]["gnn"]["conv"] = spec["conv"]
            model = GNNClassifier(cfg, nd, len(GTZAN_GENRES)).to(device)
        log.info("-" * 78)
        log.info("TASK 2 [%s] conv=%s graph=%s edges=%s node_dim=%d",
                 run, spec["conv"], spec["kind"], spec["edges"], nd)
        res = _fit_graph_classifier(cfg, args, model, s, device, run)
        out[run] = res | {"config": spec}

    # ---- B2: CNN on mel-spectrogram --------------------------------------
    try:
        mels = load_cached_mels(cfg, "gtzan", "segment")
        set_seed(cfg.get("seed", 425))
        tr = MelDataset(mels, sp["train"])
        stats = tr.stats
        ds = {"train": tr, "val": MelDataset(mels, sp["val"], stats),
              "test": MelDataset(mels, sp["test"], stats)}
        model = MelCNN(cfg, len(GTZAN_GENRES)).to(device)
        log.info("-" * 78)
        log.info("TASK 2 [B2_mel_cnn] mel patch %s", tuple(ds["train"][0][0].shape))
        out["B2_mel_cnn"] = _fit_mel_cnn(cfg, args, model, ds, device, "B2_mel_cnn")
    except FileNotFoundError as exc:
        log.warning("skipping B2 CNN baseline: %s", exc)
        out["B2_mel_cnn"] = {"skipped": str(exc)}

    cfg["model"]["gnn"]["conv"] = "sage"        # restore for later tasks
    record(cfg, "task2", out)
    return out


def _fit_graph_classifier(cfg, args, model, splits, device, run) -> dict:
    tc = cfg.dotted("train.task2_gnn")
    bs = int(tc["batch_size"])
    loaders = {s: DataLoader(GraphDataset(v), batch_size=bs, shuffle=(s == "train"),
                             collate_fn=collate_graphs, num_workers=0)
               for s, v in splits.items()}
    info = describe_model(model, run)
    epochs = args.epochs or int(tc["epochs"])
    opt, sched = build_optimizer(model, lr=float(tc["lr"]), weight_decay=float(tc["weight_decay"]),
                                total_steps=max(1, len(loaders["train"])) * epochs,
                                warmup_ratio=0.05, schedule=tc.get("scheduler", "cosine"))
    hist, stopper = History(), EarlyStopper(patience=int(tc["patience"]), mode="max")

    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            loss = F.cross_entropy(model(batch), batch.y.view(-1))
            loss.backward()
            clip_and_step(model, opt, sched, 5.0)
            tot += float(loss.detach()); nb += 1
        vy, vl = _infer_graph(model, loaders["val"], device)
        vm = single_label_metrics(vy, vl)
        hist.log_epoch(ep, train_loss=tot / max(nb, 1), val_macro_f1=vm["macro_f1"],
                       val_acc=vm["accuracy"])
        if ep % 5 == 0 or ep == 1:
            log.info(hist.summary_line(ep, ("train_loss", "val_macro_f1", "val_acc")))
        if stopper.step(vm["macro_f1"], model, ep):
            log.info("early stop at epoch %d (best epoch %d, val macro-F1 %.4f)",
                     ep, stopper.best_epoch, stopper.best)
            break
    stopper.restore(model)

    ty, tl = _infer_graph(model, loaders["test"], device)
    test = single_label_metrics(ty, tl, GTZAN_GENRES)
    vy, vl = _infer_graph(model, loaders["val"], device)
    log.info("TASK 2 [%s] TEST acc %.4f | macro-F1 %.4f | AUC-PR %.4f",
             run, test["accuracy"], test["macro_f1"], test["auc_pr"])

    torch.save(model.state_dict(), ckpt_dir() / f"task2_{run}.pt")
    save_preds(f"task2_{run}", y_true=ty, logits=tl, class_names=np.array(GTZAN_GENRES))
    return {
        "model": info, "best_epoch": stopper.best_epoch, "epochs_run": len(hist.rows),
        "history": hist.to_list(), "test": test,
        "val": {k: v for k, v in single_label_metrics(vy, vl).items()
                if k not in ("confusion_matrix", "per_class")},
    }


@torch.no_grad()
def _infer_graph(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ls = [], []
    for batch in loader:
        batch = batch.to(device)
        ls.append(model(batch).cpu().numpy())
        ys.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ls)


def _fit_mel_cnn(cfg, args, model, ds, device, run) -> dict:
    tc = cfg.dotted("train.task2_gnn")
    bs = 16                          # mel patches are 128x640 floats; 32 thrashes RAM
    loaders = {s: DataLoader(v, batch_size=bs, shuffle=(s == "train"), num_workers=0)
               for s, v in ds.items()}
    info = describe_model(model, run)
    epochs = args.epochs or min(30, int(tc["epochs"]))     # CNN epochs cost ~20x a GNN epoch
    opt, sched = build_optimizer(model, lr=1e-3, weight_decay=float(tc["weight_decay"]),
                                 total_steps=max(1, len(loaders["train"])) * epochs,
                                 warmup_ratio=0.05, schedule="cosine")
    hist, stopper = History(), EarlyStopper(patience=8, mode="max")

    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y in loaders["train"]:
            loss = F.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward()
            clip_and_step(model, opt, sched, 5.0)
            tot += float(loss.detach()); nb += 1
        vy, vl = _infer_mel(model, loaders["val"], device)
        vm = single_label_metrics(vy, vl)
        hist.log_epoch(ep, train_loss=tot / max(nb, 1), val_macro_f1=vm["macro_f1"],
                       val_acc=vm["accuracy"])
        log.info(hist.summary_line(ep, ("train_loss", "val_macro_f1", "val_acc")))
        if stopper.step(vm["macro_f1"], model, ep):
            log.info("early stop at epoch %d (best %d)", ep, stopper.best_epoch)
            break
    stopper.restore(model)

    ty, tl = _infer_mel(model, loaders["test"], device)
    test = single_label_metrics(ty, tl, GTZAN_GENRES)
    log.info("TASK 2 [%s] TEST acc %.4f | macro-F1 %.4f", run, test["accuracy"], test["macro_f1"])
    torch.save(model.state_dict(), ckpt_dir() / f"task2_{run}.pt")
    save_preds(f"task2_{run}", y_true=ty, logits=tl, class_names=np.array(GTZAN_GENRES))
    return {"model": info, "best_epoch": stopper.best_epoch, "epochs_run": len(hist.rows),
            "history": hist.to_list(), "test": test}


@torch.no_grad()
def _infer_mel(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ls = [], []
    for x, y in loader:
        ls.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ls)


# =========================================================================== #
# TASK 3 -- GNN-BERT fusion for multi-context understanding  (spec section 4.3)
# =========================================================================== #
def _build_task3_data(cfg, tokenizer, max_len: int):
    """Union of MusicCaps (graph + caption -> 50 tags) and DEAM (graph +
    release metadata -> valence/arousal).

    The two corpora annotate complementary things, which is precisely the
    "multi-context" setting Task 3 describes: one model, two supervision signals,
    each present on only part of the data. Per-sample masks keep a missing label
    from being read as a negative.

    MusicCaps text is the *masked* caption. Using the raw caption would let the
    text branch string-match the tag it is being asked to predict, and every
    fusion comparison would then be measuring that shortcut rather than fusion.
    """
    K = int(cfg.dotted("train.task1_bert.num_tags", 50))
    mc_df, vocab = load_musiccaps(cfg, num_tags=K)
    mc_graphs = load_cached_graphs(cfg, "musiccaps", "segment")
    mc_text = dict(zip(mc_df["ytid"], mc_df["caption_masked"]))
    mc_y = {t: y for t, y in zip(mc_df["ytid"], mc_df["y"])}

    deam_graphs = load_cached_graphs(cfg, "deam", "segment")
    deam_df = load_deam(cfg)
    deam_text = {f"deam_{int(s)}": t for s, t in zip(deam_df["song_id"], deam_df["text"])}
    v_stats = (float(deam_df.attrs["valence_mu"]), float(deam_df.attrs["valence_sd"]))
    a_stats = (float(deam_df.attrs["arousal_mu"]), float(deam_df.attrs["arousal_sd"]))

    per_split: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        G, T, Y, YM, V, A = [], [], [], [], [], []
        for g in filter_split(mc_graphs, split):
            tid = str(g.track_id)
            if tid not in mc_text:
                continue
            G.append(canonical_graph(g)); T.append(str(mc_text[tid]))
            Y.append(mc_y[tid]); YM.append(True)
            V.append(np.nan); A.append(np.nan)
        for g in filter_split(deam_graphs, split):
            tid = str(g.track_id)
            G.append(canonical_graph(g)); T.append(deam_text.get(tid, "A music track."))
            Y.append(np.zeros(K, dtype=np.float32)); YM.append(False)
            V.append(float(g.valence_z.item())); A.append(float(g.arousal_z.item()))
        per_split[split] = dict(
            ds=GraphTextDataset(G, T, np.stack(Y), tokenizer, max_len,
                                valence=np.array(V, dtype=np.float64),
                                arousal=np.array(A, dtype=np.float64),
                                y_mask=np.array(YM, bool)),
            n_tagged=int(sum(YM)), n_emotion=int(len(YM) - sum(YM)),
        )
        log.info("Task 3 %-5s: %d rows (%d tagged MusicCaps / %d emotion DEAM)",
                 split, len(G), sum(YM), len(YM) - sum(YM))
    return per_split, list(vocab.tags), v_stats, a_stats


def train_task3(cfg, args) -> dict:
    tc = cfg.dotted("train.task3_fusion")
    device = get_device(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.text["model_name"])
    max_len = int(cfg.text["max_length"])

    data, tag_names, v_stats, a_stats = _build_task3_data(cfg, tokenizer, max_len)
    node_dim = int(data["train"]["ds"].graphs[0].x.shape[1])
    modes = list(tc["ablations"]) if not args.quick else ["cross_attention"]
    out: dict = {"tag_names": tag_names, "node_dim": node_dim,
                 "valence_scale": v_stats, "arousal_scale": a_stats,
                 "split_sizes": {k: {"n": len(v["ds"]), "n_tagged": v["n_tagged"],
                                     "n_emotion": v["n_emotion"]} for k, v in data.items()}}

    bs = int(tc["batch_size"])
    loaders = {s: DataLoader(v["ds"], batch_size=bs, shuffle=(s == "train"),
                             collate_fn=collate_graph_text, num_workers=0,
                             drop_last=(s == "train"))
               for s, v in data.items()}

    # pos_weight from the tagged training rows only
    ytr = data["train"]["ds"].y.numpy()[data["train"]["ds"].y_mask.numpy()]
    pw = compute_pos_weight(ytr)

    for mode in modes:
        run = f"task3_{mode}"
        log.info("=" * 78)
        log.info("TASK 3 [%s]", mode)
        set_seed(cfg.get("seed", 425))
        model = GNNBertFusion(cfg, node_dim, len(tag_names), mode=mode, with_emotion=True).to(device)
        info = describe_model(model, run)
        crit = MultiTaskLoss(float(tc["alpha_valence"]), float(tc["beta_arousal"]), pw).to(device)

        epochs = args.epochs or int(tc["epochs"])
        opt, sched = build_optimizer(
            model.param_groups(float(tc["lr_bert"]), float(tc["lr_head"]), float(tc["weight_decay"])),
            total_steps=max(1, len(loaders["train"])) * epochs, warmup_ratio=0.1, schedule="cosine")
        # selection on total val loss: it is the only scalar that weighs the tag
        # and emotion terms with the same alpha/beta the model is trained under
        hist, stopper = History(), EarlyStopper(patience=int(tc["patience"]), mode="min")

        for ep in range(1, epochs + 1):
            model.train()
            tot, nb = 0.0, 0
            for batch in loaders["train"]:
                b = _to_device(batch, device, mode)
                loss, _ = crit(model(b), b)
                loss.backward()
                clip_and_step(model, opt, sched, 1.0)
                tot += float(loss.detach()); nb += 1
            vl, vres = _infer_fusion(model, loaders["val"], device, crit, mode)
            vm = multilabel_metrics(vres["y"][vres["y_mask"]], vres["p"][vres["y_mask"]],
                                    threshold=0.3) if vres["y_mask"].any() else {"macro_f1": 0.0}
            hist.log_epoch(ep, train_loss=tot / max(nb, 1), val_loss=vl,
                           val_macro_f1=vm["macro_f1"])
            log.info(hist.summary_line(ep, ("train_loss", "val_loss", "val_macro_f1")))
            if stopper.step(vl, model, ep):
                log.info("early stop at epoch %d (best %d)", ep, stopper.best_epoch)
                break
        stopper.restore(model)

        _, vres = _infer_fusion(model, loaders["val"], device, crit, mode)
        thr, _ = tune_threshold(vres["y"][vres["y_mask"]], vres["p"][vres["y_mask"]])
        _, tres = _infer_fusion(model, loaders["test"], device, crit, mode)

        m = tres["y_mask"]
        test_tags = multilabel_metrics(tres["y"][m], tres["p"][m], threshold=thr,
                                       tag_names=tag_names, topk_report=15)
        emo: dict = {}
        if tres["v_mask"].any():
            emo |= regression_metrics(tres["v"][tres["v_mask"]], tres["v_hat"][tres["v_mask"]],
                                      "valence", scale=v_stats)
        if tres["a_mask"].any():
            emo |= regression_metrics(tres["a"][tres["a_mask"]], tres["a_hat"][tres["a_mask"]],
                                      "arousal", scale=a_stats)
        log.info("TASK 3 [%s] TEST tags macro-F1 %.4f AUC-PR %.4f | valence MAE %.3f R2 %.3f "
                 "| arousal MAE %.3f R2 %.3f", mode, test_tags["macro_f1"], test_tags["auc_pr"],
                 emo.get("mae_valence", float("nan")), emo.get("r2_valence", float("nan")),
                 emo.get("mae_arousal", float("nan")), emo.get("r2_arousal", float("nan")))

        torch.save(model.state_dict(), ckpt_dir() / f"{run}.pt")
        save_preds(run, y_true=tres["y"], y_prob=tres["p"], y_mask=tres["y_mask"],
                   valence=tres["v"], valence_pred=tres["v_hat"], valence_mask=tres["v_mask"],
                   arousal=tres["a"], arousal_pred=tres["a_hat"], arousal_mask=tres["a_mask"],
                   z=tres["z"], tag_names=np.array(tag_names))
        out[run] = {
            "mode": mode, "model": info, "threshold_from_val": thr,
            "best_epoch": stopper.best_epoch, "epochs_run": len(hist.rows),
            "history": hist.to_list(), "test_tags": test_tags, "test_emotion": emo,
        }

    record(cfg, "task3", out)
    return out


def _to_device(batch: dict, device, mode: str) -> dict:
    b = dict(batch)
    if mode != "gnn_only":
        ids, am = trim_text_batch(batch["input_ids"], batch["attention_mask"])
        b["input_ids"], b["attention_mask"] = ids.to(device), am.to(device)
    if mode != "bert_only":
        b["graph"] = batch["graph"].to(device)
    for k in ("y", "y_mask", "valence", "arousal", "valence_mask", "arousal_mask"):
        b[k] = batch[k].to(device)
    return b


@torch.no_grad()
def _infer_fusion(model, loader, device, crit, mode) -> tuple[float, dict]:
    model.eval()
    acc: dict[str, list] = {k: [] for k in
                            ("y", "p", "y_mask", "v", "v_hat", "v_mask", "a", "a_hat", "a_mask", "z")}
    tot, nb = 0.0, 0
    for batch in loader:
        b = _to_device(batch, device, mode)
        o = model(b)
        loss, _ = crit(o, b)
        tot += float(loss); nb += 1
        acc["y"].append(b["y"].cpu().numpy())
        acc["p"].append(torch.sigmoid(o["tag_logits"]).cpu().numpy())
        acc["y_mask"].append(b["y_mask"].cpu().numpy())
        acc["v"].append(b["valence"].cpu().numpy()); acc["v_hat"].append(o["valence"].cpu().numpy())
        acc["v_mask"].append(b["valence_mask"].cpu().numpy())
        acc["a"].append(b["arousal"].cpu().numpy()); acc["a_hat"].append(o["arousal"].cpu().numpy())
        acc["a_mask"].append(b["arousal_mask"].cpu().numpy())
        acc["z"].append(o["z"].cpu().numpy())
    res = {k: np.concatenate(v) for k, v in acc.items()}
    for k in ("y_mask", "v_mask", "a_mask"):
        res[k] = res[k].astype(bool)
    return tot / max(nb, 1), res


# =========================================================================== #
# TASK 4 -- contrastive audio-text alignment  (spec section 4.4)
# =========================================================================== #
def train_task4(cfg, args) -> dict:
    tc = cfg.dotted("train.task4_contrastive")
    device = get_device(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.text["model_name"])
    max_len = int(cfg.text["max_length"])

    K = int(cfg.dotted("train.task1_bert.num_tags", 50))
    df, vocab = load_musiccaps(cfg, num_tags=K)
    graphs = load_cached_graphs(cfg, "musiccaps", "segment")
    cap = dict(zip(df["ytid"], df["caption"]))          # real captions, unmasked
    ymap = {t: y for t, y in zip(df["ytid"], df["y"])}

    data, ytrue = {}, {}
    for s in ("train", "val", "test"):
        G = [g for g in filter_split(graphs, s) if str(g.track_id) in cap]
        data[s] = PairedGraphTextDataset(G, [cap[str(g.track_id)] for g in G], tokenizer, max_len,
                                        ids=[str(g.track_id) for g in G])
        ytrue[s] = np.stack([ymap[str(g.track_id)] for g in G]) if G else np.zeros((0, K))
        log.info("Task 4 %-5s: %d (graph, caption) pairs", s, len(G))
    if len(data["train"]) < 32:
        raise SystemExit(f"only {len(data['train'])} training pairs -- "
                         "run scripts/download_musiccaps_audio.py then preprocess musiccaps")

    bs = int(tc["batch_size"])
    loaders = {s: DataLoader(v, batch_size=bs, shuffle=(s == "train"),
                             collate_fn=collate_paired, num_workers=0, drop_last=(s == "train"))
               for s, v in data.items()}

    node_dim = int(data["train"].graphs[0].x.shape[1])
    set_seed(cfg.get("seed", 425))
    model = ContrastiveGNNBert(cfg, node_dim).to(device)
    info = describe_model(model, "task4_contrastive")
    crit = InfoNCELoss(symmetric=bool(tc.get("symmetric", True)), duplicate_safe=True)

    epochs = args.epochs or int(tc["epochs"])
    opt, sched = build_optimizer(
        model.param_groups(float(tc["lr"]), float(cfg.dotted("train.task3_fusion.lr_bert")),
                           float(cfg.dotted("train.task3_fusion.weight_decay"))),
        total_steps=max(1, len(loaders["train"])) * epochs, warmup_ratio=0.1, schedule="cosine")
    ks = tuple(int(k) for k in tc["recall_at"])
    hist, stopper = History(), EarlyStopper(patience=8, mode="max")

    for ep in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loaders["train"]:
            ids, am = trim_text_batch(batch["input_ids"], batch["attention_mask"])
            b = {"graph": batch["graph"].to(device), "input_ids": ids.to(device),
                 "attention_mask": am.to(device)}
            o = model(b)
            loss, _ = crit(o["sim"], batch["group_ids"].to(device))
            loss.backward()
            clip_and_step(model, opt, sched, 1.0)
            tot += float(loss.detach()); nb += 1

        emb = encode_split(model, loaders["val"], device)
        vr = recall_at_k(similarity_matrix(emb["g"], emb["t"]), ks)
        hist.log_epoch(ep, train_loss=tot / max(nb, 1), val_R1=vr["R@1"], val_R10=vr[f"R@{ks[-1]}"],
                       logit_scale=float(model.logit_scale.detach()))
        log.info(hist.summary_line(ep, ("train_loss", "val_R1", "val_R10", "logit_scale")))
        if stopper.step(vr[f"R@{ks[-1]}"], model, ep):
            log.info("early stop at epoch %d (best %d)", ep, stopper.best_epoch)
            break
    stopper.restore(model)

    # ---- full-split retrieval on test ------------------------------------
    emb = encode_split(model, loaders["test"], device)
    sim = similarity_matrix(emb["g"], emb["t"])
    test = recall_at_k(sim, ks)
    log.info("TASK 4 TEST over %d candidates: R@1 %.4f R@5 %.4f R@10 %.4f | medr g2t %.0f t2g %.0f",
             test["n_candidates"], test["R@1"], test["R@5"], test["R@10"],
             test["medr_g2t"], test["medr_t2g"])

    # random-embedding control: R@K on a shuffled similarity matrix, which is the
    # only honest floor for a retrieval number on N candidates
    rng = np.random.default_rng(cfg.get("seed", 425))
    rand = recall_at_k(rng.standard_normal(sim.shape), ks)

    # ---- zero-shot tagging vs the Task 3 supervised model ----------------
    zs = zero_shot_tags(model, emb["g"], list(vocab.tags), tokenizer, device)
    zs_y = np.stack([ymap[i] for i in emb["ids"]])
    # cosine scores are not probabilities: rank-normalise per tag before applying
    # a threshold, so macro-F1 is not an artefact of the score scale
    zs_rank = zs.argsort(axis=0).argsort(axis=0) / max(1, zs.shape[0] - 1)
    zs_metrics = multilabel_metrics(zs_y, zs_rank, threshold=0.9, tag_names=list(vocab.tags))
    log.info("TASK 4 zero-shot tagging: macro-F1 %.4f AUC-PR %.4f (no tag supervision)",
             zs_metrics["macro_f1"], zs_metrics["auc_pr"])

    torch.save(model.state_dict(), ckpt_dir() / "task4_contrastive.pt")
    save_preds("task4_contrastive", sim=sim, g=emb["g"].numpy(), t=emb["t"].numpy(),
               ids=np.array(emb["ids"]), captions=np.array(emb["captions"], dtype=object),
               zs_scores=zs, zs_y=zs_y, tag_names=np.array(list(vocab.tags)))
    out = {
        "model": info, "best_epoch": stopper.best_epoch, "epochs_run": len(hist.rows),
        "history": hist.to_list(), "test_retrieval": test,
        "random_control_retrieval": {k: v for k, v in rand.items() if k.startswith("R@")},
        "zero_shot_tagging": zs_metrics,
        "final_logit_scale": float(model.logit_scale.detach()),
        "implied_temperature": float(1.0 / np.exp(min(float(model.logit_scale.detach()),
                                                     np.log(100.0)))),
        "symmetric_loss": bool(tc.get("symmetric", True)),
        "n_train": len(data["train"]), "n_val": len(data["val"]), "n_test": len(data["test"]),
    }
    record(cfg, "task4", out)
    return out


# =========================================================================== #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["1", "2", "3", "4", "all"])
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    ap.add_argument("--quick", action="store_true", help="smoke test: fewest models/epochs")
    ap.add_argument("--task1-variants", default="both", choices=["both", "naive", "masked"])
    args = ap.parse_args()

    cfg = load_config()
    setup_logging(logging.INFO, resolve(cfg, "results") / "train.log")
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    log.info("device=%s torch_threads=%d seed=%d", get_device(cfg), torch.get_num_threads(),
             cfg.get("seed", 425))

    tasks = ["1", "2", "3", "4"] if args.task == "all" else [args.task]
    fns = {"1": train_task1, "2": train_task2, "3": train_task3, "4": train_task4}
    t0 = time.time()
    for t in tasks:
        log.info("#" * 78)
        log.info("### TASK %s", t)
        log.info("#" * 78)
        ts = time.time()
        try:
            fns[t](cfg, args)
        except (FileNotFoundError, SystemExit) as exc:
            log.error("TASK %s could not run: %s", t, exc)
            continue
        log.info("### TASK %s finished in %s", t, human_time(time.time() - ts))
    log.info("all requested tasks done in %s", human_time(time.time() - t0))


if __name__ == "__main__":
    main()
