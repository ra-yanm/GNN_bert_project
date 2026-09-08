"""Run the trained checkpoints on an audio file that was never in any split.

Why this exists
---------------
``notebooks/demo_context.ipynb`` is the spec's end-to-end inference example, and it
does load real checkpoints -- but every demo in it reads a *preprocessed* graph out of
``data/processed`` via ``load_cached_graphs`` and then picks a test-split clip. Demo E
looks like new-song inference and is not: it holds one cached graph fixed and swaps the
caption. So the notebook cannot answer "what does this system say about a song of mine",
which is the first question anyone asks of a trained model.

Nothing is trained or fitted here. The file is featurised and turned into a segment
graph by the same two functions ``scripts/preprocess.py`` calls, so the tensors reaching
each checkpoint are distributed the way its training data was. Every prediction is
printed next to that model's measured test score, because several of these heads are
close to chance and a bare label would read as far more confident than the model is.

Three things about the outputs are worth knowing before trusting them:

* Only the first ``audio.clip_seconds`` (30 s) of the file is used. That is the window
  the models were trained on; a longer read would change the node count and the
  per-track normalisation statistics.
* Genre is GTZAN's ten classes and tags are the 50 most frequent MusicCaps aspects.
  Neither is an open vocabulary -- the model must answer from its list even when the
  right answer is not on it.
* Emotion comes back on DEAM's original 1--9 valence/arousal scale, de-standardised
  with the training mean and standard deviation stored in ``results/metrics.json``.

    python scripts/predict.py song.mp3
    python scripts/predict.py song.mp3 --caption "a mellow jazz trio with brushed drums"
    python scripts/predict.py song.mp3 --json results/predict_song.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import warnings

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

# This script's output is meant to be read by a person, and transformers prints a
# multi-line "LOAD REPORT" table about DistilBERT's unused MLM head every time a
# checkpoint loads. The unexpected keys are the pretraining vocab projector, which this
# architecture legitimately does not use, so the report is noise here -- unlike in
# train.py, where the same message is worth seeing in the log.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
import transformers                                                     # noqa: E402
from torch_geometric.data import Batch                                   # noqa: E402
from transformers import AutoTokenizer                                   # noqa: E402

# The env vars above only take effect if transformers has not been imported yet, which is
# true for the CLI and false inside the demo notebook -- section G imports this module
# after several cells have already loaded DistilBERT. Setting it through the API works
# either way. Safe to do globally here: nothing in this project relies on transformers'
# info-level logging for correctness, and train.py's own logging is separate.
transformers.utils.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

from src.audio_features import extract_features, mel_patch               # noqa: E402
from src.contrastive import ContrastiveGNNBert, zero_shot_tags           # noqa: E402
from src.data_loading import GTZAN_GENRES                                # noqa: E402
from src.fusion_model import GNNBertFusion                               # noqa: E402
from src.gnn_model import GNNClassifier, MelCNN                          # noqa: E402
from src.graph_builder import build_segment_graph                        # noqa: E402
from src.graph_dataset import (GraphTextDataset, ablate_edges,           # noqa: E402
                               collate_graph_text)
from src.trainer import trim_text_batch                                  # noqa: E402
from src.utils import ROOT, get_device, load_config, load_metrics, set_seed  # noqa: E402

# canonical_graph is imported rather than reimplemented: it defines what a graph looks
# like when it enters the Task 3/4 models, and a private copy here would silently drift
# from the version the checkpoints were trained under.
from train import canonical_graph                                        # noqa: E402

CKPT = ROOT / "results" / "checkpoints"


def _fmt(pairs, n=3):
    """Top-n `name score` pairs on one line."""
    return "  |  ".join(f"{k} {v:.3f}" for k, v in pairs[:n])


def _topk(scores, names, k):
    order = np.argsort(scores)[::-1][:k]
    return [(names[i], float(scores[i])) for i in order]


# --------------------------------------------------------------------------- #
# Front end: audio file -> segment graph, exactly as preprocess.py builds them
# --------------------------------------------------------------------------- #
def featurise(path: pathlib.Path, cfg):
    tf = extract_features(path, cfg, track_id=path.stem)
    g = build_segment_graph(tf, cfg)
    ea = g.edge_attr.numpy()
    return tf, g, {
        "nodes": int(g.num_nodes),
        "edges": int(g.edge_index.shape[1]),
        "temporal_edges": int((ea[:, 1] > 0.5).sum()),
        "similarity_edges": int((ea[:, 2] > 0.5).sum()),
        "node_dim": int(g.x.shape[1]),
        "seconds_used": round(float(tf.duration), 2),
    }


# --------------------------------------------------------------------------- #
# Task 2: genre from audio alone
# --------------------------------------------------------------------------- #
def predict_genre(cfg, M, tf, g, device, k):
    """Three Task 2 models on the same clip.

    All three are reported rather than just the best, because they disagree and that
    disagreement *is* the Task 2 result: the CNN on the raw spectrogram beats both
    graph models, and dropping the temporal edges beats keeping them.
    """
    out, t2 = {}, M.get("task2", {})
    runs = [
        # (checkpoint stem, metrics key, how to prepare the input)
        ("task2_B2_mel_cnn", "B2_mel_cnn", "mel"),
        ("task2_gnn_sage_similarity_only", "gnn_sage_similarity_only", "similarity"),
        ("task2_gnn_sage_segment", "gnn_sage_segment", "full"),
    ]
    for stem, key, kind in runs:
        p = CKPT / f"{stem}.pt"
        if not p.exists():
            continue
        if kind == "mel":
            model = MelCNN(cfg, len(GTZAN_GENRES)).to(device)
            x = torch.from_numpy(mel_patch(tf)).unsqueeze(0).to(device)
        else:
            model = GNNClassifier(cfg, int(g.x.shape[1]), len(GTZAN_GENRES)).to(device)
            # similarity_only was trained on graphs with the temporal edges deleted, so
            # it must be evaluated on the same topology or its inputs are off-manifold.
            gi = ablate_edges([g], "similarity")[0] if kind == "similarity" else g
            x = Batch.from_data_list([gi]).to(device)
        model.load_state_dict(torch.load(p, map_location=device))
        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0].cpu().numpy()
        out[key] = {
            "top": _topk(prob, list(GTZAN_GENRES), k),
            "test_macro_f1": (t2.get(key, {}).get("test") or {}).get("macro_f1"),
            "probs": {gname: round(float(v), 4) for gname, v in zip(GTZAN_GENRES, prob)},
        }
    return out


# --------------------------------------------------------------------------- #
# Task 3: tags + valence/arousal, with and without the caption
# --------------------------------------------------------------------------- #
def _fuse_once(cfg, fusion, tok, g, text, n_tags, device):
    """One (graph, caption) pair through the real Dataset/collate path.

    Going through GraphTextDataset rather than hand-building tensors is deliberate:
    padding, truncation and the trim step are all part of how the model was trained,
    and reproducing them by hand is how an inference script quietly disagrees with
    its own training code.
    """
    ds = GraphTextDataset([canonical_graph(g)], [text],
                          np.zeros((1, n_tags), dtype=np.float32),
                          tok, int(cfg.text["max_length"]))
    b = collate_graph_text([ds[0]])
    ids, am = trim_text_batch(b["input_ids"], b["attention_mask"])
    with torch.no_grad():
        return fusion({"graph": b["graph"].to(device),
                       "input_ids": ids.to(device),
                       "attention_mask": am.to(device)})


def predict_tags_emotion(cfg, M, g, device, caption, k):
    t3 = M.get("task3")
    if not t3:
        return {}
    tag_names = list(t3["tag_names"])
    vmu, vsd = t3["valence_scale"]
    amu, asd = t3["arousal_scale"]
    tok = AutoTokenizer.from_pretrained(cfg.text["model_name"])

    # gnn_only always runs: it is the only Task 3 head that needs no caption, so it is
    # what an audio file on its own can be scored with. cross_attention runs only when
    # a caption is supplied, and the pair shows what the caption is worth on this clip.
    plan = [("gnn_only", "task3_gnn_only", "")]
    if caption:
        plan.append(("cross_attention", "task3_cross_attention", caption))

    out = {}
    for mode, stem, text in plan:
        p = CKPT / f"{stem}.pt"
        if not p.exists():
            continue
        e = t3.get(stem, {})
        fusion = GNNBertFusion(cfg, int(t3["node_dim"]), len(tag_names),
                               mode=mode, with_emotion=True).to(device)
        fusion.load_state_dict(torch.load(p, map_location=device))
        fusion.eval()
        # mode="gnn_only" ignores input_ids entirely; the placeholder exists only so the
        # tokeniser has something to encode.
        o = _fuse_once(cfg, fusion, tok, g, text or "A music track.",
                       len(tag_names), device)
        prob = torch.sigmoid(o["tag_logits"])[0].cpu().numpy()
        thr = float(e.get("threshold_from_val", 0.5))
        val = float(o["valence"]) * vsd + vmu
        aro = float(o["arousal"]) * asd + amu
        out[mode] = {
            "used_caption": bool(text),
            "threshold": thr,
            "n_above_threshold": int((prob >= thr).sum()),
            "top_tags": _topk(prob, tag_names, k),
            "valence": round(val, 3),
            "arousal": round(aro, 3),
            "quadrant": ("happy/excited" if val >= 5 and aro >= 5 else
                         "angry/tense" if val < 5 and aro >= 5 else
                         "sad/depressed" if val < 5 else "calm/content"),
            "test_tag_macro_f1": (e.get("test_tags") or {}).get("macro_f1"),
            "test_mae_valence": (e.get("test_emotion") or {}).get("mae_valence"),
            "test_mae_arousal": (e.get("test_emotion") or {}).get("mae_arousal"),
            "test_r2_arousal": (e.get("test_emotion") or {}).get("r2_arousal"),
        }
        del fusion
    return out


# --------------------------------------------------------------------------- #
# Task 4: zero-shot tags from the contrastive space, no tag supervision anywhere
# --------------------------------------------------------------------------- #
def predict_zero_shot(cfg, M, g, device, k):
    p = CKPT / "task4_contrastive.pt"
    if not p.exists() or "task4" not in M or "task3" not in M:
        return {}
    tag_names = list(M["task3"]["tag_names"])
    con = ContrastiveGNNBert(cfg, int(g.x.shape[1])).to(device)
    con.load_state_dict(torch.load(p, map_location=device))
    con.eval()
    tok = AutoTokenizer.from_pretrained(cfg.text["model_name"])
    with torch.no_grad():
        emb = con.encode_graph(Batch.from_data_list([canonical_graph(g)]).to(device)).cpu()
    emb = emb / emb.norm(dim=1, keepdim=True)
    scores = zero_shot_tags(con, emb, tag_names, tok, device)[0]
    zs = M["task4"]["zero_shot_tagging"]
    return {"top_tags": _topk(scores, tag_names, k),
            "test_macro_f1": zs.get("macro_f1"), "test_auc_roc": zs.get("auc_roc")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("audio", help="path to any audio file librosa can read")
    ap.add_argument("--caption", default="",
                    help="free-text description; enables the cross-attention fusion head")
    ap.add_argument("--top", type=int, default=5, help="how many labels to print")
    ap.add_argument("--json", default="", help="also write the full result to this path")
    args = ap.parse_args()

    path = pathlib.Path(args.audio).expanduser()
    if not path.exists():
        print(f"no such file: {path}")
        return 1

    cfg = load_config(ROOT / "config.yaml")
    set_seed(cfg["seed"])
    device = get_device(cfg)
    M = load_metrics(cfg)

    try:
        tf, g, gs = featurise(path, cfg)
    except Exception as exc:                      # unreadable container, or a corrupt file
        print(f"could not read audio: {type(exc).__name__}: {exc}")
        # librosa reads through soundfile/libsndfile, which handles wav, flac, mp3, ogg
        # and aiff without any system ffmpeg. It does *not* handle m4a or aac, and
        # librosa's only fallback for those (audioread) is not a dependency here -- so
        # convert those to wav first rather than installing a decoder chain.
        print("readable without extra tools: wav, flac, mp3, ogg, aiff. "
              "m4a/aac are not supported -- convert to wav first.")
        return 1

    cap = float(cfg.audio.get("clip_seconds") or 0)
    print(f"\nfile     : {path.name}")
    print(f"audio    : {gs['seconds_used']:.1f} s analysed"
          + (f" (capped at audio.clip_seconds = {cap:.0f} s)" if cap and
             gs["seconds_used"] >= cap - 0.5 else ""))
    print(f"graph    : {gs['nodes']} nodes, {gs['edges']} edges "
          f"({gs['temporal_edges']} temporal, {gs['similarity_edges']} similarity), "
          f"{gs['node_dim']}-dim node features")

    res = {"file": str(path), "graph": gs, "caption": args.caption}

    res["genre"] = predict_genre(cfg, M, tf, g, device, args.top)
    if res["genre"]:
        print(f"\nGENRE  -- GTZAN's 10 classes only; chance is 0.100 macro-F1")
        for key, r in res["genre"].items():
            f1 = r["test_macro_f1"]
            print(f"  {key:26s} [test macro-F1 {f1:.3f}]  {_fmt(r['top'], 3)}"
                  if f1 is not None else f"  {key:26s}  {_fmt(r['top'], 3)}")

    res["task3"] = predict_tags_emotion(cfg, M, g, device, args.caption, args.top)
    for mode, r in res["task3"].items():
        src = "audio + caption" if r["used_caption"] else "audio only"
        print(f"\nTAGS + EMOTION  -- task3_{mode} ({src})")
        print(f"  valence {r['valence']:.2f} / arousal {r['arousal']:.2f} "
              f"on DEAM's 1-9 scale (5 = neutral)  ->  {r['quadrant']}")
        print(f"    test MAE {r['test_mae_valence']:.3f} valence, "
              f"{r['test_mae_arousal']:.3f} arousal; arousal R2 {r['test_r2_arousal']:.3f}")
        print(f"  top tags: {_fmt(r['top_tags'], args.top)}")
        print(f"    test tag macro-F1 {r['test_tag_macro_f1']:.3f}; "
              f"{r['n_above_threshold']}/50 tags clear the tuned threshold "
              f"{r['threshold']:.2f}")

    if not args.caption and res["task3"]:
        print("\n  (--caption enables task3_cross_attention, which scores tag macro-F1 "
              "0.306 against\n   gnn_only's 0.113 -- most of the tag signal is in the "
              "text, not the audio.)")

    res["zero_shot"] = predict_zero_shot(cfg, M, g, device, args.top)
    if res["zero_shot"]:
        z = res["zero_shot"]
        print(f"\nZERO-SHOT TAGS  -- task4_contrastive, never trained on tag labels")
        print(f"  {_fmt(z['top_tags'], args.top)}")
        print(f"    test macro-F1 {z['test_macro_f1']:.3f}, AUC-ROC "
              f"{z['test_auc_roc']:.3f} -- near chance. Read as a demo of the "
              f"mechanism,\n    not as a usable tagger.")

    if args.json:
        outp = pathlib.Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\nwrote {outp}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
