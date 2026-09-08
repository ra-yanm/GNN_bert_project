"""Compute the corpus-level statistics the report quotes, into results/corpus_stats.json.

Why this exists
---------------
The report's claim is that no number in it is typed by hand. That was true of every
*model* metric -- those come from ``results/metrics.json`` via
``tools/build_report_numbers.py`` -- and false of about a dozen *corpus* numbers:
the size of the aspect vocabulary, the tag support range, the fraction of captions
that leak their own tags, the DEAM valence/arousal correlation. Those were measured
once in a scratch session and then typed into the prose, where they could not be
re-derived and at least one of them was simply wrong (the caption-leak figure was
stated as 4,684 / 93.8%; it is 4,671 / 93.5% under the matcher the pipeline
actually uses).

Anything the report states about a corpus rather than a model is computed here and
written to JSON, so it has the same provenance as everything else.

This is a read-only analysis over ``data/raw`` and ``data/processed``. It writes one
file and no trainer touches that file, which avoids the read-modify-write race that
sharing ``metrics.json`` between concurrent jobs would create.

    python scripts/corpus_stats.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from src.data_loading import (_parse_aspects, load_deam,        # noqa: E402
                              load_gtzan, load_musiccaps)
from src.utils import ROOT, get_logger, load_config, setup_logging   # noqa: E402

log = get_logger("corpus_stats")


def musiccaps_stats(cfg) -> dict:
    """Aspect vocabulary, tag support spread, and the size of the lexical shortcut."""
    import pandas as pd

    raw = ROOT / cfg.dotted("paths.raw")
    csv = pd.read_csv(raw / "musiccaps-public.csv")
    aspects = csv["aspect_list"].apply(_parse_aspects)
    counts = Counter(a for row in aspects for a in row)

    df, vocab = load_musiccaps(cfg)
    Y = np.stack(df["y"].to_list()).astype(bool)
    support = Y.sum(0)
    captions = df["caption"].astype(str).to_list()

    # Word-boundary matching over the top-K vocabulary: the same matcher that
    # `TagVocab.mask_caption` deletes with and that baseline B0 predicts with. Using
    # anything else here would make the headline statistic describe a different
    # experiment from the one in the results table.
    P = np.stack([vocab.lexical_predict(c) for c in captions]).astype(bool)
    any_vocab = int(P.any(1).sum())
    # Stricter, and the one that actually explains B0's score: the caption contains a
    # tag the clip is *labelled* with, so string matching alone yields a true positive.
    own_tag = int((P & Y).any(1).sum())

    # Independent of the top-K truncation: does the caption echo any of its own
    # aspects, vocabulary or not? This is the ceiling on the leak.
    echo = sum(
        any(re.search(r"\b" + re.escape(a).replace(r"\ ", r"\s+") + r"\b", c, re.I)
            for a in asps)
        for c, asps in zip(captions, df["aspects"])
    )

    n = len(df)
    return {
        "nominal_rows": int(len(csv)),
        "rows_with_aspects": int((aspects.str.len() > 0).sum()),
        "distinct_aspects": len(counts),
        "rows_kept_top_k": n,
        "num_tags": len(vocab.tags),
        "tag_support_min": int(support.min()),
        "tag_support_max": int(support.max()),
        "tag_support_imbalance": float(support.max() / support.min()),
        "captions_with_any_vocab_tag": any_vocab,
        "captions_with_any_vocab_tag_frac": any_vocab / n,
        "captions_with_own_tag": own_tag,
        "captions_with_own_tag_frac": own_tag / n,
        "captions_echoing_any_own_aspect": int(echo),
        "captions_echoing_any_own_aspect_frac": echo / n,
        "matcher": "word-boundary, case-insensitive, whitespace-flexible",
    }


def gtzan_stats(cfg) -> dict:
    """Usable tracks vs what the published partition lists."""
    raw = ROOT / cfg.dotted("paths.raw")
    lines = 0
    for split in ("train", "valid", "test"):
        p = raw / "gtzan_splits" / f"{split}_filtered.txt"
        if p.exists():
            lines += sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    df = load_gtzan(cfg)
    return {
        "partition_file_tracks": lines,
        "usable_tracks": int(len(df)),
        "dropped": lines - int(len(df)) if lines else None,
        "note": "jazz.00054 is corrupt in every distribution of GTZAN",
    }


def deam_stats(cfg) -> dict:
    """The valence/arousal correlation the report warns readers about."""
    df = load_deam(cfg)
    return {
        "n_rows": int(len(df)),
        "valence_arousal_corr": float(df["valence"].corr(df["arousal"])),
        "valence_arousal_corr_train": float(
            df[df.split == "train"]["valence"].corr(df[df.split == "train"]["arousal"])),
    }


def preprocessing_stats(cfg) -> dict:
    """What the graph-building pass actually read, and how long it took.

    ``distinct_audio_files`` is deliberately not the sum of the four manifests:
    the chord graphs are a second pass over the same GTZAN audio, so counting them
    again would inflate the figure.
    """
    proc = ROOT / cfg.dotted("paths.processed")
    per: dict[str, dict] = {}
    seconds = 0.0
    graphs = 0
    for p in sorted(proc.glob("*_manifest.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        per[p.stem.replace("_manifest", "")] = {
            "num_graphs": j.get("num_graphs"),
            "elapsed_seconds": j.get("elapsed_seconds"),
        }
        seconds += float(j.get("elapsed_seconds") or 0.0)
        graphs += int(j.get("num_graphs") or 0)

    mani = ROOT / cfg.dotted("paths.raw") / "musiccaps_audio" / "manifest.json"
    mc_wav = int(json.loads(mani.read_text(encoding="utf-8"))["total_wav_on_disk"]) \
        if mani.exists() else 0
    gtzan = per.get("gtzan_segment", {}).get("num_graphs") or 0
    deam = per.get("deam_segment", {}).get("num_graphs") or 0

    return {
        "per_dataset": per,
        "graphs_built": graphs,
        "distinct_audio_files": int(gtzan + deam + mc_wav),
        "elapsed_seconds": round(seconds, 1),
    }


def main() -> int:
    setup_logging(logfile=ROOT / "results" / "corpus_stats.log")
    cfg = load_config()
    out = ROOT / "results" / "corpus_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}
    for name, fn in (("musiccaps", musiccaps_stats), ("gtzan", gtzan_stats),
                     ("deam", deam_stats), ("preprocessing", preprocessing_stats)):
        try:
            stats[name] = fn(cfg)
            log.info("%s: ok", name)
        except Exception as exc:                                     # noqa: BLE001
            # A missing corpus must not take the whole file down; the macros for
            # whatever is absent will render as ?? and say so.
            log.error("%s: FAILED (%s)", name, exc)
            stats[name] = {"error": str(exc)}

    out.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    log.info("wrote %s", out.relative_to(ROOT))

    mc = stats.get("musiccaps", {})
    if "captions_with_any_vocab_tag" in mc:
        log.info("lexical shortcut: %d/%d captions (%.1f%%) contain a vocabulary tag; "
                 "%d (%.1f%%) contain one of their own",
                 mc["captions_with_any_vocab_tag"], mc["rows_kept_top_k"],
                 100 * mc["captions_with_any_vocab_tag_frac"],
                 mc["captions_with_own_tag"], 100 * mc["captions_with_own_tag_frac"])
    return 0 if all("error" not in v for v in stats.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
