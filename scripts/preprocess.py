"""Build and cache music structure graphs from audio.

Feature extraction dominates the cost (~9.5 s/track single-threaded, mostly
``chroma_cqt``), so tracks are processed across a worker pool and the results are
cached to ``data/processed/<dataset>_graphs.pt``. Downstream training scripts
never touch audio.

Usage:
    python scripts/preprocess.py --dataset gtzan
    python scripts/preprocess.py --dataset deam
    python scripts/preprocess.py --dataset gtzan --graph-kind chord
    python scripts/preprocess.py --dataset fma --subset small     # scale-up path

Failures are collected, not fatal: a corrupt or unreadable file is logged, listed
in the manifest, and skipped, so one bad file cannot abandon a 2-hour run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio_features import extract_features, mel_patch                       # noqa: E402
from src.data_loading import (                                                   # noqa: E402
    load_deam, load_fma, load_gtzan, load_mtat, load_musiccaps,
)
from src.graph_builder import (                                                  # noqa: E402
    build_chord_graph, build_segment_graph, graph_stats, save_graph_json, save_graph_pt,
)
from src.utils import ROOT, human_time, load_config, resolve, set_seed, setup_logging  # noqa: E402

log = logging.getLogger("preprocess")

# Workers each get a fresh config; passing the Config object through pickle is
# fine but re-reading keeps the worker independent of parent mutations.
_WORKER_CFG = None
_WORKER_KIND = "segment"
_WORKER_MEL = False


def _init_worker(cfg_path: str, kind: str, save_mel: bool,
                 graph_overrides: dict | None = None) -> None:
    global _WORKER_CFG, _WORKER_KIND, _WORKER_MEL
    from src.utils import load_config as _lc
    _WORKER_CFG = _lc(cfg_path)
    # the parent's cfg mutation does not survive the process boundary, so
    # per-dataset segmentation overrides must be re-applied inside each worker
    if graph_overrides:
        _WORKER_CFG["graph"].update(graph_overrides)
    _WORKER_KIND = kind
    _WORKER_MEL = save_mel
    # librosa/numba spawn their own threads; with 12 processes that oversubscribes
    # the CPU badly. Pin each worker to one thread.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    torch.set_num_threads(1)


def _process_one(payload: dict) -> dict:
    """Worker body: audio file -> serialised graph (+ optional mel patch)."""
    try:
        cfg = _WORKER_CFG
        tf = extract_features(payload["path"], cfg, payload["track_id"])
        builder = build_chord_graph if _WORKER_KIND == "chord" else build_segment_graph
        data = builder(tf, cfg)

        # attach targets
        if payload.get("label") is not None:
            data.y = torch.tensor([int(payload["label"])], dtype=torch.long)
        if payload.get("y_multi") is not None:
            data.y_multi = torch.tensor(payload["y_multi"], dtype=torch.float32).unsqueeze(0)
        for k in ("valence", "arousal", "valence_z", "arousal_z"):
            if payload.get(k) is not None:
                setattr(data, k, torch.tensor([float(payload[k])], dtype=torch.float32))
        data.split = payload["split"]
        data.genre = payload.get("genre")
        data.duration = float(tf.duration)

        out = {"ok": True, "track_id": payload["track_id"], "data": data}
        if _WORKER_MEL:
            out["mel"] = torch.from_numpy(mel_patch(tf))
        return out
    except Exception as exc:                                   # noqa: BLE001
        return {
            "ok": False, "track_id": payload.get("track_id", "?"),
            "path": payload.get("path"), "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=3),
        }


# --------------------------------------------------------------------------- #
def build_payloads(dataset: str, cfg, args) -> list[dict]:
    if dataset == "gtzan":
        df = load_gtzan(cfg)
        return [
            {"track_id": r.track_id, "path": r.path, "label": int(r.label),
             "split": r.split, "genre": r.genre}
            for r in df.itertuples()
        ]
    if dataset == "deam":
        df = load_deam(cfg)
        return [
            {"track_id": f"deam_{int(r.song_id)}", "path": r.path, "label": None,
             "split": r.split, "valence": float(r.valence), "arousal": float(r.arousal),
             "valence_z": float(r.valence_z), "arousal_z": float(r.arousal_z)}
            for r in df.itertuples()
        ]
    if dataset == "fma":
        df = load_fma(cfg, subset=args.subset)
        df = df[df["audio_exists"]]
        if df.empty:
            raise SystemExit(
                f"No FMA-{args.subset} audio found under data/raw/fma_{args.subset}/. "
                f"Download it first (fma_{args.subset}.zip), then rerun."
            )
        return [
            {"track_id": f"fma_{int(r.track_id)}", "path": r.path, "label": int(r.label),
             "split": r.split, "genre": r.genre}
            for r in df.itertuples()
        ]
    if dataset == "musiccaps":
        df, vocab = load_musiccaps(cfg)
        df = df[df["audio_exists"]]
        if df.empty:
            raise SystemExit(
                "No MusicCaps audio found under data/raw/musiccaps_audio/. "
                "Run: python scripts/download_musiccaps_audio.py"
            )
        log.info("MusicCaps: %d clips with recovered audio", len(df))
        return [
            {"track_id": str(r.ytid), "path": r.audio_path, "label": None,
             "y_multi": r.y.tolist(), "split": r.split}
            for r in df.itertuples()
        ]
    if dataset == "mtat":
        df, _ = load_mtat(cfg)
        df = df[df["audio_exists"]]
        if df.empty:
            raise SystemExit("No MTAT audio found under data/raw/mtat_audio/.")
        raw = ROOT / cfg.dotted("paths.raw")
        return [
            {"track_id": f"mtat_{int(r.clip_id)}", "path": str(raw / "mtat_audio" / r.mp3_path),
             "label": None, "y_multi": r.y.tolist(), "split": r.split}
            for r in df.itertuples()
        ]
    raise ValueError(f"unknown dataset {dataset!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["gtzan", "deam", "fma", "mtat", "musiccaps"])
    ap.add_argument("--graph-kind", default="segment", choices=["segment", "chord"])
    ap.add_argument("--subset", default="small", help="FMA subset (small|medium)")
    ap.add_argument("--workers", type=int, default=0, help="0 = os.cpu_count()-2")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of tracks")
    ap.add_argument("--save-mel", action="store_true",
                    help="also cache fixed-size mel patches for the CNN baseline")
    ap.add_argument("--export-json", type=int, default=0,
                    help="write N example graphs as JSON to results/graph_samples/")
    ap.add_argument("--force", action="store_true", help="rebuild even if cache exists")
    args = ap.parse_args()

    setup_logging(logging.INFO)
    cfg = load_config()
    set_seed(cfg.get("seed", 425))

    # Apply per-dataset segmentation overrides before anything reads cfg["graph"].
    # Workers re-read config.yaml from disk, so the override has to be passed
    # explicitly rather than mutated in the parent -- see _init_worker.
    overrides = dict(cfg.dotted(f"graph.dataset_overrides.{args.dataset}", {}) or {})
    if overrides:
        log.info("applying %s segmentation overrides: %s", args.dataset, overrides)
        cfg["graph"].update(overrides)

    tag = f"{args.dataset}{'_' + args.subset if args.dataset == 'fma' else ''}"
    stem = f"{tag}_{args.graph_kind}"
    out_pt = resolve(cfg, "processed") / f"{stem}_graphs.pt"
    if out_pt.exists() and not args.force:
        log.info("%s already exists -- pass --force to rebuild", out_pt.name)
        return

    payloads = build_payloads(args.dataset, cfg, args)
    if args.limit:
        payloads = payloads[: args.limit]
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    log.info("building %s graphs for %d %s tracks with %d workers",
             args.graph_kind, len(payloads), tag, workers)

    graphs, mels, failures = [], {}, []
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(cfg["_config_path"], args.graph_kind, args.save_mel, overrides),
    ) as pool:
        futs = [pool.submit(_process_one, p) for p in payloads]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["ok"]:
                graphs.append(res["data"])
                if "mel" in res:
                    mels[res["track_id"]] = res["mel"]
            else:
                failures.append(res)
                log.warning("FAILED %s -- %s", res["track_id"], res["error"])
            if i % 50 == 0 or i == len(futs):
                rate = i / max(1e-9, time.time() - t0)
                eta = (len(futs) - i) / max(rate, 1e-9)
                log.info("  %4d/%d done (%.1f trk/s, ETA %s)", i, len(futs), rate, human_time(eta))

    graphs.sort(key=lambda d: str(d.track_id))     # deterministic order regardless of completion race
    elapsed = time.time() - t0
    log.info("built %d graphs in %s (%d failures)", len(graphs), human_time(elapsed), len(failures))

    if not graphs:
        raise SystemExit("no graphs were built -- check the failure log above")

    torch.save(graphs, out_pt)
    log.info("wrote %s (%.1f MB)", out_pt, out_pt.stat().st_size / 1e6)
    if mels:
        mel_pt = resolve(cfg, "processed") / f"{stem}_mel.pt"
        torch.save(mels, mel_pt)
        log.info("wrote %s (%.1f MB)", mel_pt, mel_pt.stat().st_size / 1e6)

    # --- manifest: everything needed to audit this preprocessing run --------
    stats = graph_stats(graphs)
    by_split: dict[str, int] = {}
    for g in graphs:
        by_split[str(g.split)] = by_split.get(str(g.split), 0) + 1
    manifest = {
        "dataset": tag,
        "graph_kind": args.graph_kind,
        "num_graphs": len(graphs),
        "num_failures": len(failures),
        "failures": [{"track_id": f["track_id"], "error": f["error"]} for f in failures],
        "splits": by_split,
        "graph_stats": stats,
        "elapsed_seconds": round(elapsed, 1),
        "workers": workers,
        "audio_config": dict(cfg["audio"]),
        "graph_config": dict(cfg["graph"]),
        "graph_overrides_applied": overrides,
    }
    man = resolve(cfg, "processed") / f"{stem}_manifest.json"
    man.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log.info("graph stats: %s", json.dumps(stats))
    log.info("splits: %s", by_split)

    # --- exported examples (spec section 10.2) ------------------------------
    n_export = args.export_json or int(cfg.dotted("eval.num_exported_graphs", 0) or 0)
    if n_export:
        sample_dir = resolve(cfg, "graph_samples")
        step = max(1, len(graphs) // n_export)
        chosen = graphs[::step][:n_export]
        for g in chosen:
            save_graph_json(g, sample_dir / f"{g.track_id}_{args.graph_kind}.json")
            save_graph_pt(g, sample_dir / f"{g.track_id}_{args.graph_kind}.pt")
        log.info("exported %d example graphs (.pt + .json) to %s", len(chosen), sample_dir)


if __name__ == "__main__":
    main()
