"""Dataset loading, tag-vocabulary mining, and leakage-safe splits.

Covers the four sources actually used for the reported results:

* **GTZAN**   -- 10-genre single-label audio, with the published *fault-filtered*
  partition of Kereliuk et al. so no artist appears on both sides of the split.
* **MusicCaps** -- 5.5k expert captions; drives the Task 1 caption->tag task and
  the Task 4 contrastive pairs.
* **DEAM**    -- 1,802 excerpts with mean valence/arousal, for the Task 3
  auxiliary regression term.
* **MagnaTagATune** -- 25,863 clips x 188 tags; loader + official folder split
  provided for scale-up (audio archive not downloaded for the reported runs).

A note on the Task 1 label design
---------------------------------
MusicCaps ships an ``aspect_list`` per clip. Turning those aspects into
multi-label targets makes the caption->tag task partly *lexical*: the caption
usually contains the aspect string verbatim ("...it sounds sad" for aspect
``sad``). We therefore build two variants and report both:

  ``naive``  -- caption used as-is. A model can win by string matching.
  ``masked`` -- every surface form of every tag in the vocabulary is removed
                from the caption before tokenisation, so the label can only be
                recovered from the surrounding description. This is the honest
                measure of context understanding, and it is what the headline
                Task 1 number in the report refers to.

:func:`lexical_match_baseline` scores the pure string-matching predictor, which
quantifies exactly how much of the naive task is lookup rather than inference.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import Config, ROOT, get_logger

log = get_logger("data_loading")

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# Known-corrupt file in the original GTZAN distribution (truncated payload).
# Excluded from every split; the fault-filtered lists already omit it, but we
# guard anyway so a naive stratified split cannot pick it up.
GTZAN_CORRUPT = {"jazz.00054"}


# --------------------------------------------------------------------------- #
# GTZAN
# --------------------------------------------------------------------------- #
def load_gtzan(cfg: Config) -> pd.DataFrame:
    """-> DataFrame[track_id, path, genre, label, split].

    ``split`` comes from the fault-filtered partition when the list files are
    present, otherwise from a seeded genre-stratified split (and we log loudly,
    because that variant leaks artists and inflates accuracy).
    """
    raw = ROOT / cfg.dotted("paths.raw")
    root = raw / "genres"
    if not root.exists():
        raise FileNotFoundError(f"GTZAN not found at {root} -- run scripts/download_data.py")

    rows = []
    for gi, genre in enumerate(GTZAN_GENRES):
        for wav in sorted((root / genre).glob("*.wav")):
            tid = wav.stem
            if tid in GTZAN_CORRUPT:
                log.warning("skipping known-corrupt GTZAN file %s", tid)
                continue
            rows.append({"track_id": tid, "path": str(wav), "genre": genre, "label": gi})
    df = pd.DataFrame(rows)

    split_map = _gtzan_filtered_splits(raw)
    if split_map:
        df["split"] = df["track_id"].map(split_map)
        n_drop = int(df["split"].isna().sum())
        df = df.dropna(subset=["split"]).reset_index(drop=True)
        log.info(
            "GTZAN: fault-filtered partition -- %d tracks (%d excluded as faulty/duplicate)",
            len(df), n_drop,
        )
    else:
        log.warning(
            "GTZAN fault-filtered lists absent -- falling back to a stratified random "
            "split. Artist replication WILL leak across train/test and inflate results."
        )
        df["split"] = stratified_split(df["label"].to_numpy(), cfg)
    return df


def _gtzan_filtered_splits(raw: Path) -> dict[str, str]:
    """Parse the published fault-filtered lists (Kereliuk, Sturm & Larsen 2015).
    Their ``valid`` becomes our ``val``."""
    out: dict[str, str] = {}
    name_map = {"train": "train", "valid": "val", "test": "test"}
    d = raw / "gtzan_splits"
    for fname, split in name_map.items():
        f = d / f"{fname}_filtered.txt"
        if not f.exists():
            return {}
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out[Path(line).stem] = split
    return out


# --------------------------------------------------------------------------- #
# MusicCaps
# --------------------------------------------------------------------------- #
@dataclass
class TagVocab:
    """Multi-label tag vocabulary mined from MusicCaps aspects."""

    tags: list[str]
    index: dict[str, int] = field(default_factory=dict)
    patterns: list[re.Pattern] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.tags)}
        # word-boundary match on the whole tag phrase, case-insensitive
        self.patterns = [
            re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
            for t in self.tags
        ]

    def __len__(self) -> int:
        return len(self.tags)

    def encode(self, aspects: list[str]) -> np.ndarray:
        y = np.zeros(len(self.tags), dtype=np.float32)
        for a in aspects:
            i = self.index.get(a.strip().lower())
            if i is not None:
                y[i] = 1.0
        return y

    def mask_caption(self, caption: str) -> str:
        """Remove every vocabulary tag's surface form from the caption."""
        out = caption
        for pat in self.patterns:
            out = pat.sub(" ", out)
        return re.sub(r"\s{2,}", " ", out).strip()

    def lexical_predict(self, caption: str) -> np.ndarray:
        """1 where the tag phrase literally occurs in the caption."""
        return np.array(
            [1.0 if pat.search(caption) else 0.0 for pat in self.patterns], dtype=np.float32
        )


def load_musiccaps(cfg: Config, num_tags: int | None = None) -> tuple[pd.DataFrame, TagVocab]:
    """-> (DataFrame[ytid, caption, caption_masked, aspects, y, split], TagVocab).

    Split policy: MusicCaps marks an ``is_audioset_eval`` subset. We hold that
    out as test, then split the remainder 82/18 into train/val with a fixed
    seed, so the test set is a published subset rather than our own draw.
    """
    raw = ROOT / cfg.dotted("paths.raw")
    csv = raw / "musiccaps-public.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} missing -- run scripts/download_data.py")

    df = pd.read_csv(csv)
    df["aspects"] = df["aspect_list"].apply(_parse_aspects)
    df = df[df["aspects"].str.len() > 0].reset_index(drop=True)

    k = int(num_tags or cfg.dotted("train.task1_bert.num_tags", 50))
    counts = Counter(a for row in df["aspects"] for a in row)
    tags = [t for t, _ in counts.most_common(k)]
    vocab = TagVocab(tags=tags)
    log.info(
        "MusicCaps: %d clips, %d distinct aspects -> top-%d vocabulary "
        "(most common: %s)",
        len(df), len(counts), len(tags), ", ".join(tags[:5]),
    )

    df["y"] = df["aspects"].apply(vocab.encode)
    # clips with no positive label in the top-K carry no training signal
    keep = df["y"].apply(lambda v: v.sum() > 0)
    dropped = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    if dropped:
        log.info("MusicCaps: dropped %d clips with no top-%d tag", dropped, len(tags))

    df["caption"] = df["caption"].astype(str)
    df["caption_masked"] = df["caption"].apply(vocab.mask_caption)

    # Audio is reconstructed locally by scripts/download_musiccaps_audio.py (the
    # dataset ships YouTube IDs, not audio). Tasks 3 and 4 use only the rows where
    # the clip was actually recoverable; Task 1 is text-only and uses all rows.
    audio_dir = raw / "musiccaps_audio"
    df["audio_path"] = [str(audio_dir / f"{y}.wav") for y in df["ytid"]]
    df["audio_exists"] = [Path(p).exists() for p in df["audio_path"]]
    log.info("MusicCaps: %d/%d clips have local audio (%.1f%% recovered)",
             int(df["audio_exists"].sum()), len(df),
             100 * df["audio_exists"].mean())

    # --- splits -------------------------------------------------------------
    rng = np.random.default_rng(cfg.get("seed", 425))
    is_eval = df["is_audioset_eval"].astype(str).str.lower().isin({"true", "1"})
    split = np.where(is_eval, "test", "train").astype(object)
    pool = np.flatnonzero(~is_eval.to_numpy())
    rng.shuffle(pool)
    n_val = int(round(0.18 * pool.size))
    split[pool[:n_val]] = "val"
    df["split"] = split
    log.info(
        "MusicCaps splits -- train %d / val %d / test %d (test = published audioset_eval subset)",
        int((df.split == "train").sum()), int((df.split == "val").sum()),
        int((df.split == "test").sum()),
    )
    return df, vocab


def _parse_aspects(cell: object) -> list[str]:
    """``aspect_list`` is a stringified Python list."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        vals = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(vals, (list, tuple)):
        return []
    return [str(v).strip().lower() for v in vals if str(v).strip()]


def lexical_match_baseline(df: pd.DataFrame, vocab: TagVocab, text_col: str = "caption"):
    """Baseline B0: predict a tag iff its phrase occurs in the text.

    Quantifies how much of the naive caption->tag task is string lookup.
    Returns (y_true, y_pred) stacked over rows.
    """
    y_true = np.stack(df["y"].to_list())
    y_pred = np.stack([vocab.lexical_predict(t) for t in df[text_col].astype(str)])
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# DEAM
# --------------------------------------------------------------------------- #
def _deam_metadata(raw: Path) -> dict[int, str]:
    """song_id -> a short natural-language description built from DEAM's own
    release metadata (``metadata.zip``: artist, title, album, genre).

    Why this and not a templated caption: Task 3 needs a text channel for the
    valence/arousal term, and DEAM ships no captions. Templating text from the
    emotion label would leak the regression target into the text branch and make
    the fusion result meaningless. Release metadata is genuinely available at
    inference time for any catalogue track and contains no emotion annotation.

    The 2014/2015 files also carry free-form last.fm tags, and those *do* include
    affective words ("melancholy", "energetic"). We deliberately exclude them --
    including them would reintroduce the leak we are avoiding. Only the structured
    artist/title/album/genre fields are used.

    The three yearly CSVs are inconsistently formatted (tab-padded quoted fields
    in 2013, a variable number of trailing tag columns in 2014/2015), so they are
    parsed positionally with the stdlib csv reader rather than pandas.
    """
    import csv
    import io
    import zipfile

    zpath = raw / "deam_metadata.zip"
    if not zpath.exists():
        log.warning("DEAM metadata.zip absent -- Task 3 text channel will be genre-free")
        return {}

    def clean(s: str) -> str:
        return re.sub(r"\s{2,}", " ", str(s).replace("\t", " ").strip().strip('"').strip())

    def is_texty(s: str) -> bool:
        """Guards against the column-shifted rows in metadata_2015.csv, where an
        extra unquoted field pushes a numeric id into the title slot."""
        return bool(s) and s.lower() != "n/a" and not s.replace(".", "").isdigit()

    # (file, index of song_id, {field: column index})
    layouts = {
        "metadata/metadata_2013.csv": (0, {"artist": 2, "title": 3, "genre": 6}),
        "metadata/metadata_2014.csv": (0, {"artist": 1, "album": 2, "title": 3, "genre": 4}),
        "metadata/metadata_2015.csv": (0, {"title": 2, "artist": 3, "album": 4, "genre": 5}),
    }

    out: dict[int, str] = {}
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        for fname, (id_col, cols) in layouts.items():
            if fname not in names:
                continue
            text = z.read(fname).decode("utf-8", errors="replace")
            for row in list(csv.reader(io.StringIO(text)))[1:]:
                if len(row) <= max([id_col, *cols.values()]):
                    continue
                try:
                    sid = int(clean(row[id_col]))
                except ValueError:
                    continue
                parts = {k: clean(row[i]) for k, i in cols.items()}
                # genre fields are hyphen-joined multi-labels in 2014
                genre = parts.get("genre", "").replace("-", ", ")
                bits = []
                if is_texty(parts.get("title", "")):
                    bits.append(f'"{parts["title"]}"')
                if is_texty(parts.get("artist", "")):
                    bits.append(f'by {parts["artist"]}')
                if is_texty(parts.get("album", "")):
                    bits.append(f'from the album {parts["album"]}')
                if is_texty(genre):
                    bits.append(f"in the {genre.lower()} genre")
                if bits:
                    out[sid] = "A music track " + " ".join(bits) + "."
    log.info("DEAM metadata: text descriptions for %d songs", len(out))
    return out


def load_deam(cfg: Config, with_text: bool = True) -> pd.DataFrame:
    """-> DataFrame[song_id, path, valence, arousal, valence_z, arousal_z, split, text].

    Targets are the per-song means on the original 1-9 scale, plus z-scored
    copies (statistics computed on the *train* split only) which is what the
    regression head actually fits. ``text`` is the release-metadata description
    used as Task 3's text channel (see :func:`_deam_metadata`).
    """
    raw = ROOT / cfg.dotted("paths.raw")
    ann_dir = raw / "deam_annotations" / "annotations" / "annotations averaged per song" / "song_level"
    if not ann_dir.exists():
        raise FileNotFoundError(f"DEAM annotations missing at {ann_dir}")

    frames = []
    for f in sorted(ann_dir.glob("static_annotations_averaged_songs_*.csv")):
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["song_id", "valence_mean", "arousal_mean"]])
    ann = pd.concat(frames, ignore_index=True).drop_duplicates("song_id")

    audio_dir = raw / "deam_audio"
    paths = {int(p.stem): str(p) for p in audio_dir.rglob("*.mp3") if p.stem.isdigit()}
    ann["path"] = ann["song_id"].map(paths)
    missing = int(ann["path"].isna().sum())
    ann = ann.dropna(subset=["path"]).reset_index(drop=True)
    if missing:
        log.warning("DEAM: %d annotated songs have no audio file -- dropped", missing)

    ann = ann.rename(columns={"valence_mean": "valence", "arousal_mean": "arousal"})
    ann["split"] = stratified_split(_quadrant(ann), cfg)

    if with_text:
        meta = _deam_metadata(raw)
        ann["text"] = ann["song_id"].astype(int).map(meta)
        n_missing = int(ann["text"].isna().sum())
        # a song with no metadata row still has a usable graph and V/A target;
        # give it a neutral non-empty string rather than dropping it
        ann["text"] = ann["text"].fillna("A music track.")
        if n_missing:
            log.info("DEAM: %d/%d songs lack release metadata -> neutral placeholder text",
                     n_missing, len(ann))

    tr = ann["split"] == "train"
    for col in ("valence", "arousal"):
        mu, sd = ann.loc[tr, col].mean(), ann.loc[tr, col].std()
        ann[f"{col}_z"] = (ann[col] - mu) / (sd + 1e-8)
        ann.attrs[f"{col}_mu"], ann.attrs[f"{col}_sd"] = float(mu), float(sd)

    log.info(
        "DEAM: %d songs with audio+annotation (train %d / val %d / test %d)",
        len(ann), int(tr.sum()),
        int((ann.split == "val").sum()), int((ann.split == "test").sum()),
    )
    return ann


def _quadrant(df: pd.DataFrame) -> np.ndarray:
    """Valence/arousal quadrant (mid = 5 on the 1-9 scale), used only to
    stratify the split so all four emotion quadrants appear in every fold."""
    v = (df["valence"].to_numpy() >= 5).astype(int)
    a = (df["arousal"].to_numpy() >= 5).astype(int)
    return v * 2 + a


# --------------------------------------------------------------------------- #
# MagnaTagATune (loader for scale-up; audio archive optional)
# --------------------------------------------------------------------------- #
def load_mtat(cfg: Config, num_tags: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """-> (DataFrame[clip_id, mp3_path, y, split], top-N tag names).

    Uses the standard MTAT protocol: top-50 tags by frequency, and the official
    *directory* split (folders 1-c train, d val, e-f test). Splitting by folder
    rather than at random is essential -- MTAT contains several 29 s clips cut
    from the same song, so a random clip-level split leaks songs across folds.
    """
    raw = ROOT / cfg.dotted("paths.raw")
    csv = raw / "mtat_annotations_final.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} missing -- run scripts/download_data.py")

    df = pd.read_csv(csv, sep="\t")
    df.columns = [c.strip().strip('"') for c in df.columns]
    tag_cols = [c for c in df.columns if c not in {"clip_id", "mp3_path"}]

    freq = df[tag_cols].sum(axis=0).sort_values(ascending=False)
    top = [str(t) for t in freq.head(num_tags).index]
    df["y"] = list(df[top].to_numpy(dtype=np.float32))
    df = df[df[top].sum(axis=1) > 0].reset_index(drop=True)

    folder = df["mp3_path"].astype(str).str.split("/").str[0]
    split = np.full(len(df), "train", dtype=object)
    split[folder.isin(["d"]).to_numpy()] = "val"
    split[folder.isin(["e", "f"]).to_numpy()] = "test"
    df["split"] = split

    audio_root = raw / "mtat_audio"
    df["audio_exists"] = [(audio_root / p).exists() for p in df["mp3_path"].astype(str)]
    log.info(
        "MTAT: %d clips with >=1 top-%d tag; %d have local audio. "
        "Official folder split -- train %d / val %d / test %d",
        len(df), num_tags, int(df["audio_exists"].sum()),
        int((split == "train").sum()), int((split == "val").sum()), int((split == "test").sum()),
    )
    return df, top


# --------------------------------------------------------------------------- #
# FMA (scale-up path; metadata is downloaded, audio is opt-in)
# --------------------------------------------------------------------------- #
def load_fma(cfg: Config, subset: str = "small") -> pd.DataFrame:
    """-> DataFrame[track_id, path, genre, label, split, artist_id].

    Reads ``fma_metadata.zip`` directly (no extraction needed). Splits are
    grouped by ``artist_id`` so no artist crosses folds -- FMA's own
    ``set,split`` column is used when present, since it is the published one.
    """
    import zipfile

    raw = ROOT / cfg.dotted("paths.raw")
    zpath = raw / "fma_metadata.zip"
    if not zpath.exists():
        raise FileNotFoundError(f"{zpath} missing -- run scripts/download_data.py")

    with zipfile.ZipFile(zpath) as z:
        with z.open("fma_metadata/tracks.csv") as fh:
            tracks = pd.read_csv(fh, index_col=0, header=[0, 1], low_memory=False)

    sel = tracks[("set", "subset")].astype(str).str.strip() <= subset if False else None
    order = {"small": 0, "medium": 1, "large": 2, "full": 3}
    rank = tracks[("set", "subset")].astype(str).str.strip().map(order)
    df = tracks[rank <= order[subset]].copy()

    out = pd.DataFrame({
        "track_id": df.index.astype(int),
        "genre": df[("track", "genre_top")].astype(str),
        "artist_id": df[("artist", "id")] if ("artist", "id") in df.columns else -1,
        "split": df[("set", "split")].astype(str),
    })
    out = out[out["genre"].notna() & (out["genre"] != "nan")].reset_index(drop=True)
    genres = sorted(out["genre"].unique())
    out["label"] = out["genre"].map({g: i for i, g in enumerate(genres)})
    out["split"] = out["split"].replace({"validation": "val"})

    audio_root = raw / f"fma_{subset}"
    out["path"] = [str(audio_root / f"{t//1000:03d}" / f"{t:06d}.mp3") for t in out["track_id"]]
    out["audio_exists"] = [Path(p).exists() for p in out["path"]]
    log.info(
        "FMA-%s: %d tracks, %d genres, %d with local audio (published split: "
        "train %d / val %d / test %d)",
        subset, len(out), len(genres), int(out["audio_exists"].sum()),
        int((out.split == "train").sum()), int((out.split == "val").sum()),
        int((out.split == "test").sum()),
    )
    return out


# --------------------------------------------------------------------------- #
# Generic splits
# --------------------------------------------------------------------------- #
def stratified_split(labels: np.ndarray, cfg: Config) -> np.ndarray:
    """Seeded stratified train/val/test assignment over integer class labels."""
    s = cfg.splits
    rng = np.random.default_rng(cfg.get("seed", 425))
    labels = np.asarray(labels)
    out = np.empty(labels.shape[0], dtype=object)
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = idx.size
        n_tr = int(round(s["train"] * n))
        n_va = int(round(s["val"] * n))
        out[idx[:n_tr]] = "train"
        out[idx[n_tr:n_tr + n_va]] = "val"
        out[idx[n_tr + n_va:]] = "test"
    return out


def grouped_split(groups: np.ndarray, cfg: Config) -> np.ndarray:
    """Assign whole groups (artist, song, album) to one fold each -- the
    mechanism behind the spec's 'no artist leakage' requirement."""
    s = cfg.splits
    rng = np.random.default_rng(cfg.get("seed", 425))
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n = uniq.size
    n_tr, n_va = int(round(s["train"] * n)), int(round(s["val"] * n))
    assign = {}
    for i, g in enumerate(uniq):
        assign[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return np.array([assign[g] for g in groups], dtype=object)


def split_report(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Per-split class counts -- printed before every training run and saved to
    data/splits/ so the report can show the splits were balanced."""
    return (
        df.groupby(["split", label_col]).size().unstack(fill_value=0)
        if label_col in df.columns else df.groupby("split").size().to_frame("n")
    )
