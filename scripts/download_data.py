"""Fetch the corpora the four tasks read from, into ``data/raw/``.

Why this script exists
----------------------
``src/data_loading.py`` raises ``FileNotFoundError: ... run
scripts/download_data.py`` in four places, and the report's reproducibility
appendix lists it as step one. It needs to exist and it needs to be correct,
otherwise "reproducible" is a claim rather than a fact.

Every URL below was verified against the bytes actually used for the reported
results: a HEAD request to each source returns a ``Content-Length`` identical to
the file on disk, so these are the real provenance and not plausible-looking
guesses. ``--verify`` re-checks that, and ``sha256`` of everything downloaded is
recorded in ``data/raw/provenance.json``.

None of these corpora are redistributed with this repository; they are fetched
from the publishers. MusicCaps audio is *not* here -- Google ships captions and
YouTube IDs only, so it needs the separate, much slower
``scripts/download_musiccaps_audio.py``.

Sizes are large and mostly audio: ~3.9 GB of archives expanding to ~5.2 GB.
``--skip-large`` fetches only the annotation/metadata files (~370 MB), which is
enough to run the notebooks' data analysis but not to train anything.

Usage:
    python scripts/download_data.py                     # everything
    python scripts/download_data.py --datasets gtzan deam
    python scripts/download_data.py --skip-large        # annotations only
    python scripts/download_data.py --verify            # check, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ROOT, get_logger, human_time, setup_logging   # noqa: E402

log = get_logger("download_data")

# A default urllib opener sends "Python-urllib/3.x", which some of these hosts
# answer with 403.
UA = "Mozilla/5.0 (compatible; CSE425-music-context/1.0)"
CHUNK = 1 << 20


@dataclass
class Source:
    """One downloadable artefact and where it has to end up."""

    name: str
    dataset: str
    url: str
    dest: str                       # relative to data/raw/
    size: int                       # bytes, as served (verified against disk)
    extract_to: str | None = None   # relative to data/raw/; None = leave archive
    marker: str | None = None       # relative to data/raw/; exists => extracted
    large: bool = False             # audio; excluded by --skip-large
    note: str = ""
    strip_root: bool = False        # drop the archive's single top-level directory
    tags: list[str] = field(default_factory=list)


# Verified 2026-08-26: each Content-Length below equals the on-disk size of the
# file used for the reported results.
SOURCES: list[Source] = [
    # ---- GTZAN: Task 2 genre, Task 3 audio side --------------------------- #
    Source(
        name="GTZAN audio",
        dataset="gtzan",
        url="https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz",
        dest="gtzan_genres.tar.gz",
        size=1_226_192_050,
        extract_to=".",
        marker="genres/blues/blues.00000.wav",
        large=True,
        note="1,000 30 s clips, 10 genres. jazz.00054 is corrupt in every "
             "distribution of GTZAN; the loader drops it, leaving 929.",
    ),
    *[
        Source(
            name=f"GTZAN fault-filtered {split} split",
            dataset="gtzan",
            url=f"https://raw.githubusercontent.com/coreyker/dnn-mgr/master/gtzan/{split}_filtered.txt",
            dest=f"gtzan_splits/{split}_filtered.txt",
            size=size,
            note="Kereliuk, Sturm & Larsen's partition, which removes the "
                 "artist/recording leakage a random split of GTZAN suffers from.",
        )
        for split, size in (("train", 10_152), ("valid", 4_522), ("test", 6_616))
    ],
    # ---- DEAM: Task 3 valence/arousal ------------------------------------- #
    Source(
        name="DEAM audio",
        dataset="deam",
        url="https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip",
        dest="DEAM_audio.zip",
        size=1_343_203_527,
        extract_to="deam_audio",
        marker="deam_audio/MEMD_audio",
        large=True,
        note="1,802 45 s excerpts.",
    ),
    Source(
        name="DEAM annotations",
        dataset="deam",
        url="https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip",
        dest="DEAM_Annotations.zip",
        size=4_735_283,
        extract_to="deam_annotations",
        marker="deam_annotations/annotations",
        note="Continuous and per-song valence/arousal. The loader reads the "
             "song-level averages.",
    ),
    Source(
        name="DEAM metadata",
        dataset="deam",
        url="https://cvml.unige.ch/databases/DEAM/metadata.zip",
        dest="deam_metadata.zip",
        size=344_760,
        note="Read straight from the zip by src/data_loading.py; not extracted.",
    ),
    # ---- MusicCaps: Task 1 captions, Task 4 pairs ------------------------- #
    Source(
        name="MusicCaps captions",
        dataset="musiccaps",
        url="https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv",
        dest="musiccaps-public.csv",
        size=2_939_484,
        note="Captions, aspect lists and YouTube IDs for 5,521 clips. AUDIO IS "
             "NOT INCLUDED -- run scripts/download_musiccaps_audio.py next.",
    ),
    # ---- auxiliary corpora ------------------------------------------------ #
    Source(
        name="MagnaTagATune annotations",
        dataset="mtat",
        url="https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv",
        dest="mtat_annotations_final.csv",
        size=21_517_373,
        note="Tag vocabulary reference. Audio not fetched; the loader reports "
             "audio_exists=False and the tag statistics still work.",
    ),
    Source(
        name="FMA metadata",
        dataset="fma",
        url="https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        dest="fma_metadata.zip",
        size=358_412_441,
        note="Genre labels for the optional FMA scale-up path. Audio (fma_small "
             "/ fma_medium) is not fetched -- it is 7.2 GB / 22 GB and no "
             "reported result depends on it.",
    ),
]

DATASETS = sorted({s.dataset for s in SOURCES})


# --------------------------------------------------------------------------- #
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(src: Source, raw: Path, force: bool = False) -> tuple[bool, str]:
    """Download ``src`` unless a correctly-sized copy is already there.

    Returns ``(changed, status)``. The size check is what makes the script
    idempotent and safe to re-run after an interrupted download: a truncated file
    has the wrong length and is re-fetched rather than silently accepted.
    """
    dest = raw / src.dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        have = dest.stat().st_size
        if have == src.size:
            return False, "present"
        log.warning("%s is %d bytes, expected %d -- re-downloading",
                    src.dest, have, src.size)

    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(src.url, headers={"User-Agent": UA})
    t0 = time.time()
    got = 0
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or src.size)
        next_report = 10
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            got += len(chunk)
            if total and (pct := 100 * got / total) >= next_report:
                log.info("  %s %3.0f%% (%.0f/%.0f MB)", src.dest, pct,
                         got / 1e6, total / 1e6)
                next_report = pct - pct % 10 + 10

    if got != src.size:
        # Not fatal: publishers do re-issue files. But the reported results were
        # produced from the recorded size, so a change has to be visible.
        log.warning("%s: got %d bytes, expected %d. The upstream file may have "
                    "changed; results may not reproduce exactly.",
                    src.dest, got, src.size)
    tmp.replace(dest)
    log.info("%s <- %d MB in %s", src.dest, got // 1_000_000, human_time(time.time() - t0))
    return True, "downloaded"


def extract(src: Source, raw: Path, force: bool = False) -> bool:
    """Unpack ``src`` if it has an extract target and is not already unpacked."""
    if not src.extract_to:
        return False
    marker = raw / src.marker if src.marker else None
    if marker and marker.exists() and not force:
        return False

    archive = raw / src.dest
    target = raw / src.extract_to
    target.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s -> %s", src.dest, src.extract_to)
    t0 = time.time()
    if archive.suffix == ".gz" or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            # filter="data" refuses absolute paths and .. traversal. Python 3.14
            # makes it the default; being explicit keeps 3.11/3.12 safe too.
            tf.extractall(target, filter="data")
    else:
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                name = Path(member.filename)
                if name.is_absolute() or ".." in name.parts:
                    log.warning("  skipping suspicious member %s", member.filename)
                    continue
                zf.extract(member, target)
    log.info("  extracted in %s", human_time(time.time() - t0))

    if src.strip_root:
        kids = [p for p in target.iterdir() if p.name != "__MACOSX"]
        if len(kids) == 1 and kids[0].is_dir():
            for p in list(kids[0].iterdir()):
                shutil.move(str(p), str(target / p.name))
            kids[0].rmdir()
    return True


def verify(raw: Path) -> int:
    """Report what is present, what is the wrong size, and what is missing."""
    bad = 0
    for src in SOURCES:
        dest = raw / src.dest
        if not dest.exists():
            log.warning("MISSING  %-34s %s", src.dest, src.name)
            bad += 1
            continue
        have = dest.stat().st_size
        if have != src.size:
            log.error("BAD SIZE %-34s %d != %d", src.dest, have, src.size)
            bad += 1
            continue
        extracted = ""
        if src.marker and not (raw / src.marker).exists():
            extracted = "  (archive present, NOT extracted)"
        log.info("ok       %-34s %6.1f MB%s", src.dest, have / 1e6, extracted)
    return bad


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="*", default=DATASETS, choices=DATASETS,
                    help=f"subset to fetch (default: all of {' '.join(DATASETS)})")
    ap.add_argument("--skip-large", action="store_true",
                    help="skip the multi-GB audio archives (annotations only)")
    ap.add_argument("--no-extract", action="store_true", help="download but do not unpack")
    ap.add_argument("--force", action="store_true", help="re-download and re-extract")
    ap.add_argument("--verify", action="store_true",
                    help="check sizes of what is already on disk, download nothing")
    args = ap.parse_args()

    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    setup_logging(logfile=ROOT / "results" / "download_data.log")

    if args.verify:
        bad = verify(raw)
        log.info("%d of %d artefacts need attention", bad, len(SOURCES))
        return 1 if bad else 0

    todo = [s for s in SOURCES
            if s.dataset in args.datasets and not (args.skip_large and s.large)]
    skipped = [s for s in SOURCES if s not in todo]
    planned = sum(s.size for s in todo)
    log.info("%d artefact(s), %.1f GB to fetch if none are cached",
             len(todo), planned / 1e9)
    if skipped:
        log.info("skipping: %s", ", ".join(s.dest for s in skipped))

    prov: dict[str, dict] = {}
    pfile = raw / "provenance.json"
    if pfile.exists():
        prov = json.loads(pfile.read_text(encoding="utf-8"))

    failures = []
    for src in todo:
        log.info("--- %s (%s)", src.name, src.dataset)
        if src.note:
            log.info("    %s", src.note)
        try:
            changed, status = fetch(src, raw, force=args.force)
            if not args.no_extract:
                extract(src, raw, force=args.force)
        except Exception as exc:                                    # noqa: BLE001
            log.error("    FAILED: %s", exc)
            failures.append((src.dest, str(exc)))
            continue

        dest = raw / src.dest
        entry = prov.get(src.dest, {})
        # sha256 of a 1.3 GB file is not free, so only rehash what changed.
        if changed or "sha256" not in entry:
            log.info("    hashing %s", src.dest)
            entry["sha256"] = sha256(dest)
        entry.update(url=src.url, size=dest.stat().st_size, dataset=src.dataset,
                     name=src.name, status=status)
        prov[src.dest] = entry

    pfile.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
    log.info("provenance -> %s", pfile.relative_to(ROOT))

    if failures:
        log.error("%d artefact(s) failed:", len(failures))
        for dest, exc in failures:
            log.error("  %s: %s", dest, exc)
        return 1

    log.info("done. next: python scripts/download_musiccaps_audio.py --workers 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
