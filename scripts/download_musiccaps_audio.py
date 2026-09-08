"""Fetch MusicCaps audio and cut the 10 s window each caption describes.

Why this script exists
----------------------
MusicCaps ships captions plus YouTube IDs -- Google does not redistribute the
audio. Without it, Tasks 3 and 4 have no *real* paired (audio, text) data, and
the only alternative is templating captions from metadata labels, which leaks the
label into the text channel and makes the fusion result meaningless.

So we reconstruct the audio: download the audio track, decode only the
[start_s, end_s] window with PyAV (which bundles ffmpeg, so no system ffmpeg is
required), and write a 22.05 kHz mono WAV.

Honest accounting: a meaningful fraction of MusicCaps videos are now deleted,
private, or region-blocked. The script records exactly which IDs failed and why,
and writes the recovery rate to ``data/raw/musiccaps_audio/manifest.json``. That
rate is reported in the paper -- results are on the clips we could recover, not
on the nominal 5,521.

Usage:
    python scripts/download_musiccaps_audio.py --workers 6
    python scripts/download_musiccaps_audio.py --limit 200        # quick trial
    python scripts/download_musiccaps_audio.py --retry-failed     # second pass
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ROOT, human_time, load_config, setup_logging   # noqa: E402

log = logging.getLogger("musiccaps_dl")

TARGET_SR = 22050
_print_lock = threading.Lock()


# --------------------------------------------------------------------------- #
def decode_window(path: Path, start_s: float, end_s: float, sr: int = TARGET_SR) -> np.ndarray:
    """Decode only [start_s, end_s] from a media file, resampled to mono/sr.

    Seeks to ``start_s`` and stops as soon as the window is filled, so cost is
    proportional to the 10 s window rather than the whole video.
    """
    import av

    want = int(round((end_s - start_s) * sr))
    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream")
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)

        if start_s > 0:
            # seek in the stream's own time base, landing on the keyframe at or
            # before start_s; leading slack is trimmed after decode
            ts = int(start_s / float(stream.time_base)) if stream.time_base else 0
            try:
                container.seek(ts, stream=stream, any_frame=False, backward=True)
            except Exception:                                  # noqa: BLE001
                container.seek(0)

        got = 0
        base_pts = None
        for frame in container.decode(stream):
            if base_pts is None:
                base_pts = float(frame.pts * frame.time_base) if frame.pts is not None else start_s
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray().ravel()
                out.append(arr)
                got += arr.size
            # decode a little past the window to absorb seek imprecision
            if got >= want + 2 * sr:
                break
        for rf in resampler.resample(None):
            out.append(rf.to_ndarray().ravel())

    if not out:
        raise ValueError("decoded zero samples")
    y = np.concatenate(out).astype(np.float32)

    # trim the slack between the keyframe we landed on and the true start
    offset = 0
    if base_pts is not None and start_s > base_pts:
        offset = int(round((start_s - base_pts) * sr))
    y = y[offset:offset + want] if offset + want <= y.size else y[-want:] if y.size >= want else y

    if y.size < int(0.5 * want):
        raise ValueError(f"window too short: {y.size}/{want} samples")
    if y.size < want:                                  # pad a slightly short tail
        y = np.pad(y, (0, want - y.size), mode="constant")
    peak = float(np.abs(y).max())
    if peak < 1e-4:
        raise ValueError("window is silent")
    return (y / peak).astype(np.float32)


# --------------------------------------------------------------------------- #
# YouTube's anti-bot gate is the dominant failure mode, not genuinely-missing
# videos: the first full pass over MusicCaps lost 3622 clips to "sign in to
# confirm you're not a bot" against only 231 actually-unavailable ones. Which
# player client yt-dlp impersonates is what decides whether that gate fires, and
# no single client works reliably for long, so each ID is retried across several.
PLAYER_CLIENTS = ["tv", "ios", "web_safari", "mweb", "android_vr", "web"]


def fetch_one(row: dict, tmp_dir: Path, out_dir: Path, quiet: bool = True,
              clients: list[str] | None = None) -> dict:
    """Download -> decode window -> write WAV -> delete the source file.

    Tries each player client in turn and only gives up when all of them fail, so
    a bot-check on one client does not cost the clip.
    """
    import yt_dlp

    ytid = row["ytid"]
    out_wav = out_dir / f"{ytid}.wav"
    if out_wav.exists():
        return {"ytid": ytid, "ok": True, "cached": True}

    tmpl = str(tmp_dir / f"{ytid}.%(ext)s")
    last_err = "no attempt made"
    for client in (clients or PLAYER_CLIENTS):
        opts = {
            # opus at <=130 kbps is ample for 22 kHz mel/chroma and keeps the
            # download an order of magnitude smaller than bestaudio
            "format": "bestaudio[abr<=130]/bestaudio/best",
            "outtmpl": tmpl,
            "quiet": quiet, "no_warnings": quiet, "noprogress": True,
            "noplaylist": True, "retries": 2, "socket_timeout": 25,
            "nocheckcertificate": True, "geo_bypass": True,
            "extractor_args": {"youtube": {"player_client": [client]}},
        }
        src: Path | None = None
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
            cands = list(tmp_dir.glob(f"{ytid}.*"))
            if not cands:
                last_err = "download produced no file"
                continue
            src = max(cands, key=lambda p: p.stat().st_size)

            y = decode_window(src, float(row["start_s"]), float(row["end_s"]))
            sf.write(out_wav, y, TARGET_SR, subtype="PCM_16")
            return {"ytid": ytid, "ok": True, "cached": False, "client": client,
                    "samples": int(y.size)}
        except Exception as exc:                                   # noqa: BLE001
            last_err = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:160]}"
            # a video that is genuinely gone will be gone on every client, so stop
            if classify(last_err) in {"private_video", "video_unavailable", "geo_blocked"}:
                break
        finally:
            for p in tmp_dir.glob(f"{ytid}.*"):
                try:
                    p.unlink()
                except OSError:
                    pass
    return {"ytid": ytid, "ok": False, "error": last_err}


# --------------------------------------------------------------------------- #
def classify(err: str) -> str:
    """Bucket failures so the manifest explains *why* clips were lost."""
    e = err.lower()
    if "private" in e:
        return "private_video"
    if "unavailable" in e or "removed" in e or "terminated" in e:
        return "video_unavailable"
    if "not available in your country" in e or "geo" in e or "blocked" in e:
        return "geo_blocked"
    if "sign in" in e or "bot" in e or "captcha" in e:
        return "bot_check"
    if "silent" in e or "too short" in e or "zero samples" in e:
        return "bad_audio_window"
    if "timeout" in e or "timed out" in e or "connection" in e:
        return "network"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent downloads; >8 tends to trigger throttling")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-failed", action="store_true",
                    help="only attempt IDs that failed in a previous run")
    ap.add_argument("--verbose-ytdlp", action="store_true")
    args = ap.parse_args()

    setup_logging(logging.INFO)
    cfg = load_config()
    raw = ROOT / cfg.dotted("paths.raw")
    out_dir = raw / "musiccaps_audio"
    tmp_dir = raw / "_musiccaps_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    import pandas as pd
    df = pd.read_csv(raw / "musiccaps-public.csv")
    rows = df[["ytid", "start_s", "end_s"]].to_dict("records")

    if args.retry_failed and manifest_path.exists():
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        failed = {f["ytid"] for f in prev.get("failures", [])}
        rows = [r for r in rows if r["ytid"] in failed]
        log.info("retry mode: %d previously-failed IDs", len(rows))

    have = {p.stem for p in out_dir.glob("*.wav")}
    rows = [r for r in rows if r["ytid"] not in have]
    if args.limit:
        rows = rows[: args.limit]
    log.info("%d clips already on disk; attempting %d with %d workers",
             len(have), len(rows), args.workers)
    if not rows:
        log.info("nothing to do")
        return

    ok, failures = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(fetch_one, r, tmp_dir, out_dir, not args.verbose_ytdlp): r for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["ok"]:
                ok += 1
            else:
                failures.append({"ytid": res["ytid"], "error": res["error"],
                                 "reason": classify(res["error"])})
            if i % 25 == 0 or i == len(futs):
                rate = i / max(1e-9, time.time() - t0)
                with _print_lock:
                    log.info("  %4d/%d | ok %d | fail %d | %.2f clip/s | ETA %s",
                             i, len(futs), ok, len(failures), rate,
                             human_time((len(futs) - i) / max(rate, 1e-9)))

    total_on_disk = len(list(out_dir.glob("*.wav")))
    reasons: dict[str, int] = {}
    for f in failures:
        reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1

    this_pass = {
        "mode": "retry-failed" if args.retry_failed else "full",
        "workers": args.workers,
        "player_clients": PLAYER_CLIENTS,
        "attempted": len(rows),
        "succeeded": ok,
        "failed": len(failures),
        "failure_reasons": reasons,
        "wav_on_disk_after": total_on_disk,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # The manifest accumulates one entry per pass. Earlier versions overwrote the
    # failure list in retry mode, which threw away the reason breakdown from the
    # first full pass -- the number the paper actually needs to quote for the
    # recovery rate. Passes are append-only now, and the surviving failure list is
    # merged rather than replaced.
    prev: dict = {}
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    on_disk = {p.stem for p in out_dir.glob("*.wav")}
    seen = {f["ytid"] for f in failures}
    merged = failures + [f for f in prev.get("failures", [])
                         if f["ytid"] not in seen and f["ytid"] not in on_disk]
    cumulative: dict[str, int] = {}
    for f in merged:
        cumulative[f["reason"]] = cumulative.get(f["reason"], 0) + 1

    manifest = {
        "nominal_clips": int(len(df)),
        "total_wav_on_disk": total_on_disk,
        "recovery_rate_vs_nominal": round(total_on_disk / max(1, len(df)), 4),
        "still_missing": int(len(df) - total_on_disk),
        "sample_rate": TARGET_SR,
        "outstanding_failure_reasons": cumulative,
        "passes": prev.get("passes", []) + [this_pass],
        "failures": merged,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    log.info("=" * 70)
    log.info("recovered %d / %d nominal clips (%.1f%%) in %s",
             total_on_disk, len(df), 100 * total_on_disk / max(1, len(df)),
             human_time(time.time() - t0))
    log.info("this pass: %s", json.dumps(reasons))
    log.info("outstanding: %s", json.dumps(cumulative))
    log.info("manifest -> %s", manifest_path)


if __name__ == "__main__":
    main()
