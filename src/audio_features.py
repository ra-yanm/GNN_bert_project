"""Audio front end: log-mel / chroma / MFCC extraction and segmentation.

Implements the spec's preprocessing pipeline (section 3):

  1. resample to 22,050 Hz
  2. log-mel spectrogram (128 bins) and chroma (12 bins), normalised per track
  3. split into fixed windows (5-10 s) or beat-synchronous segments via librosa

The per-segment feature vector returned by :func:`segment_features` becomes
``h_i^(0)`` -- the initial node features for the GNN in Tasks 2-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .utils import Config, get_logger

log = get_logger("audio_features")

# 24 major/minor triad templates + a no-chord symbol. Used by graph_builder to
# label segments for the chord-transition graph.
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_LABELS = [f"{p}:maj" for p in PITCH_CLASSES] + [f"{p}:min" for p in PITCH_CLASSES] + ["N"]


@dataclass
class TrackFeatures:
    """Everything downstream code needs from one audio file."""

    track_id: str
    mel: np.ndarray          # (n_mels, T)  log-power, per-track normalised
    chroma: np.ndarray       # (12, T)      CQT chroma, per-frame L1-normalised
    mfcc: np.ndarray         # (n_mfcc, T)
    sr: int
    duration: float
    hop_length: int

    @property
    def n_frames(self) -> int:
        return self.mel.shape[1]


# --------------------------------------------------------------------------- #
# Loading + framewise features
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, cfg: Config) -> tuple[np.ndarray, int]:
    """Load and resample. ``clip_seconds`` caps the read so a stray long file
    cannot blow up memory or skew graph sizes."""
    a = cfg.audio
    y, sr = librosa.load(
        str(path),
        sr=a["sample_rate"],
        mono=a.get("mono", True),
        duration=a.get("clip_seconds") or None,
    )
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")
    # Trim DC offset; leading silence is informative for segment graphs so we
    # deliberately do NOT trim silence here.
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), sr


def extract_features(path: str | Path, cfg: Config, track_id: str | None = None) -> TrackFeatures:
    """Framewise log-mel, chroma and MFCC for one file."""
    a = cfg.audio
    y, sr = load_audio(path, cfg)
    n_fft, hop = a["n_fft"], a["hop_length"]

    mel_power = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=a["n_mels"], fmin=a["fmin"], fmax=min(a["fmax"], sr // 2),
        power=2.0,
    )
    mel = librosa.power_to_db(mel_power, ref=np.max)          # (n_mels, T), dB

    # CQT chroma is the right choice for harmony: it is pitch-aligned, so the
    # 12 bins really are pitch classes rather than smeared FFT bins.
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=a["n_chroma"])
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_power), n_mfcc=a["n_mfcc"])

    if a.get("normalize") == "per_track":
        # Spec: "normalize per track". Standardise mel/mfcc over time; chroma is
        # L1-normalised per frame instead, which preserves its simplex geometry
        # and is what chord template matching expects.
        mel = _standardise(mel)
        mfcc = _standardise(mfcc)
        chroma = chroma / (np.linalg.norm(chroma, ord=1, axis=0, keepdims=True) + 1e-8)

    return TrackFeatures(
        track_id=track_id or Path(path).stem,
        mel=mel.astype(np.float32),
        chroma=chroma.astype(np.float32),
        mfcc=mfcc.astype(np.float32),
        sr=sr,
        duration=float(len(y) / sr),
        hop_length=hop,
    )


def _standardise(x: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance per feature row, across the whole track."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + 1e-8)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def segment_bounds(tf: TrackFeatures, cfg: Config) -> np.ndarray:
    """Frame indices delimiting segments -> array of shape (n_segments + 1,).

    ``segment_mode: fixed`` gives overlapping fixed windows; ``beat`` gives
    beat-synchronous segments grouped into bars (librosa beat tracker).
    """
    g = cfg.graph
    mode = g.get("segment_mode", "fixed")
    n_frames = tf.n_frames
    fps = tf.sr / tf.hop_length

    if mode == "beat":
        bounds = _beat_bounds(tf, cfg, fps, n_frames)
        if bounds is not None:
            return bounds
        log.debug("%s: beat tracking gave too few beats -- falling back to fixed", tf.track_id)

    win = max(1, int(round(g["segment_seconds"] * fps)))
    hop = max(1, int(round(g.get("segment_hop_seconds", g["segment_seconds"]) * fps)))
    starts = np.arange(0, max(1, n_frames - win + 1), hop, dtype=int)
    if starts.size == 0:
        starts = np.array([0], dtype=int)

    max_nodes = g.get("max_nodes", 64)
    if starts.size > max_nodes:
        # Subsample uniformly rather than truncating, so a long track is still
        # represented across its whole duration.
        starts = starts[np.linspace(0, starts.size - 1, max_nodes).round().astype(int)]

    ends = np.minimum(starts + win, n_frames)
    return np.stack([starts, ends], axis=1)


def _beat_bounds(tf: TrackFeatures, cfg: Config, fps: float, n_frames: int) -> np.ndarray | None:
    """Beat-synchronous segments, grouped into groups of 4 beats (~1 bar)."""
    try:
        # onset_strength on the mel we already have avoids a second STFT
        onset_env = librosa.onset.onset_strength(S=tf.mel, sr=tf.sr, hop_length=tf.hop_length)
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=tf.sr, hop_length=tf.hop_length, units="frames"
        )
    except Exception as exc:                                   # pragma: no cover
        log.debug("%s: beat_track failed (%s)", tf.track_id, exc)
        return None

    g = cfg.graph
    if beats is None or len(beats) < 8:
        return None
    grouped = beats[::4]
    if len(grouped) < g.get("min_nodes", 4) + 1:
        return None
    grouped = np.concatenate([grouped, [n_frames]])
    bounds = np.stack([grouped[:-1], grouped[1:]], axis=1)
    bounds = bounds[bounds[:, 1] > bounds[:, 0]]
    max_nodes = g.get("max_nodes", 64)
    if bounds.shape[0] > max_nodes:
        bounds = bounds[np.linspace(0, bounds.shape[0] - 1, max_nodes).round().astype(int)]
    return bounds if bounds.shape[0] >= g.get("min_nodes", 4) else None


def segment_features(tf: TrackFeatures, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool framewise features inside each segment.

    Returns
    -------
    node_x : (n_segments, D) float32
        Concatenated per-segment statistics -- the GNN's ``h_i^(0)``.
    seg_chroma : (n_segments, 12) float32
        Mean chroma per segment, kept separately for chord labelling.
    bounds : (n_segments, 2) int
        [start_frame, end_frame) for each segment, for case-study plots.
    """
    bounds = segment_bounds(tf, cfg)
    which = cfg.graph.get("node_features", ["mel_stats", "chroma_stats", "mfcc_stats"])

    rows: list[np.ndarray] = []
    chromas: list[np.ndarray] = []
    for s, e in bounds:
        e = max(int(e), int(s) + 1)
        s = int(s)
        parts: list[np.ndarray] = []
        mel_w, chr_w, mfc_w = tf.mel[:, s:e], tf.chroma[:, s:e], tf.mfcc[:, s:e]
        if "mel_stats" in which:
            parts += [mel_w.mean(1), mel_w.std(1)]
        if "chroma_stats" in which:
            parts += [chr_w.mean(1), chr_w.std(1)]
        if "mfcc_stats" in which:
            # delta captures within-segment motion, which plain means wash out
            parts += [mfc_w.mean(1), mfc_w.std(1), _delta_mean(mfc_w)]
        rows.append(np.concatenate(parts))
        chromas.append(chr_w.mean(1))

    node_x = np.nan_to_num(np.stack(rows), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    seg_chroma = np.stack(chromas).astype(np.float32)
    return node_x, seg_chroma, bounds.astype(np.int32)


def _delta_mean(w: np.ndarray) -> np.ndarray:
    if w.shape[1] < 2:
        return np.zeros(w.shape[0], dtype=np.float32)
    return np.abs(np.diff(w, axis=1)).mean(axis=1)


def node_feature_dim(cfg: Config) -> int:
    """Dimension of ``h^(0)`` implied by config -- lets models size themselves
    without touching audio."""
    which = cfg.graph.get("node_features", ["mel_stats", "chroma_stats", "mfcc_stats"])
    d = 0
    if "mel_stats" in which:
        d += 2 * cfg.audio["n_mels"]
    if "chroma_stats" in which:
        d += 2 * cfg.audio["n_chroma"]
    if "mfcc_stats" in which:
        d += 3 * cfg.audio["n_mfcc"]
    return d


# --------------------------------------------------------------------------- #
# Mel patch for the CNN baseline (spec baseline B2)
# --------------------------------------------------------------------------- #
def mel_patch(tf: TrackFeatures, n_frames: int = 640) -> np.ndarray:
    """Fixed-size (n_mels, n_frames) log-mel patch, centre-cropped or padded.
    This is the *only* input the CNN baseline sees -- no graph, no text."""
    mel = tf.mel
    T = mel.shape[1]
    if T >= n_frames:
        start = (T - n_frames) // 2
        return mel[:, start:start + n_frames].copy()
    pad = n_frames - T
    left = pad // 2
    return np.pad(mel, ((0, 0), (left, pad - left)), mode="edge")


def chord_sequence(seg_chroma: np.ndarray, cfg: Config) -> list[str]:
    """Label each segment with its best-matching major/minor triad.

    Binary triad templates are correlated against the segment's mean chroma;
    the argmax wins. A segment whose chroma is near-flat (low energy contrast)
    is labelled "N" (no chord) rather than forced into a triad.
    """
    templates, labels = _chord_templates(cfg)
    out: list[str] = []
    for vec in seg_chroma:
        v = vec - vec.mean()
        denom = np.linalg.norm(v)
        if denom < 1e-6:
            out.append("N")
            continue
        scores = templates @ (v / denom)
        best = int(np.argmax(scores))
        out.append(labels[best] if scores[best] > 0.20 else "N")
    return out


_TEMPLATE_CACHE: dict[str, tuple[np.ndarray, list[str]]] = {}


def _chord_templates(cfg: Config) -> tuple[np.ndarray, list[str]]:
    key = cfg.dotted("graph.chord.template_set", "majmin")
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]
    maj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)  # root, M3, P5
    min_ = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)  # root, m3, P5
    rows, labels = [], []
    for shift, pc in enumerate(PITCH_CLASSES):
        rows.append(np.roll(maj, shift)); labels.append(f"{pc}:maj")
    for shift, pc in enumerate(PITCH_CLASSES):
        rows.append(np.roll(min_, shift)); labels.append(f"{pc}:min")
    T = np.stack(rows)
    T = T - T.mean(axis=1, keepdims=True)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    _TEMPLATE_CACHE[key] = (T, labels)
    return T, labels
