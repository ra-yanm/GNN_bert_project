"""Shared utilities: config loading, seeding, logging, device handling.

Every entry point (train.py, evaluate.py, the notebooks) goes through
``load_config`` and ``set_seed`` so that a run is reproducible from
``config.yaml`` + the seed alone.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Repo root = parent of src/. Everything in config.yaml is relative to this.
ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config(dict):
    """dict with attribute access and dotted lookup, so both ``cfg["audio"]``
    and ``cfg.audio.sample_rate`` and ``cfg.get_path("audio.n_mels")`` work."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def dotted(self, path: str, default: Any = None) -> Any:
        """``cfg.dotted("model.gnn.hidden_dim")`` -> 256."""
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(raw)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(cfg: Config, key: str) -> Path:
    """Turn a ``paths.*`` config entry into an absolute path, creating it."""
    rel = cfg.dotted(f"paths.{key}")
    if rel is None:
        raise KeyError(f"paths.{key} not in config")
    out = ROOT / rel
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 425) -> None:
    """Seed every RNG we touch. Also pins cuDNN to deterministic kernels so a
    GPU rerun matches; on CPU this is a no-op but harmless."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Config | None = None) -> torch.device:
    want = (cfg or {}).get("device", "cpu") if cfg else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        logging.warning("config asks for cuda but it is unavailable -- using cpu")
        want = "cpu"
    return torch.device(want)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts -- reported in the paper's model table."""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"


def setup_logging(level: int = logging.INFO, logfile: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=level, format=_LOG_FMT, datefmt="%H:%M:%S", handlers=handlers, force=True
    )
    # librosa/numba and HF are extremely chatty at INFO
    for noisy in ("numba", "matplotlib", "urllib3", "filelock", "transformers",
                  "httpx", "httpcore", "huggingface_hub", "PIL", "fontTools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Metrics IO -- results/metrics.json is a single merged dict keyed by run name,
# so re-running one task never clobbers another task's numbers.
# --------------------------------------------------------------------------- #
def save_metrics(cfg: Config, run_name: str, payload: dict) -> Path:
    out = resolve(cfg, "results") / "metrics.json"
    blob: dict = {}
    if out.exists():
        try:
            blob = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("metrics.json was corrupt -- starting a fresh one")
    blob[run_name] = payload
    out.write_text(json.dumps(blob, indent=2, default=_jsonable), encoding="utf-8")
    return out


def load_metrics(cfg: Config) -> dict:
    out = resolve(cfg, "results") / "metrics.json"
    if not out.exists():
        return {}
    return json.loads(out.read_text(encoding="utf-8"))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
@dataclass
class EarlyStopper:
    """Stop when the monitored metric has not improved for ``patience`` epochs.
    Tracks the best state_dict so the reported test number always comes from the
    best-val checkpoint, never the last epoch."""

    patience: int = 10
    mode: str = "max"
    min_delta: float = 1e-4
    best: float = float("nan")
    bad_epochs: int = 0
    best_state: dict | None = None
    best_epoch: int = -1

    def __post_init__(self) -> None:
        self.best = -float("inf") if self.mode == "max" else float("inf")

    def _improved(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, model: torch.nn.Module, epoch: int) -> bool:
        """Returns True if training should stop."""
        if self._improved(value):
            self.best = value
            self.bad_epochs = 0
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")
