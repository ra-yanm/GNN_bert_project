"""Dataset and collate plumbing for graph-only, graph+text, and paired tasks.

Three dataset classes, all reading the cached ``.pt`` graph lists produced by
``scripts/preprocess.py``:

* :class:`GraphDataset`       -- Task 2: graphs + single-label genre targets.
* :class:`GraphTextDataset`   -- Task 3: graphs + tokenised text + multi-label
  tags + optional masked valence/arousal.
* :class:`PairedGraphTextDataset` -- Task 4: (graph, caption) pairs for InfoNCE.

Collation uses PyG's ``Batch.from_data_list`` for the graph side and plain
stacking for the text side; the custom ``collate`` functions keep the two aligned
and carry the per-sample masks the multi-task loss needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .utils import Config, ROOT, get_logger

log = get_logger("graph_dataset")


# --------------------------------------------------------------------------- #
# Cache IO
# --------------------------------------------------------------------------- #
def load_cached_graphs(cfg: Config, dataset: str, graph_kind: str = "segment") -> list[Data]:
    path = ROOT / cfg.dotted("paths.processed") / f"{dataset}_{graph_kind}_graphs.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run: python scripts/preprocess.py --dataset {dataset} "
            f"--graph-kind {graph_kind}"
        )
    # weights_only=False: these are PyG Data objects with python attrs, not tensors
    graphs = torch.load(path, weights_only=False)
    log.info("loaded %d %s/%s graphs from cache", len(graphs), dataset, graph_kind)
    return graphs


def load_cached_mels(cfg: Config, dataset: str, graph_kind: str = "segment") -> dict[str, torch.Tensor]:
    path = ROOT / cfg.dotted("paths.processed") / f"{dataset}_{graph_kind}_mel.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- rerun preprocess.py with --save-mel"
        )
    return torch.load(path, weights_only=False)


def filter_split(graphs: list[Data], split: str) -> list[Data]:
    return [g for g in graphs if str(getattr(g, "split", "")) == split]


def ablate_edges(graphs: list[Data], keep: str) -> list[Data]:
    """Return copies of ``graphs`` retaining only one class of edge.

    ``keep`` is ``"temporal"``, ``"similarity"``, or ``"none"``. This is the
    sharpest available test of whether graph *structure* contributes: the node
    features, the encoder, and the parameter count are all held fixed, and only
    the message-passing topology changes. ``"none"`` leaves self-loops only, which
    reduces the GNN to a per-node MLP followed by mean pooling.

    Edge kind is read from ``edge_attr`` column 1 (is_temporal) and 2
    (is_similarity), as written by :func:`graph_builder.build_segment_graph`.
    """
    if keep not in {"temporal", "similarity", "none"}:
        raise ValueError(f"keep must be temporal|similarity|none, got {keep!r}")
    col = {"temporal": 1, "similarity": 2}.get(keep)

    out: list[Data] = []
    for g in graphs:
        d = g.clone()
        n = int(d.num_nodes)
        if col is None or d.edge_attr is None or d.edge_attr.shape[1] <= col:
            m = torch.zeros(d.edge_index.shape[1], dtype=torch.bool)
        else:
            m = d.edge_attr[:, col] > 0.5
        if not bool(m.any()):
            # keep the graph well-formed: an empty edge_index makes SAGEConv
            # return zeros for every node, which is not the ablation we mean
            loop = torch.arange(n, dtype=torch.long)
            d.edge_index = torch.stack([loop, loop])
            d.edge_attr = torch.zeros((n, d.edge_attr.shape[1] if d.edge_attr is not None else 3))
            d.edge_weight = torch.ones(n)
        else:
            d.edge_index = d.edge_index[:, m]
            d.edge_attr = d.edge_attr[m]
            if getattr(d, "edge_weight", None) is not None:
                d.edge_weight = d.edge_weight[m]
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# Task 2: graphs only
# --------------------------------------------------------------------------- #
class GraphDataset(Dataset):
    """Graphs with an integer ``y`` target."""

    def __init__(self, graphs: list[Data]):
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> Data:
        return self.graphs[i]

    @property
    def labels(self) -> np.ndarray:
        return np.array([int(g.y.item()) for g in self.graphs])


def collate_graphs(items: list[Data]) -> Batch:
    # exclude the python-object attrs; PyG cannot collate lists of strings
    return Batch.from_data_list(items, exclude_keys=["chords", "chord_vocab", "chord_seq"])


class MelDataset(Dataset):
    """Mel patches + labels for the CNN baseline (B2). Standardisation stats are
    computed on train only and passed in, so the baseline gets the same
    no-leakage treatment as the GNN."""

    def __init__(self, mels: dict[str, torch.Tensor], graphs: list[Data],
                 stats: tuple[float, float] | None = None):
        self.items = [(mels[str(g.track_id)], int(g.y.item())) for g in graphs
                      if str(g.track_id) in mels]
        if len(self.items) < len(graphs):
            log.warning("MelDataset: %d/%d graphs had a cached mel patch",
                        len(self.items), len(graphs))
        if stats is None:
            allm = torch.stack([m for m, _ in self.items])
            stats = (float(allm.mean()), float(allm.std()) + 1e-8)
        self.mu, self.sd = stats

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        m, y = self.items[i]
        return (m - self.mu) / self.sd, y

    @property
    def stats(self) -> tuple[float, float]:
        return self.mu, self.sd

    @property
    def labels(self) -> np.ndarray:
        return np.array([y for _, y in self.items])


# --------------------------------------------------------------------------- #
# Task 3: graphs + text + multi-label tags (+ optional emotion)
# --------------------------------------------------------------------------- #
class GraphTextDataset(Dataset):
    """One item = one graph, its tokenised text, its multi-hot tags, and
    (where available) valence/arousal with a validity mask.

    ``texts`` is a parallel list of strings, ``y`` a (N, K) float array. Emotion
    targets are optional; samples lacking them get ``mask=0`` so the multi-task
    loss skips them rather than regressing toward zero.
    """

    def __init__(
        self,
        graphs: list[Data],
        texts: list[str],
        y: np.ndarray,
        tokenizer,
        max_length: int = 192,
        valence: np.ndarray | None = None,
        arousal: np.ndarray | None = None,
        y_mask: np.ndarray | None = None,
    ):
        assert len(graphs) == len(texts) == len(y), "graphs/texts/y must align"
        self.graphs = graphs
        self.texts = list(texts)
        enc = tokenizer(self.texts, truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32)

        n = len(graphs)
        # y_mask marks rows whose tag labels are real. Rows from a corpus without
        # tag annotation get 0 so MultiTaskLoss skips them instead of reading the
        # all-zero row as "none of these 50 tags apply".
        self.y_mask = torch.tensor(
            np.ones(n, bool) if y_mask is None else np.asarray(y_mask, bool), dtype=torch.bool)
        self.valence = torch.tensor(
            np.nan_to_num(valence, nan=0.0) if valence is not None else np.zeros(n),
            dtype=torch.float32)
        self.arousal = torch.tensor(
            np.nan_to_num(arousal, nan=0.0) if arousal is not None else np.zeros(n),
            dtype=torch.float32)
        self.v_mask = torch.tensor(
            ~np.isnan(valence) if valence is not None else np.zeros(n, bool), dtype=torch.bool)
        self.a_mask = torch.tensor(
            ~np.isnan(arousal) if arousal is not None else np.zeros(n, bool), dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> dict:
        return {
            "graph": self.graphs[i],
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "y": self.y[i], "y_mask": self.y_mask[i],
            "valence": self.valence[i], "arousal": self.arousal[i],
            "valence_mask": self.v_mask[i], "arousal_mask": self.a_mask[i],
            "idx": i,
        }


def collate_graph_text(items: list[dict]) -> dict:
    return {
        "graph": collate_graphs([it["graph"] for it in items]),
        "input_ids": torch.stack([it["input_ids"] for it in items]),
        "attention_mask": torch.stack([it["attention_mask"] for it in items]),
        "y": torch.stack([it["y"] for it in items]),
        "y_mask": torch.stack([it["y_mask"] for it in items]),
        "valence": torch.stack([it["valence"] for it in items]),
        "arousal": torch.stack([it["arousal"] for it in items]),
        "valence_mask": torch.stack([it["valence_mask"] for it in items]),
        "arousal_mask": torch.stack([it["arousal_mask"] for it in items]),
        "idx": torch.tensor([it["idx"] for it in items], dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Task 4: (graph, caption) pairs
# --------------------------------------------------------------------------- #
class PairedGraphTextDataset(Dataset):
    """Contrastive pairs. ``group_ids`` marks captions that are byte-identical so
    InfoNCE can avoid treating a true duplicate as a negative."""

    def __init__(self, graphs: list[Data], captions: list[str], tokenizer,
                 max_length: int = 192, ids: list[str] | None = None):
        assert len(graphs) == len(captions)
        self.graphs = graphs
        self.captions = list(captions)
        self.ids = list(ids) if ids is not None else [str(g.track_id) for g in graphs]
        enc = tokenizer(self.captions, truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

        uniq: dict[str, int] = {}
        self.group_ids = torch.tensor(
            [uniq.setdefault(c.strip().lower(), len(uniq)) for c in self.captions],
            dtype=torch.long,
        )
        n_dup = len(self.captions) - len(uniq)
        if n_dup:
            log.info("PairedGraphTextDataset: %d duplicate captions masked from negatives", n_dup)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, i: int) -> dict:
        return {
            "graph": self.graphs[i],
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "group_id": self.group_ids[i],
            "id": self.ids[i],
            "caption": self.captions[i],
            "idx": i,
        }


def collate_paired(items: list[dict]) -> dict:
    return {
        "graph": collate_graphs([it["graph"] for it in items]),
        "input_ids": torch.stack([it["input_ids"] for it in items]),
        "attention_mask": torch.stack([it["attention_mask"] for it in items]),
        "group_ids": torch.stack([it["group_id"] for it in items]),
        "ids": [it["id"] for it in items],
        "captions": [it["caption"] for it in items],
        "idx": torch.tensor([it["idx"] for it in items], dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Text synthesis for graph-only corpora (Task 3 on GTZAN)
# --------------------------------------------------------------------------- #
def metadata_caption(genre: str, rng: np.random.Generator) -> str:
    """Build a short natural-language description from a genre label.

    Why this exists: Task 3 needs *paired* audio graphs and text. GTZAN ships a
    genre label and nothing else, and MusicCaps ships captions but its audio is
    not redistributable (YouTube IDs only). Rather than silently skip the pairing,
    we template a caption from the metadata that GTZAN does have.

    This is an honest but WEAK text channel, and the report says so explicitly:
    the caption is a deterministic function of the label, so the text branch can
    reach the label without listening. Every Task 3 number on GTZAN is therefore
    reported alongside the ``bert_only`` ablation, which measures exactly that
    shortcut. The MusicCaps-based Task 4 numbers use real human captions.
    """
    templates = [
        "a {g} track", "this recording is {g} music",
        "{g} song with typical {g} instrumentation",
        "a piece in the {g} style", "{g}",
    ]
    return templates[int(rng.integers(len(templates)))].format(g=genre)


def genre_tag_matrix(graphs: list[Data], genres: list[str]) -> np.ndarray:
    """One-hot genre matrix reused as a degenerate multi-label target, so the
    same Task 3 code path serves both single-label GTZAN and true multi-label
    corpora without branching."""
    idx = {g: i for i, g in enumerate(genres)}
    y = np.zeros((len(graphs), len(genres)), dtype=np.float32)
    for r, g in enumerate(graphs):
        gi = idx.get(str(getattr(g, "genre", "")), None)
        if gi is None and getattr(g, "y", None) is not None:
            gi = int(g.y.item())
        if gi is not None:
            y[r, gi] = 1.0
    return y
