# GNN-BERT Music Context Understanding

**CSE425 — Neural Networks.** Four tasks over a shared audio-graph + BERT stack:
caption→tag prediction, graph-based genre classification, cross-attention fusion
for joint tagging and emotion regression, and contrastive audio↔text retrieval.

Every number reported here and in `report/` is measured on this machine. Nothing is
illustrative, nothing is copied from a paper, and where a result is weak it is
reported weak — the report keeps its negative results rather than burying them.

## What it found

Three findings drove how the report is written, and all three cut against the
project's own premise:

1. **The caption→tag task is largely substring matching.** 82.4% of MusicCaps
   captions contain a literal surface form of one of *their own* tags (93.5%
   contain some vocabulary tag), so a regular expression with no learning at all
   recovers most of what fine-tuned DistilBERT scores. Masking the tag words out
   of the captions is therefore the only defensible measurement of caption
   *understanding*, and it costs a large chunk of macro-F1. Both numbers are
   reported; the naive one is labelled as the shortcut it is.

2. **The segment graph is a net cost on genre classification.** A GraphSAGE model
   over the full temporal + similarity graph is beaten by the same model with the
   similarity edges only, and is not distinguishable from the same model with *no
   edges at all* — so message passing over this graph is worth no more than not
   passing messages. A plain mel-spectrogram CNN beats every graph variant with
   fewer parameters.

3. **The graph is coherent, and that coherence is why message passing does not
   help.** The spec's $S_{\text{graph}}$ statistic looks impressive until you add a
   random-pair control: post-ReLU node states live in the positive orthant, so at
   $\tau = 0.5$ even random pairs score 0.96. Sweeping $\tau$ and reporting the
   *lift over the control* shows similarity edges are the most coherent kind — which
   means averaging across them adds no information. Redundant neighbours are exactly
   the neighbours you gain nothing from.

Read the results in **[`report/report.md`](report/report.md)** (renders on GitHub,
no LaTeX needed) or build the PDF from `report/report.tex` — see
[`report/README.md`](report/README.md).

Numbers live in exactly one place: `results/metrics.json`, from which
`report/numbers.tex` is generated. This README deliberately quotes almost none of
them, so it cannot go stale.

## Setup

CPU-only; no GPU is required and none was used. Runs were made with 12 threads.

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

`requirements.lock.txt` pins the exact 77-package environment behind
`results/metrics.json`. On Windows, create the venv at a short path (e.g. `C:\v`) —
pip hits `WinError 206` installing torch into a deeply nested `site-packages`.

## Data

Not redistributed here; the loaders expect these under `data/raw/`:

| corpus | used for | note |
|---|---|---|
| GTZAN | Task 2 genre, Task 3 audio side | 1,000 clips; `jazz.00054` is a known-corrupt file and is dropped, giving 929 usable tracks |
| DEAM | Task 3 valence/arousal targets | 45 s excerpts with continuous annotations |
| MusicCaps | Task 1 captions, Task 4 pairs | **audio is not distributed** — only YouTube IDs |

MusicCaps is the binding constraint. Its audio must be recovered from YouTube
clip-by-clip, and only **30%** of the nominal set survived: 231 clips are
permanently gone (deleted, private, or geo-blocked) and the large majority of the
rest were refused by YouTube's automated-traffic check. Higher recovery is possible
with authenticated cookies; that was not done here, and the report states the cap
this puts on Task 4 rather than working around it.

Task 4 trains on 591 pairs against a 775-clip test set. That inversion is
deliberate: MusicCaps' published `is_audioset_eval` subset is held out as the test
split, so recovery losses fall almost entirely on the training side. A published
test set was judged worth more than a balanced one — but 591 pairs is very little
for InfoNCE, and the retrieval numbers should be read as a floor.

## Running it

```bash
python scripts/download_data.py
```

```bash
python scripts/download_musiccaps_audio.py --workers 6
```

```bash
python scripts/preprocess.py --dataset all --save-mel --export-json 24
```

```bash
python scripts/train.py --task all
```

```bash
python scripts/evaluate.py
```

```bash
python scripts/corpus_stats.py
```

Then regenerate the report — see [`report/README.md`](report/README.md).

`--task` takes `1`–`4` to run one task; `--quick` runs a smoke test with the
fewest models and epochs. `scripts/calibrate_tau.py` reproduces the
similarity-threshold study that fixes `graph.similarity_threshold` in
`config.yaml`. `scripts/corpus_stats.py` measures the dataset-level figures the
report quotes (caption leak rates, tag support, preprocessing timings) into
`results/corpus_stats.json`; it must run before the report is regenerated.
Everything is seeded (`seed: 425`).

Budget on a 12-thread CPU box, measured rather than estimated. These are whole-step
wall-clocks from `results/*.log`, so the training figures include the baselines and
the test-set evaluation each step runs, not just the gradient steps:

| Step | Wall-clock | What it covers |
|---|---|---|
| `scripts/download_data.py` | not timed cold | GTZAN + DEAM + metadata, ~3.9 GB of archives; the recorded pass only re-verified files already on disk (58 s of hashing) |
| `scripts/download_musiccaps_audio.py` | 1 h 01 m 22 s | MusicCaps audio recovery, 5,509 IDs attempted; the ceiling is YouTube's rate limiting, not the CPU |
| `scripts/preprocess.py` | 11 m 20 s | 5,167 graphs over 4,388 distinct audio files, all four graph sets |
| `scripts/calibrate_tau.py` | 1 m 42 s | the similarity-threshold sweep |
| `--task 1` | 1 h 14 m 37 s | two caption variants + four baselines |
| `--task 2` | 24 m 30 s | nine models (GNN variants, edge ablations, CNN, MLP, majority) |
| `--task 3` | 1 h 15 m 07 s | four fusion ablations at 6 epochs each |
| `--task 4` | 33 m 30 s | 30-epoch cap, early-stopped at epoch 20 |
| `scripts/evaluate.py` | 31 s | 24 figures, 9 tables, all analyses |
| `scripts/corpus_stats.py` | 5 s | corpus-level figures the report quotes |

About 4 h 43 m end to end excluding the cold corpus download, of which MusicCaps
audio recovery is the single largest block and the only one that is not CPU-bound.
Executing the two notebooks against the trained checkpoints adds a few minutes.

## Predicting on your own audio

Everything above is the *evaluation* path: it scores the models over fixed splits. To ask
what the trained checkpoints say about an arbitrary file:

```bash
python scripts/predict.py mysong.mp3
```

```bash
python scripts/predict.py mysong.mp3 --caption "a mellow jazz trio with brushed drums" --json out.json
```

Nothing is retrained. The file is featurised and turned into a segment graph by the same
two functions `preprocess.py` calls, then pushed through the saved checkpoints: GTZAN
genre from three Task 2 models, tags and valence/arousal from Task 3, and zero-shot tags
from the Task 4 contrastive space. Each prediction prints next to that model's measured
test score, because several of these heads are close to chance and a bare label would read
as far more confident than the model is.

`--caption` switches on the `cross_attention` fusion head alongside the audio-only
`gnn_only` one, so the two rows show directly what the text is worth on your file. Section
G of `demo_context.ipynb` is the same code in the notebook.

Three limits are structural. Only the first 30 s is analysed (`audio.clip_seconds` — the
window the models were trained on). Genre is GTZAN's ten classes and tags are the 50 most
frequent MusicCaps aspects, neither of them open-vocabulary, so the model must answer from
its list even when the right answer is not on it. Emotion comes back on DEAM's original
1–9 scale, de-standardised with the training statistics in `results/metrics.json`.

Formats: wav, flac, mp3, ogg and aiff all load with no system `ffmpeg`, because librosa
reads through libsndfile. m4a and aac do not — convert those to wav first.

## Layout

```
config.yaml            every hyperparameter; each run copies the resolved config
                       next to its metrics so any number traces back to settings
src/
  audio_features.py    mel / chroma / MFCC, per-track normalisation
  graph_builder.py     segment graphs (temporal + cosine-similarity edges) and
                       chord-transition graphs
  graph_dataset.py     cached PyG datasets, canonical key set for batching
  gnn_model.py         GraphSAGE / GAT encoders, mean readout
  bert_encoder.py      DistilBERT wrapper, CLS pooling
  fusion_model.py      cross-attention / concat / gated / single-modality
  contrastive.py       InfoNCE, symmetric variant, retrieval metrics
  trainer.py           one training loop for all four tasks
  metrics.py           macro/micro F1, AUC-PR, MAE, R², Pearson
scripts/               download, preprocess, train, evaluate, calibrate_tau, predict
tools/                 report generation and notebook builders
notebooks/             eda.ipynb (data + graph analysis), demo_context.ipynb
report/                report.tex, generated numbers.tex, generated report.md
results/               metrics.json, plots/, tables/, checkpoints/, graph_samples/
```

`results/graph_samples/` holds 83 exported graphs as JSON, against the
specification's requirement of at least 20.

## Honest limitations

- Task 3's four ablations were trained for 6 epochs each, not to convergence — 30
  epochs × 4 modes would have been ~13 h. The learning curves show whether the loss
  was still falling; where it was, the report calls the numbers lower bounds and
  declines to read an architectural conclusion off the ordering.
- Task 4 is data-starved (591 pairs), as above.
- No hyperparameter search. Values in `config.yaml` are reasoned about in comments
  and, where it mattered most (the similarity threshold τ), calibrated against
  measurements — but not tuned on a validation sweep.
- Single seed per configuration. No error bars, so small differences between
  ablations should not be over-read.
- The report is over length. `report/report.pdf` compiles (Tectonic 0.17, 18 pages, two
  minor overfull hboxes) but the specification asks for 6–10 pages, so it still needs
  cutting —
  the two levers are a two-column layout and moving most of the 16 figures and 9 tables
  into an appendix. `tools/check_report_macros.py` guards the four failure modes a
  compiler will not catch on its own: undefined macros, `\ref`s pointing at no label,
  quantities typed into the prose instead of coming from a measurement, and unbalanced
  inline `$`.
- Task 4's human evaluation is not done. The specification asks for at least five
  listeners rating retrieved clips on a 1–5 scale; every other Task 4 requirement is
  implemented and measured, but that one needs people.
