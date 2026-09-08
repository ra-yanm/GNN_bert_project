# Building the report

Two renderings of one source:

| file | what it is |
|---|---|
| `report.tex` | the source. Prose is hand-written; **every number is a macro.** |
| `numbers.tex` | generated. One `\newcommand` per scalar, from `results/metrics.json`. |
| `report.md` | generated from `report.tex`. Readable without a TeX install. |

Only `report.tex` is edited by hand. The other two are build products — regenerating
them is how the report stays in step with the runs.

## Why the numbers are macros

A report that quotes its own results in prose drifts: you re-run a model, the
metric moves, and paragraph six still claims the old figure. So no numeral is typed
into `report.tex`. `\tTwoCnnMacroFOne` is written instead of `0.5683`, and
`tools/build_report_numbers.py` defines it from `results/metrics.json`.

A macro with no measurement behind it expands to a bold **??**. That is deliberate:
an unbacked claim should be visible in the rendered PDF rather than silently
plausible. If you see `??`, the fix is to run the missing training job, not to edit
the prose.

Four guards exist because they catch things a compiler either does not check or only
warns about, and because for most of this project's life there was no TeX toolchain on
the build machine to check anything at all. All four live in
`tools/check_report_macros.py`, which exits non-zero if any of them fires:

- **Undefined macros.** Every control sequence `report.tex` uses is cross-checked
  against what `numbers.tex` defines. An undefined one is an `Undefined control
  sequence` compile error.
- **Dangling `\ref`s.** LaTeX does *not* fail on a reference with no label — it
  prints `??` and buries a warning in the log — so this one would otherwise ship.
  The `\resfig` / `\restab` helpers declare `fig:<name>` / `tab:<name>` implicitly
  and count as labels.
- **Hand-typed quantities.** Any numeral in the body that is not on a small
  allowlist (corpus definitions, notation subscripts, each with a written reason)
  is flagged. This is what enforces the claim that no result is typed by hand; it
  found a dozen violations when it was added, including a bolded 4,684 / 93.8%
  caption-leak figure that no measurement supported — the matcher the pipeline
  actually uses gives 4,671 / 93.5%.
- **Unbalanced inline math.** A single stray `$` makes TeX run past the end of the
  paragraph looking for the close, which either typesets the rest in italics or
  dies with `Missing $ inserted` several lines later. Counted per paragraph so the
  offender is localised. There was one.

One more constraint is enforced at generation time rather than checked: macro names
spell digits as words (`FOne`, not `F1`). A LaTeX control sequence is a backslash
plus **letters only** — `\tOneF1` tokenises as `\tOneF` followed by a literal `1`.
`build_report_numbers.py` raises on any name containing a non-letter.

## Regenerating

Run from the repository root, in this order:

```bash
python scripts/evaluate.py
```

```bash
python scripts/corpus_stats.py
```

```bash
python tools/build_report_numbers.py
```

```bash
python tools/check_report_macros.py
```

```bash
python tools/build_report_md.py
```

`evaluate.py` writes `results/plots/*.png` and `results/tables/*.{tex,md}`, which
the report pulls in. `corpus_stats.py` writes `results/corpus_stats.json` — the
dataset-level figures (caption leak rates, tag support, preprocessing timings) that
are properties of the corpora rather than of any model. It writes a standalone file
rather than adding to `results/metrics.json` because every trainer rewrites that
file wholesale, so a second writer would race. It must run **before**
`build_report_numbers.py`, which reads both.

Steps 4 and 5 propagate everything into the two renderings. Step 4 should report
four `OK:` lines and an empty `??` list before you treat the report as final.

## Compiling the PDF

`report.pdf` in this directory is a real compile: Tectonic 0.17.0, **18 pages**, four
overfull hboxes (two of them severe, in `task3_case_studies.tex` and
`task4_retrieval_examples.tex`). It is over the specification's 6–10 pages and still needs
cutting.

Any of `pdflatex`, `xelatex`, `lualatex`, `latexmk` or `tectonic` will build it:

```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

Twice, so `\ref` and `\cite` resolve — the first pass only records the labels. Tectonic
does both passes itself:

```bash
cd report && tectonic -X compile report.tex --keep-logs --keep-intermediates
```

**Run it from inside `report/`.** The preamble uses
`\graphicspath{{../results/plots/}}` and `\input{../results/tables/...}`, both
relative to the working directory, so compiling from the repository root will not
find the figures. Nothing breaks loudly if you do: the `\resfig` / `\restab`
helpers wrap each artefact in `\IfFileExists` and substitute a visible
"not generated yet" box, so the document compiles at any stage of the pipeline
rather than dying on a missing figure. That is convenient during a build and
misleading at the end — check the PDF for those boxes before submitting.

There is no `.bib` file and no BibTeX run: the 17 references live in a
`thebibliography` environment inside `report.tex`.

## report.md

`tools/build_report_md.py` converts the one source rather than maintaining a
second hand-written copy — a Markdown transcription would drift from the LaTeX
exactly the way `numbers.tex` exists to prevent. It is a targeted converter for the
constructs this document uses, not a general LaTeX engine, and it prints a warning
listing anything it did not recognise.

Math is left as `$...$` / `$$...$$` for MathJax, so equations render on GitHub but
not in every Markdown viewer. Figures are relative links (`../results/plots/*.png`),
which resolve when the file is viewed inside the repository.
