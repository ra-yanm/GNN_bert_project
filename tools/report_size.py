r"""Estimate the compiled page count of report.tex without a TeX toolchain.

The specification asks for a 6--10 page report. A real compile is now the authority on
length -- see report/README.md -- but a full Tectonic run takes long enough that trimming
a 12-page draft one section at a time is tedious, and this gives an answer in
milliseconds. Use it to iterate, then confirm with the compiler.

The estimate is deliberately crude and stated as a range: prose words divided by
a words-per-page rate for the document's geometry, plus a fixed allowance per
float. It is calibrated to this document's preamble (11pt, one column, 1 in
margins, \small tables), not to LaTeX in general. Treat it as "roughly how many
pages", not as a page count.

    python tools/report_size.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "report" / "report.tex"

# For 11pt single-column text in a 1 in margin geometry, a full page of prose with
# no floats runs about this many words. Headings, list bullets and paragraph breaks
# all cost vertical space that words do not account for, hence the lower bound.
WORDS_PER_PAGE = (430, 520)
# A \resfig at 0.9\linewidth plus caption; a \restab at \small with 6--10 rows.
FIG_PAGES = 0.34
TAB_PAGES = 0.26
EQ_PAGES = 0.035


def main() -> int:
    if not TEX.exists():
        print(f"missing {TEX}")
        return 1
    tex = TEX.read_text(encoding="utf-8")
    body = tex.partition(r"\begin{document}")[2]
    body = body.partition(r"\end{document}")[0]

    counts = {
        "sections": len(re.findall(r"^\\section\{", body, flags=re.M)),
        "subsections": len(re.findall(r"^\\subsection\{", body, flags=re.M)),
        "figures": len(re.findall(r"\\resfig\{", body)),
        "tables": len(re.findall(r"\\restab\{", body)),
        "equations": len(re.findall(r"\\begin\{equation\}", body)),
        "references": len(re.findall(r"\\bibitem", body)),
    }

    # Prose only: floats carry their own allowance, and their captions are counted
    # with them rather than as body text.
    prose = body.partition(r"\begin{thebibliography}")[0]
    prose = re.sub(r"\\resfig\{[^{}]*\}\{[^{}]*\}\{(?:[^{}]|\{[^{}]*\})*\}", " ", prose)
    prose = re.sub(r"\\restab\{[^{}]*\}", " ", prose)
    prose = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", prose, flags=re.S)
    prose = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", " ", prose, flags=re.S)
    prose = re.sub(r"%.*$", " ", prose, flags=re.M)
    # A macro expands to a number or a short word, so it is worth about one word.
    prose = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " x ", prose)
    prose = re.sub(r"[{}$&~\\]", " ", prose)
    words = len(prose.split())

    float_pages = (counts["figures"] * FIG_PAGES + counts["tables"] * TAB_PAGES
                   + counts["equations"] * EQ_PAGES)
    lo = words / WORDS_PER_PAGE[1] + float_pages
    hi = words / WORDS_PER_PAGE[0] + float_pages

    for k, v in counts.items():
        print(f"{k:14s} {v}")
    print(f"{'prose words':14s} {words:,}")
    print(f"\nprose alone      {words / WORDS_PER_PAGE[1]:.1f}--{words / WORDS_PER_PAGE[0]:.1f} pages")
    print(f"floats add       {float_pages:.1f} pages "
          f"({counts['figures']} fig x {FIG_PAGES}, {counts['tables']} tab x {TAB_PAGES})")
    print(f"estimated total  {lo:.0f}--{hi:.0f} pages, plus title and references")
    print("\nSpecification asks for 6--10 pages of main matter.")
    if lo > 10:
        print("=> over the target. Moving figures/tables into the appendix is the "
              "cheapest fix; the appendix is not counted as main matter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
