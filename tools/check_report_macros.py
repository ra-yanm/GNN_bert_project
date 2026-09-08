"""Cross-check every macro report.tex uses against what numbers.tex defines.

There is no TeX toolchain on this build machine, so ``pdflatex`` cannot be used to
catch an undefined control sequence. That failure mode is easy to hit and easy to
miss: numbers.tex is generated, its macro names are derived from metric keys, and
a renamed key silently turns a number in the prose into ``Undefined control
sequence''. This script is the substitute for the compiler on that one point.

It reports two things:
  * macros used in report.tex but defined nowhere  -> would fail to compile
  * macros defined in numbers.tex but never used   -> harmless, but usually means
    a result was measured and then not reported

Usage:
    python tools/check_report_macros.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "report.tex"
NUMBERS = ROOT / "report" / "numbers.tex"

# Control sequences that come from LaTeX itself or from the packages report.tex
# loads. Anything here is assumed to exist; anything not here and not defined in
# our own two files is flagged.
KNOWN = set("""
documentclass usepackage begin end input include includegraphics graphicspath
newcommand renewcommand providecommand def let relax expandafter csname endcsname
title author date maketitle thanks abstract section subsection subsubsection
paragraph appendix label ref pageref cite bibitem thebibliography footnote
textbf textit texttt emph textsc textsf textrm underline sout
small footnotesize scriptsize tiny large Large LARGE huge Huge normalsize
normalfont itshape bfseries ttfamily rmfamily sffamily
item itemize enumerate description itemsep parskip parindent baselineskip
setlength addtolength vspace hspace hfill vfill quad qquad noindent indent
newline linebreak newpage clearpage pagebreak par bigskip medskip smallskip
centering raggedright raggedleft center flushleft flushright
table tabular figure caption captionsetup toprule midrule bottomrule cmidrule
multicolumn multirow hline cline
frac dfrac tfrac sqrt sum prod int lim log exp sin cos tan max min sup inf
mathbb mathbf mathrm mathcal mathit mathsf operatorname operatornamewithlimits
sigma alpha beta gamma delta epsilon varepsilon zeta eta theta iota kappa lambda
mu nu xi pi rho tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
times cdot cdots ldots dots vdots ddots approx neq leq geq ll gg equiv sim simeq
propto in notin subset supset cup cap emptyset forall exists neg land lor to
rightarrow leftarrow Rightarrow leftrightarrow mapsto lesssim gtrsim pm mp
left right big Big bigg Bigg langle rangle lVert rVert lfloor rfloor
mathbf top bot prime dag ddag
equation align gather split aligned array cases eqnarray nonumber
text intertext mbox fbox framebox parbox makebox raisebox rule
color textcolor colorbox definecolor
url href hyperref nolinkurl path
verbatim verb
ignorespaces protect string space empty
IfFileExists
linewidth textwidth columnwidth textheight paperwidth
best resfig restab
v hat a t
""".split())

# Single non-letter control sequences (\%, \&, \\, \_, \$, \{, \}, \#, \~, \^)
# are always valid and are stripped before matching.
USE_RE = re.compile(r"\\([A-Za-z]+)")
DEF_RE = re.compile(r"\\newcommand\s*\{?\\([A-Za-z]+)")


def strip_comments(tex: str) -> str:
    """Drop % comments so commented-out macros do not count as used."""
    out = []
    for line in tex.splitlines():
        i, esc = 0, False
        cut = len(line)
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                cut = i
                break
            else:
                esc = False
            i += 1
        out.append(line[:cut])
    return "\n".join(out)


# Numerals that are legitimately literal in the body text, with the reason each is
# not a measurement. Anything else that looks like a quantity is flagged: the report
# claims no result is typed by hand, and that claim needs enforcing rather than
# remembering. (It was violated in a dozen places before this check existed --
# including a bolded 4,684 / 93.8% that no measurement supported.)
LITERAL_OK: dict[str, str] = {
    # corpus definitions, not results
    "10": "ten GTZAN genres / 10-second MusicCaps clips",
    "100": "100 clips per GTZAN genre",
    "30": "30-second GTZAN clips",
    "45": "45-second DEAM excerpts",
    "12": "12 chroma bins / 12-thread CPU",
    "20": "20 MFCCs / the spec's 20-graph export floor",
    "24": "24 exported triads",
    "50": "50% segment overlap",
    "6": "DistilBERT is a 6-layer distillation",
    "2048": "STFT window, samples",
    "512": "STFT hop, samples",
    "1.5": "MusicCaps per-corpus window override, seconds",
    "0.75": "MusicCaps per-corpus hop override, seconds",
    "1.0": "standardised-target baseline MAE, by construction",
    "0.5": "chance AUC-ROC, by definition",
    "0.02": "the seed-noise threshold the report declines to read below",
    "1": "N-1 in-batch negatives; Russell's unit circle",
    "2": "two encoders / two passes / order-of-magnitude prose",
    "3": "three findings, three to five orders of magnitude",
    "4": "four tasks",
    "5": "five orders of magnitude",
    "425": "the seed, quoted in the reproducibility appendix",
    # mathematical notation: subscripts, dimensions and indices, not measurements
    "25": "the p_25 subscript on the cosine-percentile list",
    "90": "the p_90 subscript",
    "99": "the p_99 subscript",
    "768": "DistilBERT's hidden width, fixed by the checkpoint",
    "0": "the l = 0 lower bound of the message-passing layer index",
}
# Contexts whose numerals are typography or file plumbing, never claims.
SKIP_ARG_RE = re.compile(
    r"\\(?:label|ref|pageref|cite[a-z]*|input|include(?:graphics)?|resfig|restab|"
    r"itemsep|vspace|hspace|setlength|addtolength|multicolumn|multirow|parbox|"
    r"makebox|raisebox|rule|documentclass|usepackage|geometry|fbox)"
    r"(?:\[[^\]]*\])?(?:\{[^{}]*\})*"
)
NUM_RE = re.compile(r"\d+(?:\{,\}\d{3})*(?:\.\d+)?")


def check_literals(report: str) -> int:
    """Flag numerals typed into the body that should be macros from a measurement."""

    def blank(m: re.Match) -> str:
        # Preserve the newline count so reported line numbers stay true to the file.
        return "\n" * m.group(0).count("\n")

    marker = r"\begin{document}"
    head, _, body = report.partition(marker)
    if not body:
        return 0
    offset = head.count("\n") + marker.count("\n") + 1

    # The bibliography is years and arXiv identifiers, none of them measurements.
    body = body.partition(r"\begin{thebibliography}")[0]
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", blank, body, flags=re.S)
    body = re.sub(r"\\begin\{tabular\}\{[^}]*\}", blank, body)
    # \texttt is filenames, config keys and code identifiers -- jazz.00054, not a result.
    body = re.sub(r"\\texttt\{(?:[^{}]|\{[^{}]*\})*\}", blank, body, flags=re.S)
    # Equation bodies define the method; their subscripts and exponents are notation.
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", blank, body, flags=re.S)
    body = re.sub(r"\\begin\{align\}.*?\\end\{align\}", blank, body, flags=re.S)

    offenders: list[tuple[str, int]] = []
    for i, line in enumerate(body.splitlines(), offset):
        line = SKIP_ARG_RE.sub(" ", line)
        for tok in NUM_RE.findall(line):
            # Anything carrying a thousands separator is a count, never typography.
            if "{,}" in tok or tok not in LITERAL_OK:
                offenders.append((tok, i))

    if offenders:
        print(f"\nFAIL: {len(offenders)} literal numeral(s) in the body that are not on "
              f"the allowlist -- each should be a macro from a measurement:")
        for tok, i in offenders[:25]:
            print(f"  {tok:<12} report.tex:{i}")
        if len(offenders) > 25:
            print(f"  ... and {len(offenders) - 25} more")
    else:
        print(f"OK: no hand-typed quantities in the body "
              f"({len(LITERAL_OK)} literals allowlisted with reasons)")
    return len(offenders)


def check_math(report: str) -> int:
    """Flag paragraphs with an odd number of ``$`` delimiters.

    Inline math is delimited by matched pairs, and TeX will happily run past the end
    of a sentence looking for the closing one -- so a single stray ``$`` silently
    typesets the rest of the paragraph in italics with the spacing wrong, or dies
    with ``Missing $ inserted`` several lines later. Both are hard to spot by reading
    and impossible to spot without a compiler, which this machine does not have.
    (There was one: ``vs.\\ \\tThreeXattnRTwoV$)`` in the Task 3 emotion paragraph.)

    Counting per paragraph rather than per file localises the offender; counting per
    line would false-positive on the many equations that legitimately wrap.
    """
    body = report.partition(r"\begin{document}")[2] or report
    body = re.sub(r"\\[$]", "", body)          # \$ is a literal dollar sign, not math
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", body, flags=re.S)

    offenders: list[tuple[int, int]] = []
    line_no = report.count("\n", 0, report.find(r"\begin{document}")) + 1
    for para in body.split("\n\n"):
        n = para.count("$")
        if n % 2:
            offenders.append((line_no, n))
        line_no += para.count("\n") + 2

    if offenders:
        print(f"\nFAIL: {len(offenders)} paragraph(s) with unbalanced $ (TeX would run "
              f"past the end of the paragraph looking for the close):")
        for i, n in offenders:
            print(f"  {n} delimiters in the paragraph near report.tex:{i}")
    else:
        print("OK: inline math delimiters balance in every paragraph")
    return len(offenders)


def check_refs(report: str) -> int:
    """Cross-check every \\ref against the \\labels that exist.

    LaTeX does not fail on an undefined reference -- it prints ``??`` and buries a
    warning in the log -- so this is easy to ship. The \\resfig / \\restab helpers
    define fig:<name> / tab:<name> labels implicitly, so those count as declared.
    """
    labels = set(re.findall(r"\\label\{([^}]+)\}", report))
    labels |= {f"fig:{n}" for n in re.findall(r"\\resfig\{([A-Za-z0-9_]+)\}", report)}
    labels |= {f"tab:{n}" for n in re.findall(r"\\restab\{([A-Za-z0-9_]+)\}", report)}
    # the two helper definitions themselves carry parameterised labels
    labels -= {"fig:#1", "tab:#1"}

    refs: list[tuple[str, int]] = []
    for i, line in enumerate(report.splitlines(), 1):
        for r in re.findall(r"\\(?:ref|pageref)\{([^}]+)\}", line):
            refs.append((r, i))

    dangling = [(r, i) for r, i in refs if r not in labels]
    unused = sorted(lab for lab in labels
                    if lab.startswith("sec:") and lab not in {r for r, _ in refs})

    print(f"\nreport.tex declares {len(labels)} labels and makes {len(refs)} references")
    if dangling:
        print(f"FAIL: {len(dangling)} reference(s) point at no label "
              f"(LaTeX would print ?? and only warn):")
        for r, i in dangling:
            print(f"  \\ref{{{r}}}  report.tex:{i}")
    else:
        print("OK: every \\ref resolves to a label")
    if unused:
        print(f"note: {len(unused)} section label(s) never referenced: {', '.join(unused)}")
    return len(dangling)


def main() -> int:
    if not REPORT.exists():
        print(f"missing {REPORT}")
        return 1
    report = strip_comments(REPORT.read_text(encoding="utf-8"))
    numbers = NUMBERS.read_text(encoding="utf-8") if NUMBERS.exists() else ""

    defined = set(DEF_RE.findall(numbers)) | set(DEF_RE.findall(report))
    used = set(USE_RE.findall(report))

    undefined = sorted(used - defined - KNOWN)
    # only report unused macros from numbers.tex -- report.tex's own helpers are
    # allowed to be internal
    from_numbers = set(DEF_RE.findall(numbers))
    unused = sorted(from_numbers - used)

    print(f"report.tex uses {len(used)} distinct control sequences")
    print(f"numbers.tex defines {len(from_numbers)} macros, {len(from_numbers) - len(unused)} of "
          f"which the report cites")

    if undefined:
        print(f"\nFAIL: {len(undefined)} control sequence(s) used but not defined "
              f"(would break compilation):")
        for n in undefined:
            for i, line in enumerate(report.splitlines(), 1):
                if f"\\{n}" in line:
                    print(f"  \\{n:26s} report.tex:{i}")
                    break
    else:
        print("\nOK: every macro report.tex uses is defined")

    dangling = check_refs(report)
    literals = check_literals(report)
    math = check_math(report)

    # A ?? in the rendered PDF means a cited number has no measurement behind it.
    cited_missing = sorted(
        m for m in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{\\textbf\{\?\?\}\}", numbers)
        if m in used)
    if cited_missing:
        print(f"\nWARNING: {len(cited_missing)} macro(s) the report cites have no measurement "
              f"and will render as ?? :")
        print("  " + ", ".join(cited_missing))

    if unused:
        print(f"\nnote: {len(unused)} measured value(s) never cited in the report:")
        for i in range(0, len(unused), 6):
            print("  " + ", ".join(unused[i:i + 6]))

    return 1 if (undefined or dangling or literals or math) else 0


if __name__ == "__main__":
    sys.exit(main())
