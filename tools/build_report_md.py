"""Render report/report.md from report/report.tex -- same source, no retyping.

There is no TeX toolchain on this machine, so report.tex cannot be turned into a
PDF here. That would leave the report unreadable to anyone without LaTeX, which is
not an acceptable state for the primary deliverable. Rather than maintain a second
hand-written Markdown copy -- which is exactly the drift that numbers.tex exists to
prevent -- this converts the one source.

It is a targeted converter, not a general LaTeX engine: it handles the constructs
report.tex actually uses and shouts about anything it does not recognise, which is
the right trade for a single known document.

    python tools/build_report_md.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "report" / "report.tex"
NUMBERS = ROOT / "report" / "numbers.tex"
OUT = ROOT / "report" / "report.md"
TABLES = ROOT / "results" / "tables"

# Macros that expanded to an unmeasured ?? in this build. Populated by
# expand_macros() so the summary counts real gaps rather than the prose that
# describes the ?? convention.
EXPANDED_MISSING: set[str] = set()


# --------------------------------------------------------------------------- #
def load_macros() -> dict[str, str]:
    """numbers.tex -> {macro name: plain-text value}."""
    if not NUMBERS.exists():
        return {}
    out = {}
    for name, body in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", NUMBERS.read_text(encoding="utf-8")):
        body = body.replace("{,}", ",")                    # LaTeX thin-space digits
        body = body.replace(r"\textbf{??}", "**??**")       # unmeasured stays loud
        out[name] = body
    return out


def strip_comments(tex: str) -> str:
    keep = []
    for line in tex.splitlines():
        i, esc, cut = 0, False, len(line)
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                cut = i
                break
            else:
                esc = False
            i += 1
        s = line[:cut]
        # a line that was pure comment disappears entirely rather than becoming blank
        if cut == 0 and line.strip().startswith("%"):
            continue
        keep.append(s.rstrip())
    return "\n".join(keep)


# --------------------------------------------------------------------------- #
HEAD_RE = re.compile(r"\\(section|subsection|subsubsection)\{([^}]*)\}")

# LaTeX accents, braced form only. The braces are what make this safe: a pattern
# that also accepted \c or \v bare would rewrite the c of \cdot and \citep.
ACCENTS: dict[str, dict[str, str]] = {
    "v":  {"c": "\u010d", "s": "\u0161", "z": "\u017e", "r": "\u0159",
           "e": "\u011b", "n": "\u0148", "d": "\u010f", "t": "\u0165"},
    "'":  {"a": "\u00e1", "c": "\u0107", "e": "\u00e9", "i": "\u00ed",
           "n": "\u0144", "o": "\u00f3", "s": "\u015b", "u": "\u00fa",
           "y": "\u00fd", "z": "\u017a"},
    "`":  {"a": "\u00e0", "e": "\u00e8", "i": "\u00ec", "o": "\u00f2", "u": "\u00f9"},
    '"':  {"a": "\u00e4", "e": "\u00eb", "i": "\u00ef", "o": "\u00f6",
           "u": "\u00fc", "y": "\u00ff"},
    "^":  {"a": "\u00e2", "e": "\u00ea", "i": "\u00ee", "o": "\u00f4", "u": "\u00fb"},
    "~":  {"a": "\u00e3", "n": "\u00f1", "o": "\u00f5"},
    "c":  {"c": "\u00e7", "s": "\u015f", "g": "\u011f"},
    "r":  {"a": "\u00e5", "u": "\u016f"},
    "=":  {"a": "\u0101", "e": "\u0113", "o": "\u014d", "u": "\u016b"},
    ".":  {"z": "\u017c", "e": "\u0117"},
}
ACCENT_RE = re.compile(r"\\([`'\"^~=.vcr])\{([A-Za-z])\}")


def walk(body: str) -> tuple[dict[str, str], list[str]]:
    """One ordered pass over the document, producing both numbering artefacts.

    Returns ``(labels, heading_numbers)`` where ``labels`` maps every ``\\label``
    target to the number ``\\ref`` should print, and ``heading_numbers`` lists the
    number of each heading in document order so ``convert`` can pop them in the
    same sequence. Both come from the same walk, so they cannot disagree.

    Tables and figures are numbered by their ``\\restab`` / ``\\resfig`` position,
    which in Markdown is also their final position -- there is no float placement
    to drift against.
    """
    labels: dict[str, str] = {}
    heads: list[str] = []
    sec = sub = ssub = tab_n = fig_n = 0
    appendix = False
    current = "0"

    # \label may sit on the heading line or a line or two after it, so the "current"
    # heading number is carried forward until the next heading replaces it.
    for line in body.splitlines():
        if line.strip().startswith(r"\appendix"):
            appendix, sec, sub, ssub = True, 0, 0, 0
            continue
        if m := HEAD_RE.search(line):
            kind = m.group(1)
            if kind == "section":
                sec, sub, ssub = sec + 1, 0, 0
            elif kind == "subsection":
                sub, ssub = sub + 1, 0
            else:
                ssub += 1
            top = chr(ord("A") + sec - 1) if appendix else str(sec)
            current = top + (f".{sub}" if sub else "") + (f".{ssub}" if ssub else "")
            heads.append(current)

        for name in re.findall(r"\\restab\{([A-Za-z0-9_]+)\}", line):
            tab_n += 1
            labels[f"tab:{name}"] = str(tab_n)
        for name in re.findall(r"\\resfig\{([A-Za-z0-9_]+)\}", line):
            fig_n += 1
            labels[f"fig:{name}"] = str(fig_n)
        for lab in re.findall(r"\\label\{([^}]+)\}", line):
            if lab.startswith("sec:"):
                labels[lab] = current
    return labels, heads


# --------------------------------------------------------------------------- #
def read_group(s: str, i: int) -> tuple[str, int]:
    """Read the balanced ``{...}`` group starting at ``s[i]``; return (body, end).

    ``end`` is the index just past the closing brace. Raises if ``s[i]`` is not an
    opening brace, which would mean the caller mis-parsed the command.
    """
    if i >= len(s) or s[i] != "{":
        raise ValueError(f"expected {{ at offset {i}: {s[i:i + 40]!r}")
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{" and (j == i or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(f"unbalanced group from offset {i}")


def replace_cmd(t: str, name: str, nargs: int, fn) -> str:
    """Substitute every ``\\name{..}..`` using balanced-brace argument parsing.

    A regex cannot do this. ``\\resfig{graph_coherence}{0.98}{... $S_{\\text{graph}}$
    ...}`` nests braces two deep inside its caption, and a
    ``\\{(?:[^{}]|\\{[^{}]*\\})*\\}`` pattern silently fails to match it -- leaving the
    raw command sitting in the Markdown. Deepening the pattern only moves the
    failure to the next nesting level, so match braces properly instead.
    """
    out, i, tok = [], 0, "\\" + name
    while (k := t.find(tok, i)) != -1:
        # guard against \resfigure matching \resfig
        after = k + len(tok)
        if after < len(t) and t[after].isalpha():
            out.append(t[i:after])
            i = after
            continue
        out.append(t[i:k])
        args, j = [], after
        try:
            for _ in range(nargs):
                arg, j = read_group(t, j)
                args.append(arg)
        except ValueError as exc:
            print(f"warning: could not parse \\{name} at offset {k}: {exc}")
            out.append(tok)
            i = after
            continue
        out.append(fn(*args))
        i = j
    out.append(t[i:])
    return "".join(out)


# --------------------------------------------------------------------------- #
def convert(body: str, macros: dict[str, str], labels: dict[str, str],
            heads: list[str]) -> str:
    t = body

    # Numbers first, so macros inside equations and captions expand too. This is
    # safe everywhere: only \name sequences present in numbers.tex are touched.
    t = expand_macros(t, macros)

    # Content that must survive the punctuation rules below verbatim goes into a
    # vault and comes back at the end. Without this, the `--` -> en-dash rule
    # silently corrupts both the |---|---| separator of every inlined Markdown
    # table and every `--flag` in the reproducibility commands.
    vault: list[str] = []

    def stash(text: str) -> str:
        vault.append(text)
        return f"\x00V{len(vault) - 1}\x00"

    # --- verbatim ----------------------------------------------------------- #
    t = re.sub(r"\\begin\{verbatim\}\n?(.*?)\\end\{verbatim\}",
               lambda m: stash("```bash\n" + m.group(1).rstrip() + "\n```"), t, flags=re.S)

    # --- abstract ----------------------------------------------------------- #
    t = re.sub(r"\\begin\{abstract\}", "## Abstract\n", t)
    t = re.sub(r"\\end\{abstract\}", "", t)

    # --- display math -> $$ block (GitHub/MathJax renders these) ------------ #
    def eq(m):
        inner = re.sub(r"\s+", " ", m.group(1).strip())
        return stash(f"\n$$\n{inner}\n$$\n")
    t = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eq, t, flags=re.S)

    # --- generated artefacts ------------------------------------------------ #
    def restab(name):
        md = TABLES / f"{name}.md"
        num = labels.get(f"tab:{name}", "?")
        if not md.exists():
            return (f"\n> **Table {num}** &mdash; `results/tables/{name}.md` not generated yet; "
                    f"run `python scripts/evaluate.py`.\n")
        return stash(f"\n**Table {num}.**\n\n{md.read_text(encoding='utf-8').strip()}\n")
    t = replace_cmd(t, "restab", 1, restab)

    def resfig(name, _width, cap):
        num = labels.get(f"fig:{name}", "?")
        rel = f"../results/plots/{name}.png"
        if not (ROOT / "results" / "plots" / f"{name}.png").exists():
            return (f"\n> **Figure {num}** &mdash; `{rel}` not generated yet; "
                    f"run `python scripts/evaluate.py`. Caption: {cap}\n")
        # the caption stays in the document text so the final inline pass styles it
        return f"\n![Figure {num}]({rel})\n\n**Figure {num}.** {cap}\n"
    t = replace_cmd(t, "resfig", 3, resfig)

    # --- bibliography ------------------------------------------------------- #
    def bib(m):
        items = re.findall(r"\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem\{|\Z)", m.group(1), flags=re.S)
        lines = ["## References", ""]
        for i, (key, text) in enumerate(items, 1):
            lines.append(f"{i}. <a name=\"{key}\"></a>{text.strip()}")
        return "\n".join(lines)
    t = re.sub(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}",
               bib, t, flags=re.S)

    # --- headings ----------------------------------------------------------- #
    # One ordered pass so the numbers stay in step with walk()'s numbering, which
    # is what \ref resolved against. Three separate regex passes would not.
    depth = {"section": "##", "subsection": "###", "subsubsection": "####"}
    pending = list(heads)

    def head(m):
        num = pending.pop(0) if pending else "?"
        return f"\n{depth[m.group(1)]} {num}. {m.group(2)}\n"
    t = HEAD_RE.sub(head, t)
    t = t.replace(r"\appendix", "\n---\n")

    # --- lists -------------------------------------------------------------- #
    t = re.sub(r"\\begin\{itemize\}(\\itemsep[^\s]*)?", "\n", t)
    t = re.sub(r"\\end\{itemize\}", "\n", t)
    t = re.sub(r"^\s*\\item\s*", "- ", t, flags=re.M)

    t = inline(t, labels)

    for i, text in enumerate(vault):
        t = t.replace(f"\x00V{i}\x00", text)
    return t


def expand_macros(t: str, macros: dict[str, str]) -> str:
    """Replace \\fooBar / \\fooBar{} / \\fooBar<space> with its measured value.

    Records which unmeasured macros were actually substituted in ``EXPANDED_MISSING``.
    Grepping the finished markdown for ``??`` instead would count the appendix
    paragraph that *explains* the ?? convention, which reports two phantom
    unmeasured values on a report where every macro is backed by a measurement.
    """
    def sub(m: re.Match) -> str:
        body = macros.get(m.group(1))
        if body is None:
            return m.group(0)
        if "??" in body:
            EXPANDED_MISSING.add(m.group(1))
        return body

    return re.sub(r"\\([A-Za-z]+)(?:\{\}|\\(?=\s)|\b)", sub, t)


def inline(t: str, labels: dict[str, str]) -> str:
    """Inline-level conversions. Runs after macro expansion and vaulting."""
    # cross-references first (before styling rules eat the braces)
    t = re.sub(r"~?\\ref\{([^}]+)\}",
               lambda m: " " + labels.get(m.group(1), m.group(1).split(":")[-1]), t)
    t = re.sub(r"~?\\cite\{([^}]+)\}",
               lambda m: " [" + ", ".join(f"[{k.strip()}](#{k.strip()})"
                                          for k in m.group(1).split(",")) + "]", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)

    # text styling
    t = re.sub(r"\\(?:textbf|best)\{((?:[^{}]|\{[^{}]*\})*)\}", r"**\1**", t)
    t = re.sub(r"\\(?:emph|textit)\{((?:[^{}]|\{[^{}]*\})*)\}", r"*\1*", t)
    t = re.sub(r"\\texttt\{((?:[^{}]|\{[^{}]*\})*)\}", r"`\1`", t)
    t = re.sub(r"\\textsc\{([^}]*)\}", r"\1", t)

    # Accented names in the bibliography. LaTeX spells these as \v{c} / \'{c};
    # Markdown has no such notion, so fold them to the composed characters --
    # otherwise "Velickovic" renders as literal backslashes in the reference list.
    t = ACCENT_RE.sub(lambda m: ACCENTS[m.group(1)].get(m.group(2), m.group(2)), t)

    # escapes, spacing commands and punctuation
    for a, b in ((r"\%", "%"), (r"\_", "_"), (r"\&", "&"), (r"\$", "$"), (r"\#", "#"),
                 (r"\{", "{"), (r"\}", "}"),
                 (r"\,", " "), (r"\ ", " "), (r"\;", " "),
                 ("---", "\u2014"), ("--", "\u2013"), ("``", "\u201c"), ("''", "\u201d"),
                 (r"\ldots", "\u2026"), (r"\dots", "\u2026"),
                 (r"\emph", ""), (r"\noindent", ""), (r"\centering", ""),
                 (r"\small", ""), (r"\itemsep2pt", ""), (r"\itemsep1pt", "")):
        t = t.replace(a, b)
    t = re.sub(r"\\vspace\{[^}]*\}|\\hspace\{[^}]*\}|\\setlength\{[^}]*\}\{[^}]*\}", "", t)
    t = re.sub(r"\\\\(\[[^\]]*\])?", "  \n", t)
    t = t.replace("~", " ")
    return t


# --------------------------------------------------------------------------- #
def main() -> int:
    if not TEX.exists():
        print(f"missing {TEX}")
        return 1
    tex = strip_comments(TEX.read_text(encoding="utf-8"))
    body = tex[tex.find(r"\begin{document}") + len(r"\begin{document}"):]
    body = body[: body.find(r"\end{document}")]
    body = body.replace(r"\maketitle", "")

    macros = load_macros()
    labels, heads = walk(body)
    md = convert(body, macros, labels, heads)

    # LaTeX prose indentation is meaningless, but Markdown reads 6+ leading spaces
    # inside a list item as an indented code block -- which is exactly what the
    # \item continuation lines in this document would become. Flatten leading
    # whitespace everywhere except inside fenced blocks.
    lines, fenced = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            lines.append(line.lstrip())
            continue
        lines.append(line if fenced else line.lstrip())
    md = "\n".join(lines)

    # tidy whitespace: no more than one blank line, no trailing spaces on blanks
    md = re.sub(r"[ \t]+$", "", md, flags=re.M)
    md = re.sub(r"\n{3,}", "\n\n", md).strip()

    header = (
        "# Graph Neural Networks and BERT for Multimodal Music Context Understanding\n\n"
        "*A four-task study, with the negative results kept in.*  \n"
        "CSE425 — Neural Networks · Project Report\n\n"
        "> Generated from `report/report.tex` by `tools/build_report_md.py`; do not edit by\n"
        "> hand. Every number comes from `report/numbers.tex`, which is generated from\n"
        "> `results/metrics.json`. A **??** below marks a value that has not been measured.\n\n"
    )
    OUT.write_text(header + md + "\n", encoding="utf-8")

    leftover = sorted(set(re.findall(r"\\[A-Za-z]+", md)))
    words = len(md.split())
    print(f"wrote {OUT} ({words:,} words)")
    if leftover:
        print(f"note: {len(leftover)} unconverted LaTeX command(s) remain: "
              f"{', '.join(leftover[:20])}")
    if EXPANDED_MISSING:
        print(f"note: {len(EXPANDED_MISSING)} macro(s) the report cites are unmeasured "
              f"and render as ??: {', '.join(sorted(EXPANDED_MISSING))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
