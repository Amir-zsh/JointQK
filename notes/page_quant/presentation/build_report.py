"""Build pgq_arc_report.html from pgq_arc_report_src.html.

The source uses LaTeX delimiters \\( ... \\) (inline) and \\[ ... \\]
(display); this script compiles them to native MathML (no runtime JS, no
CDN — the artifact host blocks external requests) and writes the final
self-contained HTML next to the source.

Dependency: latex2mathml (pure python; `pip install latex2mathml`).

Usage: python notes/page_quant/presentation/build_report.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import latex2mathml.converter as l2m

HERE = Path(__file__).resolve().parent
SRC = HERE / "pgq_arc_report_src.html"
OUT = HERE / "pgq_arc_report.html"


def build() -> None:
    text = SRC.read_text()
    errors: list[str] = []

    def conv(display: str):
        def repl(m: re.Match) -> str:
            tex = m.group(1).strip()
            try:
                mml = l2m.convert(tex, display=display)
            except Exception as exc:  # surface, don't silently drop math
                errors.append(f"{display}: {tex[:60]!r}: {exc}")
                return m.group(0)
            if display == "block":
                return f'<div class="eqm">{mml}</div>'
            return mml
        return repl

    text = re.sub(r"\\\[(.+?)\\\]", conv("block"), text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", conv("inline"), text, flags=re.DOTALL)
    if errors:
        sys.exit("LaTeX conversion failures:\n" + "\n".join(errors))
    leftovers = re.findall(r"\\[\[(]", text)
    if leftovers:
        sys.exit(f"{len(leftovers)} unconverted math delimiters remain")
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)/1024:.0f} KiB)")


if __name__ == "__main__":
    build()
