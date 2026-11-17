#!/usr/bin/env python
"""Convert a Markdown file to a PDF using Python package dependencies."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from markdown_it import MarkdownIt
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound
from weasyprint import CSS, HTML


PRINT_CSS = """
@page {
    size: Letter;
    margin: 0.75in;
}

body {
    color: #1f2933;
    font-family: "Helvetica", "Arial", sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
}

h1, h2, h3, h4 {
    color: #111827;
    line-height: 1.2;
    margin: 1.4em 0 0.45em;
}

h1 {
    border-bottom: 1px solid #d8dee9;
    font-size: 24pt;
    margin-top: 0;
    padding-bottom: 0.25em;
}

h2 {
    border-bottom: 1px solid #e5e7eb;
    font-size: 17pt;
    padding-bottom: 0.15em;
}

h3 {
    font-size: 13pt;
}

p, ul, ol, pre, blockquote {
    margin: 0.65em 0;
}

li {
    margin: 0.25em 0;
}

a {
    color: #1d4ed8;
    text-decoration: none;
}

code {
    background: #f3f4f6;
    border-radius: 3px;
    color: #111827;
    font-family: "Menlo", "Consolas", monospace;
    font-size: 9pt;
    padding: 0.08em 0.25em;
}

pre {
    background: #f8fafc;
    border: 1px solid #d8dee9;
    border-radius: 4px;
    break-inside: avoid;
    overflow-wrap: break-word;
    page-break-inside: avoid;
    padding: 0.7em;
    white-space: pre-wrap;
}

pre code {
    background: transparent;
    padding: 0;
}

table {
    border-collapse: collapse;
    width: 100%;
}

th, td {
    border: 1px solid #d8dee9;
    padding: 0.35em 0.45em;
}

th {
    background: #f3f4f6;
}
"""


def _highlight_code(code: str, lang: str, attrs: str) -> str:
    """Render one fenced code block with Pygments for Markdown-It."""
    _ = attrs
    normalized_lang = lang.strip().split(maxsplit=1)[0] if lang else ""
    if normalized_lang:
        try:
            lexer = get_lexer_by_name(normalized_lang)
        except ClassNotFound:
            lexer = TextLexer()
    else:
        lexer = TextLexer()

    highlighted = highlight(
        code,
        lexer,
        HtmlFormatter(nowrap=True),
    )
    class_attr = (
        f' class="language-{html.escape(normalized_lang, quote=True)}"'
        if normalized_lang
        else ""
    )
    return f"<pre class=\"codehilite\"><code{class_attr}>{highlighted}</code></pre>"


def _document_title(markdown_text: str, fallback: str) -> str:
    """Return the first Markdown H1 as a document title, if present."""
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
    return fallback


def convert_markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    """Convert one Markdown document to a PDF file."""
    markdown_text = input_path.read_text(encoding="utf-8")
    title = _document_title(markdown_text, input_path.stem)
    md = MarkdownIt(
        "commonmark",
        {
            # Preserve inline HTML links with per-link browser targets.
            "html": True,
            "highlight": _highlight_code,
        },
    )
    body_html = md.render(markdown_text)
    pygments_css = HtmlFormatter().get_style_defs(".codehilite")
    document_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>{pygments_css}</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(
        string=document_html,
        base_url=str(input_path.parent.resolve()),
    ).write_pdf(
        str(output_path),
        stylesheets=[CSS(string=PRINT_CSS)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a Markdown document to PDF.",
    )
    parser.add_argument("input_markdown", type=Path)
    parser.add_argument("output_pdf", type=Path)
    return parser.parse_args()


def main() -> int:
    """Run the Markdown-to-PDF converter."""
    args = parse_args()
    input_path = args.input_markdown
    output_path = args.output_pdf

    if not input_path.is_file():
        raise FileNotFoundError(f"Markdown file not found: {input_path}")

    convert_markdown_to_pdf(input_path, output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
