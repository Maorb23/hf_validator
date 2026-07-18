from __future__ import annotations

import html
from typing import Any

from hf_model_watch.reporting.markdown_report import format_value


def write_html_report(title: str, sections: list[tuple[str, str]], out_path: Any) -> None:
    body = "\n".join(
        f"<section><h2>{html.escape(heading)}</h2>\n{content}</section>"
        for heading, content in sections
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
    th, td {{ border: 1px solid #d6dde5; padding: 0.45rem 0.55rem; text-align: left; }}
    th {{ background: #eef3f7; }}
    code, pre {{ background: #f6f8fa; }}
    pre {{ padding: 1rem; overflow-x: auto; }}
    .status {{ font-weight: 700; }}
  </style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{body}
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_html = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(format_value(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def html_list(items: dict[str, Any]) -> str:
    entries = "".join(
        f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(format_value(value))}</li>"
        for key, value in items.items()
    )
    return f"<ul>{entries}</ul>"
