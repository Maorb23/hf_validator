from __future__ import annotations

import json
from typing import Any


def format_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return text if text else "N/A"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(_escape_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(format_value(value)) for value in row) + " |")
    return "\n".join(lines)


def bullet_list(items: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: {format_value(value)}" for key, value in items.items())


def json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value or {}, indent=2, sort_keys=True) + "\n```"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
