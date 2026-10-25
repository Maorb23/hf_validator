"""Display metadata only; never part of execution or approval identity."""

import unicodedata
from datetime import UTC

MAX_STUDY_NAME = 160


def validate_name(value):
    if not isinstance(value, str):
        raise ValueError("Study name must be text.")
    value = value.strip()
    if len(value) > MAX_STUDY_NAME:
        raise ValueError("Study name must be at most 160 characters.")
    if any(unicodedata.category(c).startswith("C") for c in value):
        raise ValueError("Study name cannot contain control characters.")
    return value


def model_id(prepared):
    return (prepared.summary.get("model") or prepared.payload.get("model")
            or prepared.recommendation.selected.get("model_id") or "Model")


def study_name(prepared):
    if prepared.name:
        return prepared.name
    short_model = model_id(prepared).rstrip("/").rsplit("/", 1)[-1][:110]
    stamp = prepared.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"{short_model} · {stamp} · {prepared.id.hex[:8]}"
