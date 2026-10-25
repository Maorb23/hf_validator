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


def recommendation_name(rec):
    if rec.name:
        return rec.name
    description = " ".join(rec.requirements.get("request", "Recommendation").split())[:80]
    description = "".join(c for c in description if not unicodedata.category(c).startswith("C"))
    stamp = rec.created_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"{description or 'Recommendation'} · {stamp} · {rec.id.hex[:8]}"


def validate_recommendation_name(value):
    try:
        return validate_name(value)
    except ValueError as exc:
        raise ValueError(str(exc).replace("Study", "Recommendation")) from None
