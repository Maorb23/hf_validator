from __future__ import annotations

import re
from typing import Any

KNOWN_FAMILIES = {
    "qwen": "Qwen",
    "llama": "Llama",
    "mistral": "Mistral",
    "gemma": "Gemma",
    "phi": "Phi",
    "deepseek": "DeepSeek",
    "yi": "Yi",
}


def normalize_model_id(model_id: str) -> str:
    return model_id.strip().strip("/").lower()


def extract_family(model_id: str, tags: list[str] | None = None) -> str | None:
    normalized = normalize_model_id(model_id)
    tag_text = " ".join(tags or []).lower()
    text = f"{normalized} {tag_text}"
    for key, family in KNOWN_FAMILIES.items():
        if key in text:
            return family
    return None


def extract_size_billion(model_id: str, tags: list[str] | None = None) -> float | None:
    text = f"{model_id} {' '.join(tags or [])}".lower()
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?:[^a-z]|$)", text)
    if match:
        return float(match.group(1))
    return None


def extract_version_or_date(model_id: str, tags: list[str] | None = None) -> str | None:
    text = f"{model_id} {' '.join(tags or [])}"
    version = re.search(r"\bv(?:ersion)?[-_]?(\d+(?:\.\d+){0,2})\b", text, flags=re.IGNORECASE)
    if version:
        return f"v{version.group(1)}"
    date = re.search(r"\b(20\d{2}[-_]?\d{2}(?:[-_]?\d{2})?)\b", text)
    if date:
        return date.group(1).replace("_", "-")
    return None


def size_band(size_billion: float | None) -> str:
    if size_billion is None:
        return "unknown"
    if size_billion < 3:
        return "small"
    if size_billion < 8:
        return "medium"
    if size_billion <= 14:
        return "large"
    if size_billion <= 30:
        return "xlarge"
    return "too_large"


def extract_lineage(
    candidate: dict[str, Any],
    prior_tested_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model_id = str(candidate.get("model_id", ""))
    tags = candidate.get("tags") if isinstance(candidate.get("tags"), list) else []
    size = candidate.get("estimated_params_billion") or extract_size_billion(model_id, tags)
    family = extract_family(model_id, tags)
    band = size_band(float(size) if isinstance(size, int | float) else None)
    normalized = normalize_model_id(model_id)

    related_tested = []
    for tested in prior_tested_models or []:
        tested_family = extract_family(str(tested.get("model_id", "")), tested.get("tags") or [])
        tested_size = tested.get("estimated_params_billion") or extract_size_billion(
            str(tested.get("model_id", "")), tested.get("tags") or []
        )
        if family and tested_family == family and size_band(tested_size) == band:
            related_tested.append(tested.get("model_id"))

    status = "detected" if family or size else "inconclusive"
    if not prior_tested_models:
        predecessor_status = "inconclusive"
    elif related_tested:
        predecessor_status = "local_tested_related_models_found"
    else:
        predecessor_status = "inconclusive"

    return {
        "normalized_model_id": normalized,
        "family": family,
        "size_billion": size,
        "size_band": band,
        "version_or_date": extract_version_or_date(model_id, tags),
        "status": status,
        "predecessor_status": predecessor_status,
        "related_tested_models": related_tested,
    }
