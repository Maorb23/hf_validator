from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hf_model_watch.hub.candidate_filter import filter_candidate
from hf_model_watch.hub.lineage import extract_lineage

DEFAULT_WEIGHTS = {
    "recency": 0.18,
    "downloads": 0.14,
    "likes": 0.1,
    "credible_author": 0.11,
    "complete_metadata": 0.12,
    "chat_template": 0.08,
    "context_length": 0.08,
    "lineage": 0.08,
    "known_family": 0.06,
    "useful_size_band": 0.04,
    "uniqueness": 0.01,
}


def score_candidate(
    candidate: dict[str, Any],
    scan_config: dict[str, Any],
    scoring_config: dict[str, Any],
    already_tested_revisions: set[tuple[str, str]] | None = None,
    prior_tested_models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    filter_result = filter_candidate(candidate, scan_config, already_tested_revisions)
    lineage = extract_lineage(candidate, prior_tested_models)
    features = _score_features(candidate, scoring_config, lineage, filter_result)
    weights = _normalized_weights(scoring_config.get("weights") or DEFAULT_WEIGHTS)
    score = sum(features.get(name, 0.0) * weight for name, weight in weights.items())

    if not filter_result.eligible:
        score = 0.0

    return {
        "model_id": candidate.get("model_id"),
        "revision": candidate.get("revision"),
        "eligible": filter_result.eligible,
        "score": round(score, 4),
        "blocking_reasons": filter_result.blocking_reasons,
        "score_breakdown": {
            name: {
                "feature": round(features.get(name, 0.0), 4),
                "weight": round(weight, 4),
                "contribution": round(features.get(name, 0.0) * weight, 4),
            }
            for name, weight in weights.items()
        },
        "lineage": lineage,
    }


def score_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    scan_config: dict[str, Any],
    scoring_config: dict[str, Any],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for candidate in read_jsonl(input_path):
        scored.append(score_candidate(candidate, scan_config, scoring_config))
    scored.sort(key=lambda item: item["score"], reverse=True)
    max_candidates = int(scan_config.get("max_candidates_to_run", len(scored)))
    selected = scored[:max_candidates]
    write_jsonl(output_path, selected)
    return selected


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Candidate input file not found: {jsonl_path}")
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    records.append(data)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _score_features(
    candidate: dict[str, Any],
    scoring_config: dict[str, Any],
    lineage: dict[str, Any],
    filter_result: Any,
) -> dict[str, float]:
    downloads = _coerce_float(candidate.get("downloads")) or 0.0
    likes = _coerce_float(candidate.get("likes")) or 0.0
    author = str(candidate.get("author") or "").lower()
    credible_authors = {str(item).lower() for item in scoring_config.get("credible_authors", [])}
    known_families = set(scoring_config.get("known_families", []))
    size = _coerce_float(candidate.get("estimated_params_billion"))
    has_complete_metadata = all(
        candidate.get(key)
        for key in (
            "has_tokenizer",
            "has_config",
            "has_weights",
            "license",
            "architecture",
            "model_card_present",
        )
    )

    return {
        "recency": _recency_score(candidate.get("last_modified")),
        "downloads": min(1.0, math.log10(downloads + 1) / 6),
        "likes": min(1.0, math.log10(likes + 1) / 4),
        "credible_author": 1.0 if author in credible_authors else 0.0,
        "complete_metadata": 1.0 if has_complete_metadata else 0.0,
        "chat_template": 1.0 if candidate.get("chat_template_present") else 0.0,
        "context_length": 1.0 if candidate.get("context_length") else 0.0,
        "lineage": 1.0 if lineage.get("status") == "detected" else 0.0,
        "known_family": 1.0 if lineage.get("family") in known_families else 0.0,
        "useful_size_band": _useful_size_score(size),
        "uniqueness": 0.0 if _looks_duplicate(candidate, filter_result) else 1.0,
    }


def _normalized_weights(raw_weights: dict[str, Any]) -> dict[str, float]:
    weights = {str(key): float(value) for key, value in raw_weights.items() if float(value) > 0}
    total = sum(weights.values())
    if total <= 0:
        return DEFAULT_WEIGHTS
    return {key: value / total for key, value in weights.items()}


def _recency_score(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    age_days = max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds() / 86400)
    return max(0.0, min(1.0, 1.0 - (age_days / 30.0)))


def _useful_size_score(size: float | None) -> float:
    if size is None:
        return 0.5
    if 1 <= size <= 14:
        return 1.0
    if 14 < size <= 30:
        return 0.8
    if 0 < size < 1:
        return 0.4
    return 0.0


def _looks_duplicate(candidate: dict[str, Any], filter_result: Any) -> bool:
    tags = " ".join(candidate.get("tags") or []).lower()
    model_id = str(candidate.get("model_id") or "").lower()
    return (
        "adapter" in tags
        or "merge" in tags
        or "quantized" in tags
        or "-gguf" in model_id
        or bool(getattr(filter_result, "blocking_reasons", []))
    )


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
