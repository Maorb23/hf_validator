from __future__ import annotations

import json
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from hf_model_watch.hub.scan import (
    _apply_context_fallback,
    _candidate_from_model_info,
    _enrich_candidate_metadata,
)

DEFAULT_LOOKUP_OUTPUT = Path("artifacts/scan/lookup_models.jsonl")
MAX_LOOKUP_LIMIT = 100


def lookup_models(
    *,
    model_id: str | None = None,
    family: str | None = None,
    query: str | None = None,
    fuzzy: bool = False,
    limit: int = 20,
    output_path: str | Path = DEFAULT_LOOKUP_OUTPUT,
    api: HfApi | None = None,
) -> list[dict[str, Any]]:
    """Look up one Hub model or discover a model family without downloading weights."""
    selector_name, selector = _validate_selector(model_id, family, query)
    if not 1 <= limit <= MAX_LOOKUP_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LOOKUP_LIMIT}")

    client = api or HfApi()
    if selector_name == "model_id":
        info = client.model_info(selector, files_metadata=True)
        records = [_candidate_from_model_info(info)]
        records[0]["lookup_match"] = {
            "mode": "exact_model_id",
            "query": selector,
            "score": 1.0,
            "reasons": ["exact model ID"],
        }
    else:
        records = _search_models(client, selector, fuzzy=fuzzy, limit=limit)

    # Lookup is deliberately broader than scanning: users need config/tokenizer metadata even
    # when a model has no declared license or downloads yet.
    _enrich_candidate_metadata(
        records,
        {"max_candidates_to_run": limit, "min_downloads": 0},
        client,
        require_license=False,
    )
    _apply_context_fallback(records, {"fallback_context_length": 2048})
    _write_jsonl(output_path, records)
    return records


def _validate_selector(
    model_id: str | None, family: str | None, query: str | None
) -> tuple[str, str]:
    options = (("model_id", model_id), ("family", family), ("query", query))
    selectors = [(name, value.strip()) for name, value in options if value and value.strip()]
    if len(selectors) != 1:
        raise ValueError("provide exactly one of model_id, family, or query")
    return selectors[0]


def _search_models(client: HfApi, query: str, *, fuzzy: bool, limit: int) -> list[dict[str, Any]]:
    # Request a bounded pool from the Hub, then make all matching/ranking decisions locally.
    summaries = client.list_models(
        search=query,
        sort="downloads",
        limit=limit * 5,
        full=True,
        cardData=True,
        fetch_config=True,
    )
    matched: list[dict[str, Any]] = []
    for summary in summaries:
        candidate = _candidate_from_model_info(summary)
        match = _match_candidate(candidate, query, fuzzy=fuzzy)
        if match is None:
            continue
        candidate["lookup_match"] = match
        matched.append(candidate)
    matched.sort(
        key=lambda item: (
            float(item["lookup_match"]["score"]),
            int(item.get("downloads") or 0),
        ),
        reverse=True,
    )
    return matched[:limit]


def _match_candidate(
    candidate: dict[str, Any], query: str, *, fuzzy: bool
) -> dict[str, Any] | None:
    normalized_query = _normalize(query)
    if not normalized_query:
        return None
    fields = list(_candidate_fields(candidate))
    exact_fields = [name for name, value in fields if normalized_query in _normalize(value)]
    if exact_fields:
        return {
            "mode": "family" if not fuzzy else "fuzzy",
            "query": query,
            "score": 1.0,
            "reasons": [f"substring match in {name}" for name in exact_fields],
        }
    if not fuzzy:
        return None

    score, field = max((_best_fuzzy_score(normalized_query, value), name) for name, value in fields)
    if score < 0.55:
        return None
    return {
        "mode": "fuzzy",
        "query": query,
        "score": round(score, 4),
        "reasons": [f"fuzzy match in {field}"],
    }


def _candidate_fields(candidate: dict[str, Any]) -> Iterable[tuple[str, str]]:
    yield "model_id", str(candidate.get("model_id") or "")
    yield "author", str(candidate.get("author") or "")
    for tag in candidate.get("tags") or []:
        yield "tag", str(tag)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _best_fuzzy_score(query: str, value: str) -> float:
    """Compare both the whole field and its ID/tag components.

    Model IDs commonly contain a namespace and punctuation, so comparing only the complete ID
    makes a small typo in a family name look less similar than it is.
    """
    parts = [_normalize(value), *(_normalize(part) for part in re.findall(r"[a-zA-Z0-9]+", value))]
    return max((SequenceMatcher(None, query, part).ratio() for part in parts if part), default=0.0)


def _write_jsonl(output_path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
