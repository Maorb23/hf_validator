from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from hf_model_watch.hub.metadata import (
    estimate_download_size_gb,
    estimate_params_billion,
    extract_architectures,
    extract_author,
    extract_context_length,
    extract_license,
    has_config_file,
    has_safetensors_file,
    has_tokenizer_file,
    has_weight_file,
    normalize_datetime,
    read_json_file,
    sibling_names,
    tokenizer_has_chat_template,
)
from hf_model_watch.tracing import emit_trace_event

DEFAULT_SCAN_OUTPUT = Path("artifacts/scan/latest_models.jsonl")


def scan_models(
    scan_config: dict[str, Any],
    output_path: str | Path = DEFAULT_SCAN_OUTPUT,
    api: HfApi | None = None,
) -> list[dict[str, Any]]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    emit_trace_event(
        "scan_started",
        "scan",
        "started",
        "Hugging Face model scan started",
        metadata={"output_path": str(output)},
    )

    records: list[dict[str, Any]] = []
    client = api or HfApi()
    try:
        seen: set[str] = set()
        for summary in _discover_model_summaries(client, scan_config):
            candidate = _candidate_from_model_info(summary)
            model_id = str(candidate.get("model_id"))
            is_new = model_id not in seen
            allowed_pipeline = _pipeline_allowed(candidate, scan_config)
            recent_enough = _within_lookback(candidate, scan_config)
            if is_new and allowed_pipeline and recent_enough:
                seen.add(model_id)
                records.append(candidate)
        _enrich_small_json_metadata(records, scan_config)
    except Exception as exc:  # network/API failures are surfaced as trace events and empty JSONL.
        emit_trace_event(
            "scan_failed",
            "scan",
            "failed",
            "Hugging Face model scan failed",
            metadata={"error_type": type(exc).__name__, "error": str(exc)},
            artifacts={"output_path": str(output)},
        )
        _write_jsonl(output, records)
        return records

    _write_jsonl(output, records)
    emit_trace_event(
        "scan_completed",
        "scan",
        "completed",
        "Hugging Face model scan completed",
        metrics={"models_written": len(records)},
        artifacts={"output_path": str(output)},
    )
    return records


def _discover_model_summaries(api: HfApi, scan_config: dict[str, Any]) -> list[Any]:
    limit = int(scan_config.get("max_models_to_scan", 200))
    per_query_limit = max(limit, 1)
    summaries: list[Any] = []
    for sort_key in ("last_modified", "downloads"):
        for summary in api.list_models(
            sort=sort_key,
            limit=per_query_limit,
            full=True,
            cardData=True,
            fetch_config=True,
        ):
            summaries.append(summary)
            if len(summaries) >= limit * 2:
                break
    return summaries


def _candidate_from_model_info(model_info: Any) -> dict[str, Any]:
    model_id = str(getattr(model_info, "modelId", None) or getattr(model_info, "id", ""))
    info = model_info
    tags = [str(tag) for tag in (getattr(info, "tags", None) or [])]
    siblings = getattr(info, "siblings", None) or []
    files = sibling_names(siblings)
    config = getattr(info, "config", None) or {}
    if not isinstance(config, dict):
        config = {}
    tokenizer_config: dict[str, Any] = {}
    card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    pipeline_tag = getattr(info, "pipeline_tag", None) or getattr(info, "pipelineTag", None)

    return {
        "model_id": model_id,
        "author": getattr(info, "author", None) or extract_author(model_id),
        "last_modified": normalize_datetime(getattr(info, "lastModified", None)),
        "downloads": getattr(info, "downloads", None) or 0,
        "likes": getattr(info, "likes", None) or 0,
        "pipeline_tag": pipeline_tag,
        "tags": tags,
        "license": extract_license(card_data, tags),
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "sibling_files": files,
        "config": config,
        "tokenizer_config": tokenizer_config,
        "revision": getattr(info, "sha", None),
        "has_safetensors": has_safetensors_file(files),
        "has_tokenizer": has_tokenizer_file(files),
        "has_config": has_config_file(files),
        "has_weights": has_weight_file(files),
        "estimated_params_billion": estimate_params_billion(config, tags, model_id),
        "estimated_download_size_gb": estimate_download_size_gb(siblings),
        "architecture": extract_architectures(config),
        "context_length": extract_context_length(config, tags),
        "chat_template_present": tokenizer_has_chat_template(tokenizer_config),
        "model_card_present": bool(card_data),
    }


def _enrich_small_json_metadata(records: list[dict[str, Any]], scan_config: dict[str, Any]) -> None:
    limit = int(scan_config.get("max_candidates_to_run", 5))
    for candidate in records[: max(limit, 1)]:
        model_id = str(candidate.get("model_id"))
        if not candidate.get("config"):
            candidate["config"] = _download_optional_json(model_id, "config.json")
            candidate["architecture"] = extract_architectures(candidate["config"])
            candidate["context_length"] = extract_context_length(
                candidate["config"], candidate.get("tags") or []
            )
        tokenizer_config = _download_optional_json(model_id, "tokenizer_config.json")
        if tokenizer_config:
            candidate["tokenizer_config"] = tokenizer_config
            candidate["chat_template_present"] = tokenizer_has_chat_template(tokenizer_config)


def _download_optional_json(model_id: str, filename: str) -> dict[str, Any]:
    try:
        path = hf_hub_download(repo_id=model_id, filename=filename, local_files_only=False)
    except Exception:
        return {}
    try:
        return read_json_file(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _pipeline_allowed(candidate: dict[str, Any], scan_config: dict[str, Any]) -> bool:
    allowed = set(scan_config.get("pipeline_tags") or ["text-generation"])
    pipeline_tag = candidate.get("pipeline_tag")
    if pipeline_tag in allowed:
        return True
    tags = set(candidate.get("tags") or [])
    return bool(allowed & tags)


def _within_lookback(candidate: dict[str, Any], scan_config: dict[str, Any]) -> bool:
    lookback_days = int(scan_config.get("lookback_days", 2))
    raw_modified = candidate.get("last_modified")
    if not raw_modified:
        return True
    try:
        modified = datetime.fromisoformat(str(raw_modified).replace("Z", "+00:00"))
    except ValueError:
        return True
    if modified.tzinfo is None:
        modified = modified.replace(tzinfo=UTC)
    return modified.astimezone(UTC) >= datetime.now(UTC) - timedelta(days=lookback_days)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
