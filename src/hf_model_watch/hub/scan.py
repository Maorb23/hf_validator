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
        _enrich_candidate_metadata(records, scan_config, client)
        _apply_context_fallback(records, scan_config)
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
    embedded_tokenizer_config = config.get("tokenizer_config")
    tokenizer_config = (
        embedded_tokenizer_config if isinstance(embedded_tokenizer_config, dict) else {}
    )
    card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    pipeline_tag = getattr(info, "pipeline_tag", None) or getattr(info, "pipelineTag", None)
    context_length = extract_context_length(config, tags)

    return {
        "model_id": model_id,
        "author": getattr(info, "author", None) or extract_author(model_id),
        "created_at": normalize_datetime(
            getattr(info, "created_at", None) or getattr(info, "createdAt", None)
        ),
        "last_modified": normalize_datetime(
            getattr(info, "last_modified", None) or getattr(info, "lastModified", None)
        ),
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
        "estimated_params_billion": _safetensors_params_billion(info)
        or estimate_params_billion(config, tags, model_id),
        "estimated_download_size_gb": estimate_download_size_gb(siblings),
        "architecture": extract_architectures(config),
        "context_length": context_length,
        "context_length_source": "metadata" if context_length is not None else None,
        "chat_template_present": _has_chat_template(tokenizer_config, config, files),
        "model_card_present": bool(card_data),
    }


def _enrich_candidate_metadata(
    records: list[dict[str, Any]],
    scan_config: dict[str, Any],
    client: HfApi,
    *,
    require_license: bool = True,
) -> None:
    candidate_count = int(scan_config.get("max_candidates_to_run", 5))
    min_downloads = int(scan_config.get("min_downloads", 0))
    pool = [
        candidate
        for candidate in records
        if int(candidate.get("downloads") or 0) >= min_downloads
        and (candidate.get("license") or not require_license)
    ]
    for candidate in pool[: max(candidate_count * 4, 1)]:
        model_id = str(candidate.get("model_id"))
        revision = str(candidate.get("revision") or "") or None
        try:
            detailed_info = client.model_info(
                model_id,
                revision=revision,
                files_metadata=True,
            )
        except Exception:
            detailed_info = None
        if detailed_info is not None:
            siblings = getattr(detailed_info, "siblings", None) or []
            files = sibling_names(siblings)
            candidate["sibling_files"] = files
            candidate["estimated_download_size_gb"] = estimate_download_size_gb(siblings)
            candidate["estimated_params_billion"] = (
                _safetensors_params_billion(detailed_info)
                or candidate.get("estimated_params_billion")
            )

        downloaded_config = _download_optional_json(model_id, "config.json", revision)
        if downloaded_config:
            existing_config = candidate.get("config")
            candidate["config"] = {
                **(existing_config if isinstance(existing_config, dict) else {}),
                **downloaded_config,
            }
        config = candidate.get("config") if isinstance(candidate.get("config"), dict) else {}
        candidate["architecture"] = extract_architectures(config)
        context_length = extract_context_length(config, candidate.get("tags") or [])
        candidate["context_length"] = context_length
        candidate["context_length_source"] = (
            "metadata" if context_length is not None else None
        )
        candidate["estimated_params_billion"] = candidate.get(
            "estimated_params_billion"
        ) or estimate_params_billion(config, candidate.get("tags") or [], model_id)

        tokenizer_config = _download_optional_json(model_id, "tokenizer_config.json", revision)
        if tokenizer_config:
            candidate["tokenizer_config"] = tokenizer_config
        candidate["chat_template_present"] = _has_chat_template(
            candidate.get("tokenizer_config")
            if isinstance(candidate.get("tokenizer_config"), dict)
            else {},
            config,
            candidate.get("sibling_files") or [],
        )


def _apply_context_fallback(
    records: list[dict[str, Any]],
    scan_config: dict[str, Any],
) -> None:
    fallback = int(scan_config.get("fallback_context_length", 2048))
    for candidate in records:
        if candidate.get("context_length") is None:
            candidate["context_length"] = fallback
            candidate["context_length_source"] = "fallback"


def _download_optional_json(
    model_id: str,
    filename: str,
    revision: str | None = None,
) -> dict[str, Any]:
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            local_files_only=False,
        )
    except Exception:
        return {}
    try:
        return read_json_file(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _safetensors_params_billion(model_info: Any) -> float | None:
    safetensors = getattr(model_info, "safetensors", None)
    total = getattr(safetensors, "total", None)
    if isinstance(safetensors, dict):
        total = total or safetensors.get("total")
    if isinstance(total, int | float) and total > 0:
        return round(float(total) / 1_000_000_000, 3)
    return None


def _has_chat_template(
    tokenizer_config: dict[str, Any],
    config: dict[str, Any],
    files: list[str],
) -> bool:
    return (
        tokenizer_has_chat_template(tokenizer_config)
        or bool(config.get("chat_template") or config.get("chat_template_jinja"))
        or any(Path(file_name).name == "chat_template.jinja" for file_name in files)
    )


def _pipeline_allowed(candidate: dict[str, Any], scan_config: dict[str, Any]) -> bool:
    allowed = set(scan_config.get("pipeline_tags") or ["text-generation"])
    pipeline_tag = candidate.get("pipeline_tag")
    # An explicit Hub pipeline tag is authoritative. Multimodal repositories often
    # also carry a generic text-generation tag, but that does not make their model
    # class compatible with this text-only worker.
    if pipeline_tag:
        return pipeline_tag in allowed
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
