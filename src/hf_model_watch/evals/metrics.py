from __future__ import annotations

from typing import Any

SMALL_EPSILON = 1e-6


def effective_token_counts(record: dict[str, Any]) -> dict[str, int | str]:
    prompt_provider = record.get("prompt_tokens_provider")
    completion_provider = record.get("completion_tokens_provider")
    total_provider = record.get("total_tokens_provider")
    prompt_estimated = record.get("prompt_tokens_estimated")
    completion_estimated = record.get("completion_tokens_estimated")
    total_estimated = record.get("total_tokens_estimated")

    if prompt_provider is not None or completion_provider is not None or total_provider is not None:
        usage_source = "provider_reported"
    elif (
        prompt_estimated is not None
        or completion_estimated is not None
        or total_estimated is not None
    ):
        usage_source = "estimated"
    else:
        usage_source = "missing"

    effective_input = _coerce_int(prompt_provider, prompt_estimated, 0)
    effective_output = _coerce_int(completion_provider, completion_estimated, 0)
    effective_total = _coerce_int(
        total_provider,
        total_estimated,
        effective_input + effective_output,
    )
    return {
        "effective_input_tokens": effective_input,
        "effective_output_tokens": effective_output,
        "effective_total_tokens": effective_total,
        "usage_source": usage_source,
    }


def add_token_normalized_metrics(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    token_counts = effective_token_counts(enriched)
    enriched.update(token_counts)
    input_tokens = max(int(token_counts["effective_input_tokens"]), 1)
    output_tokens = max(int(token_counts["effective_output_tokens"]), 1)
    total_tokens = max(int(token_counts["effective_total_tokens"]), 1)

    ttft_ms = float(enriched.get("ttft_ms") or 0.0)
    total_latency_ms = float(enriched.get("total_latency_ms") or 0.0)
    decode_latency_ms = float(
        enriched.get("decode_latency_ms") or max(total_latency_ms - ttft_ms, 0.0)
    )
    enriched["decode_latency_ms"] = decode_latency_ms
    enriched["ttft_per_input_token"] = ttft_ms / input_tokens
    enriched["ttft_per_1k_input_tokens"] = 1000 * ttft_ms / input_tokens
    enriched["latency_per_1k_input_tokens"] = 1000 * total_latency_ms / input_tokens
    enriched["latency_per_1k_output_tokens"] = 1000 * total_latency_ms / output_tokens
    enriched["decode_latency_per_1k_output_tokens"] = 1000 * decode_latency_ms / output_tokens
    enriched["decode_tokens_per_second"] = int(token_counts["effective_output_tokens"]) / max(
        decode_latency_ms / 1000,
        SMALL_EPSILON,
    )
    enriched["ms_per_output_token"] = decode_latency_ms / output_tokens
    quality_score = enriched.get("quality_score")
    if quality_score is not None:
        enriched["quality_per_1k_tokens"] = 1000 * float(quality_score) / total_tokens
    return enriched


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("success", True)]
    failed = [record for record in records if not record.get("success", True)]
    return {
        "avg_ttft_per_1k_input_tokens": _avg(records, "ttft_per_1k_input_tokens"),
        "avg_latency_per_1k_input_tokens": _avg(records, "latency_per_1k_input_tokens"),
        "avg_latency_per_1k_output_tokens": _avg(records, "latency_per_1k_output_tokens"),
        "avg_decode_latency_per_1k_output_tokens": _avg(
            records,
            "decode_latency_per_1k_output_tokens",
        ),
        "avg_quality_per_1k_tokens": _avg(records, "quality_per_1k_tokens"),
        "total_prompt_tokens": sum(
            int(record.get("effective_input_tokens") or 0) for record in records
        ),
        "total_completion_tokens": sum(
            int(record.get("effective_output_tokens") or 0) for record in records
        ),
        "total_tokens": sum(int(record.get("effective_total_tokens") or 0) for record in records),
        "total_cached_prompt_tokens": sum(
            int(record.get("cached_prompt_tokens") or 0) for record in records
        ),
        "total_reasoning_tokens": sum(
            int(record.get("reasoning_tokens") or 0) for record in records
        ),
        "total_cost_estimate_usd": sum(
            float(record.get("cost_estimate_usd") or 0.0) for record in records
        ),
        "successful_task_count": len(successful),
        "failed_task_count": len(failed),
    }


def aggregate_by(records: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        bucket_key = tuple(record.get(key) for key in keys)
        buckets.setdefault(bucket_key, []).append(record)
    aggregated = []
    for bucket_key, bucket_records in buckets.items():
        row = {key: value for key, value in zip(keys, bucket_key, strict=False)}
        row.update(summarize_records(bucket_records))
        aggregated.append(row)
    return aggregated


def _coerce_int(primary: Any, fallback: Any, default: int) -> int:
    for value in (primary, fallback):
        if value is not None:
            return int(value)
    return default


def _avg(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)
