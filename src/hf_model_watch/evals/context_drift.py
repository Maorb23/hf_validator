from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.evals.judge import judge_quality
from hf_model_watch.evals.metrics import add_token_normalized_metrics, summarize_records
from hf_model_watch.evals.prompts import CONTEXT_PROMPTS, EvalPrompt
from hf_model_watch.runtime.backend_transformers import GenerationResult

GeneratePromptFn = Callable[[str, int], GenerationResult]


def run_context_drift_benchmark(
    model_id: str,
    backend: str,
    claimed_context: int | None,
    max_context_tokens: int,
    generate_fn: GeneratePromptFn,
    output_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for index, prompt in enumerate(CONTEXT_PROMPTS, start=1):
        context_tokens = min(max_context_tokens, max(64, index * 256))
        records.append(
            _run_one_context_prompt(
                model_id=model_id,
                backend=backend,
                claimed_context=claimed_context,
                actual_context_tested=min(context_tokens, claimed_context or context_tokens),
                prompt=prompt,
                generate_fn=generate_fn,
            )
        )
    success_rate = sum(1 for record in records if record["success"]) / max(len(records), 1)
    summary = summarize_records(records)
    summary.update(
        {
            "claimed_context": claimed_context,
            "actual_context_tested": max(record["actual_context_tested"] for record in records),
            "success_rate": success_rate,
            "drift_score": 1.0 - success_rate,
            "failure_mode": "none" if success_rate == 1 else "rule_based_context_failures",
        }
    )
    _write_jsonl(output_path, records)
    return records, summary


def _run_one_context_prompt(
    model_id: str,
    backend: str,
    claimed_context: int | None,
    actual_context_tested: int,
    prompt: EvalPrompt,
    generate_fn: GeneratePromptFn,
) -> dict[str, Any]:
    padded_prompt = _pad_prompt(prompt.prompt, actual_context_tested)
    start = time.perf_counter()
    try:
        generation = generate_fn(padded_prompt, prompt.max_new_tokens)
        total_latency_ms = max(
            generation.elapsed_seconds * 1000,
            (time.perf_counter() - start) * 1000,
        )
        judgment = judge_quality(prompt.task_type, generation.text, prompt.expected)
        base = {
            **_base_record(model_id, backend, prompt),
            "success": judgment["success"],
            "failure_category": (
                None if judgment["success"] else FailureCategory.CONTEXT_TEST_FAILED.value
            ),
            "quality_score": judgment["quality_score"],
            "context_tokens": generation.prompt_tokens,
            "claimed_context": claimed_context,
            "actual_context_tested": actual_context_tested,
            "base_prompt_text": prompt.prompt,
            "prompt_text": padded_prompt,
            "rendered_prompt_text": padded_prompt,
            "expected": prompt.expected,
            "generated_text": generation.text,
            "judge_reason": judgment.get("reason"),
            "success_rate": 1.0 if judgment["success"] else 0.0,
            "drift_score": 0.0 if judgment["success"] else 1.0,
            "failure_mode": None if judgment["success"] else judgment["reason"],
            "prompt_tokens_estimated": generation.prompt_tokens,
            "completion_tokens_estimated": generation.generated_tokens,
            "total_tokens_estimated": generation.prompt_tokens + generation.generated_tokens,
            "ttft_ms": 0.0,
            "ttft_measured": False,
            "total_latency_ms": total_latency_ms,
        }
    except Exception as exc:
        total_latency_ms = (time.perf_counter() - start) * 1000
        token_estimate = len(padded_prompt.split())
        base = {
            **_base_record(model_id, backend, prompt),
            "success": False,
            "failure_category": FailureCategory.GENERATION_FAILED.value,
            "failure_message": f"{type(exc).__name__}: {exc}",
            "quality_score": 0.0,
            "context_tokens": token_estimate,
            "claimed_context": claimed_context,
            "actual_context_tested": actual_context_tested,
            "base_prompt_text": prompt.prompt,
            "prompt_text": padded_prompt,
            "rendered_prompt_text": padded_prompt,
            "expected": prompt.expected,
            "generated_text": None,
            "judge_reason": "generation_failed",
            "success_rate": 0.0,
            "drift_score": 1.0,
            "failure_mode": "generation_failed",
            "prompt_tokens_estimated": token_estimate,
            "completion_tokens_estimated": 0,
            "total_tokens_estimated": token_estimate,
            "ttft_ms": 0.0,
            "ttft_measured": False,
            "total_latency_ms": total_latency_ms,
        }
    return add_token_normalized_metrics(base)


def _pad_prompt(prompt: str, target_tokens: int) -> str:
    filler = " context" * max(0, min(target_tokens, 1024) // 2)
    return f"{prompt}\nAdditional context:{filler}"


def _base_record(model_id: str, backend: str, prompt: EvalPrompt) -> dict[str, Any]:
    return {
        "provider": "local",
        "router_name": "local",
        "model_id": model_id,
        "backend": backend,
        "task_type": prompt.task_type,
        "prompt_id": prompt.prompt_id,
        "prompt_family": prompt.prompt_family,
        "difficulty_bucket": prompt.difficulty_bucket,
        "prompt_tokens_provider": None,
        "completion_tokens_provider": None,
        "total_tokens_provider": None,
        "cached_prompt_tokens": None,
        "reasoning_tokens": None,
        "router_overhead_ms": None,
        "queue_time_ms": None,
        "cost_estimate_usd": None,
    }


def _write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
