from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.evals.judge import judge_quality
from hf_model_watch.evals.metrics import add_token_normalized_metrics, summarize_records
from hf_model_watch.evals.prompts import EvalPrompt, load_ground_truth_prompts
from hf_model_watch.runtime.backend_transformers import GenerationResult

GeneratePromptFn = Callable[[str, int], GenerationResult]


def run_quality_benchmark(
    model_id: str,
    backend: str,
    generate_fn: GeneratePromptFn,
    output_path: str | Path,
    dataset_config: dict[str, Any] | None = None,
    prompts: list[EvalPrompt] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_prompts = prompts or load_ground_truth_prompts(dataset_config, "quality")
    records = [
        _run_one_quality_prompt(model_id, backend, prompt, generate_fn)
        for prompt in selected_prompts
    ]
    _write_jsonl(output_path, records)
    return records, summarize_records(records)


def _run_one_quality_prompt(
    model_id: str,
    backend: str,
    prompt: EvalPrompt,
    generate_fn: GeneratePromptFn,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        generation = generate_fn(prompt.prompt, prompt.max_new_tokens)
        total_latency_ms = max(
            generation.elapsed_seconds * 1000,
            (time.perf_counter() - start) * 1000,
        )
        judgment = judge_quality(prompt.task_type, generation.text, prompt.expected)
        base = {
            **_base_record(model_id, backend, prompt),
            "success": judgment["success"],
            "failure_category": (
                None if judgment["success"] else FailureCategory.QUALITY_EVAL_FAILED.value
            ),
            "quality_score": judgment["quality_score"],
            "judge": judgment,
            "judge_reason": judgment.get("reason"),
            "prompt_text": prompt.prompt,
            "expected": prompt.expected,
            "generated_text": generation.text,
            "prompt_tokens_estimated": generation.prompt_tokens,
            "completion_tokens_estimated": generation.generated_tokens,
            "total_tokens_estimated": generation.prompt_tokens + generation.generated_tokens,
            "ttft_ms": 0.0,
            "ttft_measured": False,
            "total_latency_ms": total_latency_ms,
        }
    except Exception as exc:
        total_latency_ms = (time.perf_counter() - start) * 1000
        base = {
            **_base_record(model_id, backend, prompt),
            "success": False,
            "failure_category": FailureCategory.GENERATION_FAILED.value,
            "failure_message": f"{type(exc).__name__}: {exc}",
            "quality_score": 0.0,
            "judge_reason": "generation_failed",
            "prompt_text": prompt.prompt,
            "expected": prompt.expected,
            "generated_text": None,
            "prompt_tokens_estimated": len(prompt.prompt.split()),
            "completion_tokens_estimated": 0,
            "total_tokens_estimated": len(prompt.prompt.split()),
            "ttft_ms": 0.0,
            "ttft_measured": False,
            "total_latency_ms": total_latency_ms,
        }
    return add_token_normalized_metrics(base)


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
