from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.evals.metrics import add_token_normalized_metrics, summarize_records
from hf_model_watch.evals.prompts import EvalPrompt, load_ground_truth_prompts
from hf_model_watch.runtime.backend_transformers import GenerationResult

GeneratePromptFn = Callable[[str, int], GenerationResult]


def run_latency_benchmark(
    model_id: str,
    backend: str,
    dtype: str | None,
    quantization: str | None,
    generate_fn: GeneratePromptFn,
    output_path: str | Path,
    dataset_config: dict[str, Any] | None = None,
    prompts: list[EvalPrompt] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_prompts = prompts or load_ground_truth_prompts(dataset_config, "latency")
    records = []
    warmup = selected_prompts[0]
    try:
        generate_fn(warmup.prompt, min(warmup.max_new_tokens, 8))
    except Exception:
        pass

    for prompt in selected_prompts:
        records.append(
            _run_one_latency_prompt(
                model_id=model_id,
                backend=backend,
                dtype=dtype,
                quantization=quantization,
                prompt=prompt,
                generate_fn=generate_fn,
            )
        )
    _write_jsonl(output_path, records)
    return records, summarize_records(records)


def _run_one_latency_prompt(
    model_id: str,
    backend: str,
    dtype: str | None,
    quantization: str | None,
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
        base = {
            **_base_usage_record(model_id, backend, prompt),
            "dtype": dtype,
            "quantization": quantization,
            "success": True,
            "failure_category": None,
            "prompt_text": prompt.prompt,
            "expected": prompt.expected,
            "generated_text": generation.text,
            "prompt_tokens_estimated": generation.prompt_tokens,
            "completion_tokens_estimated": generation.generated_tokens,
            "total_tokens_estimated": generation.prompt_tokens + generation.generated_tokens,
            "ttft_ms": 0.0,
            "ttft_measured": False,
            "total_latency_ms": total_latency_ms,
            "max_new_tokens": prompt.max_new_tokens,
            "batch_size": 1,
            "gpu_peak_memory": None,
        }
    except Exception as exc:
        total_latency_ms = (time.perf_counter() - start) * 1000
        base = {
            **_base_usage_record(model_id, backend, prompt),
            "dtype": dtype,
            "quantization": quantization,
            "success": False,
            "failure_category": FailureCategory.GENERATION_FAILED.value,
            "failure_message": f"{type(exc).__name__}: {exc}",
            "prompt_text": prompt.prompt,
            "expected": prompt.expected,
            "generated_text": None,
            "prompt_tokens_estimated": len(prompt.prompt.split()),
            "completion_tokens_estimated": 0,
            "total_tokens_estimated": len(prompt.prompt.split()),
            "ttft_ms": 0.0,
            "ttft_measured": False,
            "total_latency_ms": total_latency_ms,
            "max_new_tokens": prompt.max_new_tokens,
            "batch_size": 1,
            "gpu_peak_memory": None,
        }
    return add_token_normalized_metrics(base)


def _base_usage_record(model_id: str, backend: str, prompt: EvalPrompt) -> dict[str, Any]:
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
