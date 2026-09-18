from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.evals.judge import judge_quality
from hf_model_watch.evals.metrics import add_token_normalized_metrics, summarize_records
from hf_model_watch.evals.prompts import EvalPrompt, load_quality_prompts
from hf_model_watch.runtime.backend_transformers import GenerationResult

GeneratePromptFn = Callable[[str, int], GenerationResult]
EmbeddingScoreFn = Callable[[str, list[str]], dict[str, Any]]
ReferenceScoreFn = Callable[[str, str], dict[str, Any]]


def run_quality_benchmark(
    model_id: str,
    backend: str,
    generate_fn: GeneratePromptFn,
    output_path: str | Path,
    dataset_config: dict[str, Any] | None = None,
    prompts: list[EvalPrompt] | None = None,
    embedding_score_fn: EmbeddingScoreFn | None = None,
    reference_score_fn: ReferenceScoreFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_prompts = prompts if prompts is not None else load_quality_prompts(dataset_config)
    if embedding_score_fn is None and any(
        prompt.task_type == "sentence_completion" for prompt in selected_prompts
    ):
        embedding_score_fn = _build_embedding_scorer(dataset_config)
    records = [
        _run_one_quality_prompt(
            model_id,
            backend,
            prompt,
            generate_fn,
            embedding_score_fn,
            reference_score_fn,
        )
        for prompt in selected_prompts
    ]
    _write_jsonl(output_path, records)
    return records, summarize_quality_records(records)


def _run_one_quality_prompt(
    model_id: str,
    backend: str,
    prompt: EvalPrompt,
    generate_fn: GeneratePromptFn,
    embedding_score_fn: EmbeddingScoreFn | None,
    reference_score_fn: ReferenceScoreFn | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        generation = generate_fn(prompt.prompt, prompt.max_new_tokens)
        total_latency_ms = max(
            generation.elapsed_seconds * 1000,
            (time.perf_counter() - start) * 1000,
        )
        judgment = (
            _score_sentence_completion(
                generation.text,
                prompt,
                embedding_score_fn,
                reference_score_fn,
            )
            if prompt.task_type == "sentence_completion"
            else judge_quality(prompt.task_type, generation.text, prompt.expected)
        )
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
        if prompt.task_type == "custom_math":
            base["answer_latency_ms"] = total_latency_ms
        base.update(judgment.get("metrics") or {})
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
        if prompt.task_type == "custom_math":
            base["answer_latency_ms"] = total_latency_ms
    return add_token_normalized_metrics(base)


def _score_sentence_completion(
    generated: str,
    prompt: EvalPrompt,
    embedding_score_fn: EmbeddingScoreFn | None,
    reference_score_fn: ReferenceScoreFn | None,
) -> dict[str, Any]:
    references = [str(value) for value in prompt.expected.get("references", [])]
    metrics: dict[str, Any] = {}
    errors: list[str] = []
    if embedding_score_fn is None:
        errors.append("embedding scorer unavailable")
    else:
        try:
            metrics.update(embedding_score_fn(generated, references))
        except Exception as exc:
            errors.append(f"embedding: {type(exc).__name__}: {exc}")
    if reference_score_fn is not None:
        reference_scores = []
        for reference in references:
            try:
                reference_scores.append(reference_score_fn(prompt.prompt, reference))
            except Exception as exc:
                errors.append(f"reference loss: {type(exc).__name__}: {exc}")
        if reference_scores:
            best_index = min(
                range(len(reference_scores)),
                key=lambda index: float(reference_scores[index]["reference_bits_per_byte"]),
            )
            metrics.update(reference_scores[best_index])
            metrics["reference_scores"] = reference_scores
            metrics["best_loss_reference_index"] = best_index
    similarity = metrics.get("embedding_similarity")
    success = similarity is not None
    return {
        "quality_score": float(similarity) if similarity is not None else 0.0,
        "success": success,
        "reason": "ok" if success else "sentence_scoring_failed",
        "errors": errors,
        "metrics": metrics,
    }


def summarize_quality_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_records(records)
    math_records = [record for record in records if record.get("task_type") == "custom_math"]
    sentence_records = [
        record for record in records if record.get("task_type") == "sentence_completion"
    ]
    math_latencies = _values(math_records, "answer_latency_ms")
    similarities = _values(sentence_records, "embedding_similarity")
    bits_per_byte = _values(sentence_records, "reference_bits_per_byte")
    summary.update(
        {
            "status": "completed" if records else "disabled",
            "math_example_count": len(math_records),
            "math_accuracy": (
                sum(1 for record in math_records if record.get("success")) / len(math_records)
                if math_records
                else None
            ),
            "math_answer_latency_mean_ms": _mean(math_latencies),
            "math_answer_latency_median_ms": _median(math_latencies),
            "math_answer_latency_p90_ms": _percentile(math_latencies, 0.90),
            "math_answer_latency_p95_ms": _percentile(math_latencies, 0.95),
            "sentence_example_count": len(sentence_records),
            "sentence_embedding_similarity_mean": _mean(similarities),
            "sentence_embedding_similarity_median": _median(similarities),
            "sentence_embedding_similarity_p10": _percentile(similarities, 0.10),
            "sentence_reference_bits_per_byte_mean": _mean(bits_per_byte),
            "sentence_reference_bits_per_byte_median": _median(bits_per_byte),
        }
    )
    return summary


def _build_embedding_scorer(config: dict[str, Any] | None) -> EmbeddingScoreFn:
    from hf_model_watch.evals.semantic_similarity import SentenceEmbeddingScorer

    sentence = (config or {}).get("sentence_completion") or {}
    return SentenceEmbeddingScorer(
        model_id=str(sentence.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B"),
        revision=(
            str(sentence["embedding_revision"]) if sentence.get("embedding_revision") else None
        ),
        instruction=(
            str(sentence["embedding_instruction"])
            if sentence.get("embedding_instruction")
            else None
        ),
    )


def _values(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(record[key]) for record in records if record.get(key) is not None]


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]


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
