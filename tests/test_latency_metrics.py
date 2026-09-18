from __future__ import annotations

from hf_model_watch.evals.latency_bench import run_latency_benchmark
from hf_model_watch.evals.prompts import EvalPrompt
from hf_model_watch.runtime.backend_transformers import GenerationResult

PROMPTS = [
    EvalPrompt(
        "gsm8k_latency",
        "grade_school_math",
        "ground_truth_math",
        "medium",
        "Question: What is 2 + 2?",
        16,
        {"dataset": "openai/gsm8k", "final_answer": "4"},
    )
]


def test_latency_metrics_calculation(tmp_path) -> None:
    def generate(prompt: str, max_new_tokens: int) -> GenerationResult:
        return GenerationResult(
            prompt=prompt,
            text=f"{prompt} ok",
            prompt_tokens=10,
            generated_tokens=5,
            elapsed_seconds=0.2,
        )

    records, summary = run_latency_benchmark(
        model_id="local/model",
        backend="transformers",
        dtype="bf16",
        quantization=None,
        generate_fn=generate,
        output_path=tmp_path / "latency.jsonl",
        prompts=PROMPTS,
    )

    first = records[0]
    assert first["usage_source"] == "estimated"
    assert first["decode_latency_ms"] == 200
    assert first["decode_tokens_per_second"] == 25
    for removed in (
        "decode_latency_per_1k_output_tokens",
        "latency_per_1k_input_tokens",
        "latency_per_1k_output_tokens",
        "ms_per_output_token",
        "ttft_per_1k_input_tokens",
        "ttft_per_input_token",
    ):
        assert removed not in first
    assert first["ttft_measured"] is False
    assert first["prompt_text"]
    assert first["generated_text"]
    assert "expected" in first
    assert summary["successful_task_count"] == len(records)
    assert summary["avg_decode_tokens_per_second"] == 25


def test_latency_records_generation_failure_category(tmp_path) -> None:
    def generate(prompt: str, max_new_tokens: int) -> GenerationResult:
        raise RuntimeError("nope")

    records, summary = run_latency_benchmark(
        model_id="local/model",
        backend="transformers",
        dtype="bf16",
        quantization=None,
        generate_fn=generate,
        output_path=tmp_path / "latency.jsonl",
        prompts=PROMPTS,
    )

    assert records[0]["success"] is False
    assert records[0]["failure_category"] == "generation_failed"
    assert records[0]["prompt_text"]
    assert records[0]["generated_text"] is None
    assert summary["failed_task_count"] == len(records)
