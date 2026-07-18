from __future__ import annotations

from hf_model_watch.evals.context_drift import run_context_drift_benchmark
from hf_model_watch.runtime.backend_transformers import GenerationResult


def test_context_drift_scoring(tmp_path) -> None:
    def generate(prompt: str, max_new_tokens: int) -> GenerationResult:
        return GenerationResult(
            prompt=prompt,
            text="NEEDLE42 Monday Friday contradiction blue",
            prompt_tokens=100,
            generated_tokens=5,
            elapsed_seconds=0.1,
        )

    records, summary = run_context_drift_benchmark(
        model_id="local/model",
        backend="transformers",
        claimed_context=2048,
        max_context_tokens=512,
        generate_fn=generate,
        output_path=tmp_path / "context.jsonl",
    )

    assert all(record["success"] for record in records)
    assert records[0]["base_prompt_text"]
    assert records[0]["rendered_prompt_text"]
    assert records[0]["expected"]
    assert records[0]["generated_text"]
    assert "judge_reason" in records[0]
    assert summary["success_rate"] == 1.0
    assert summary["drift_score"] == 0.0
