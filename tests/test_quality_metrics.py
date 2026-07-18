from __future__ import annotations

from hf_model_watch.evals.judge import judge_json_schema, judge_numeric_answer, judge_sql
from hf_model_watch.evals.prompts import EvalPrompt
from hf_model_watch.evals.quality_eval import run_quality_benchmark
from hf_model_watch.runtime.backend_transformers import GenerationResult


def test_json_schema_quality_metric() -> None:
    result = judge_json_schema('{"ticker": "MSFT", "rating": "neutral"}', ["ticker", "rating"])

    assert result["success"] is True
    assert result["quality_score"] == 1.0


def test_sql_syntax_metric() -> None:
    result = judge_sql("SELECT * FROM users WHERE active = 1", ["select", "from", "where"])

    assert result["success"] is True
    assert result["quality_score"] == 1.0


def test_ground_truth_numeric_answer_metric() -> None:
    result = judge_numeric_answer("Reasoning... #### 1,250", "1250")

    assert result["success"] is True
    assert result["quality_score"] == 1.0


def test_quality_records_include_prompt_expected_and_output(tmp_path) -> None:
    def generate(prompt: str, max_new_tokens: int) -> GenerationResult:
        return GenerationResult(
            prompt=prompt,
            text='{"ticker":"MSFT","rating":"neutral"}',
            prompt_tokens=8,
            generated_tokens=4,
            elapsed_seconds=0.1,
        )

    records, _summary = run_quality_benchmark(
        model_id="local/model",
        backend="transformers",
        generate_fn=generate,
        output_path=tmp_path / "quality.jsonl",
        prompts=[
            EvalPrompt(
                "gsm8k_test",
                "grade_school_math",
                "ground_truth_math",
                "medium",
                "Question: What is 20 + 5?",
                32,
                {"dataset": "openai/gsm8k", "final_answer": "25"},
            )
        ],
    )

    first = records[0]
    assert first["prompt_text"]
    assert first["expected"]
    assert first["generated_text"]
    assert "judge_reason" in first
