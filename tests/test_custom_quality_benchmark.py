from __future__ import annotations

import json

import pytest

from hf_model_watch.evals.judge import judge_numeric_answer
from hf_model_watch.evals.prompts import EvalPrompt, load_quality_prompts
from hf_model_watch.evals.quality_eval import run_quality_benchmark
from hf_model_watch.runtime.backend_transformers import GenerationResult


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_loads_enabled_custom_quality_datasets(tmp_path) -> None:
    math_path = tmp_path / "math.jsonl"
    sentence_path = tmp_path / "sentence.jsonl"
    _write_jsonl(
        math_path,
        [
            {
                "id": "math_1",
                "question": "What is one half plus one quarter?",
                "expected_answer": "0.75",
                "answer_type": "decimal",
                "absolute_tolerance": "0.001",
                "source_url": "original",
                "license": "internal",
            }
        ],
    )
    _write_jsonl(
        sentence_path,
        [
            {
                "id": "sentence_1",
                "query": "I tested the patch and",
                "references": ["it works now.", "the issue is fixed."],
                "source_url": "https://example.test/discourse/1",
                "license": "CC-BY-4.0",
            }
        ],
    )

    prompts = load_quality_prompts(
        {
            "math": {"enabled": True, "dataset_path": str(math_path)},
            "sentence_completion": {
                "enabled": True,
                "dataset_path": str(sentence_path),
            },
        }
    )

    assert [prompt.task_type for prompt in prompts] == [
        "custom_math",
        "sentence_completion",
    ]
    assert prompts[0].expected["absolute_tolerance"] == "0.001"
    assert prompts[1].expected["references"] == ["it works now.", "the issue is fixed."]


def test_disabled_quality_tasks_do_not_read_placeholder_files() -> None:
    assert load_quality_prompts(
        {
            "math": {"enabled": False, "dataset_path": "does-not-exist.jsonl"},
            "sentence_completion": {
                "enabled": False,
                "dataset_path": "does-not-exist.jsonl",
            },
        }
    ) == []


def test_enabled_placeholder_dataset_fails_clearly(tmp_path) -> None:
    path = tmp_path / "math.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "math_1",
                "question": "<MATH QUESTION>",
                "expected_answer": "4",
                "answer_type": "integer",
                "source_url": "original",
                "license": "internal",
            }
        ],
    )

    with pytest.raises(ValueError, match="placeholder"):
        load_quality_prompts({"math": {"enabled": True, "dataset_path": str(path)}})


def test_numeric_math_accuracy_supports_fractions_and_tolerance() -> None:
    assert judge_numeric_answer("Final answer: 3/4", "0.75")["success"] is True
    assert judge_numeric_answer("Final answer: 0.7504", "0.75", "0.001")["success"] is True
    assert judge_numeric_answer("Final answer: 0.8", "0.75", "0.001")["success"] is False


def test_quality_benchmark_reports_task_specific_metrics(tmp_path) -> None:
    prompts = [
        EvalPrompt(
            "math_1",
            "custom_math",
            "custom_math",
            "easy",
            "What is 2 + 2?",
            16,
            {"final_answer": "4", "absolute_tolerance": "0"},
        ),
        EvalPrompt(
            "sentence_1",
            "sentence_completion",
            "online_discourse",
            "medium",
            "I tested the patch and",
            16,
            {"references": ["it works now.", "the issue is fixed."]},
        ),
    ]

    def generate(prompt: str, max_new_tokens: int) -> GenerationResult:
        text = "Final answer: 4" if "2 + 2" in prompt else "the problem is resolved."
        return GenerationResult(prompt, text, 5, 4, 0.05)

    def embedding_score(generated: str, references: list[str]) -> dict:
        assert generated == "the problem is resolved."
        assert len(references) == 2
        return {
            "embedding_similarity": 0.91,
            "embedding_similarities": [0.8, 0.91],
            "best_embedding_reference_index": 1,
        }

    def reference_score(prompt: str, reference: str) -> dict:
        bits_per_byte = 0.7 if reference.startswith("the issue") else 0.9
        return {
            "reference_cross_entropy": 1.2,
            "reference_perplexity": 3.32,
            "reference_negative_log_likelihood": 4.8,
            "reference_bits_per_byte": bits_per_byte,
            "reference_completion_tokens": 4,
            "reference_utf8_bytes": len(reference.encode()),
        }

    records, summary = run_quality_benchmark(
        model_id="local/model",
        backend="transformers",
        generate_fn=generate,
        output_path=tmp_path / "quality.jsonl",
        prompts=prompts,
        embedding_score_fn=embedding_score,
        reference_score_fn=reference_score,
    )

    assert summary["math_accuracy"] == 1.0
    assert summary["math_answer_latency_median_ms"] >= 50
    assert summary["sentence_embedding_similarity_mean"] == 0.91
    assert summary["sentence_reference_bits_per_byte_mean"] == 0.7
    assert records[1]["best_embedding_reference_index"] == 1
    assert records[1]["best_loss_reference_index"] == 1
