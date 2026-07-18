from __future__ import annotations

from hf_model_watch.evals.metrics import add_token_normalized_metrics, effective_token_counts


def test_effective_token_fallback_provider_reported() -> None:
    counts = effective_token_counts(
        {
            "prompt_tokens_provider": 7,
            "completion_tokens_provider": 3,
            "total_tokens_provider": 10,
            "prompt_tokens_estimated": 70,
            "completion_tokens_estimated": 30,
        }
    )

    assert counts["effective_input_tokens"] == 7
    assert counts["effective_output_tokens"] == 3
    assert counts["usage_source"] == "provider_reported"


def test_effective_token_fallback_estimated() -> None:
    counts = effective_token_counts(
        {"prompt_tokens_estimated": 11, "completion_tokens_estimated": 4}
    )

    assert counts["effective_input_tokens"] == 11
    assert counts["effective_output_tokens"] == 4
    assert counts["usage_source"] == "estimated"


def test_effective_token_fallback_missing() -> None:
    counts = effective_token_counts({})

    assert counts["effective_input_tokens"] == 0
    assert counts["effective_output_tokens"] == 0
    assert counts["usage_source"] == "missing"


def test_quality_per_1k_tokens() -> None:
    record = add_token_normalized_metrics(
        {
            "prompt_tokens_estimated": 80,
            "completion_tokens_estimated": 20,
            "total_tokens_estimated": 100,
            "ttft_ms": 50,
            "total_latency_ms": 150,
            "quality_score": 0.5,
        }
    )

    assert record["decode_latency_ms"] == 100
    assert record["ttft_per_1k_input_tokens"] == 625
    assert record["latency_per_1k_input_tokens"] == 1875
    assert record["latency_per_1k_output_tokens"] == 7500
    assert record["decode_latency_per_1k_output_tokens"] == 5000
    assert record["quality_per_1k_tokens"] == 5
