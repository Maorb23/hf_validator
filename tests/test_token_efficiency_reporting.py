from __future__ import annotations

import json

from hf_model_watch.reporting.leaderboard import build_leaderboard_rows, render_run_reports


def test_token_efficiency_metrics_are_calculated() -> None:
    rows = build_leaderboard_rows(
        [
            {
                "run_id": "run_1",
                "model_id": "org/model",
                "base_score": 0.5,
                "latency_summary": json.dumps(
                    {
                        "total_tokens": 100,
                        "total_prompt_tokens": 80,
                        "total_completion_tokens": 20,
                        "successful_task_count": 4,
                        "total_cost_estimate_usd": 2.0,
                    }
                ),
            }
        ]
    )

    row = rows[0]
    assert row["quality_per_1k_tokens"] == 5
    assert row["cost_per_successful_task"] == 0.5
    assert row["tokens_per_successful_task"] == 25


def test_missing_token_metrics_render_as_na(tmp_path) -> None:
    paths = render_run_reports(
        {"run_id": "run_1", "model_id": "org/model", "status": "completed"},
        tmp_path,
    )

    text = paths["markdown"].read_text(encoding="utf-8")
    assert "Input tokens: N/A" in text
    assert "Quality per 1k tokens: N/A" in text


def test_estimated_vs_provider_reported_usage_is_surfaced() -> None:
    estimated = build_leaderboard_rows(
        [
            {
                "run_id": "run_1",
                "model_id": "org/model",
                "latency_results": {
                    "payload_json": json.dumps(
                        [{"usage_source": "estimated", "effective_total_tokens": 10}]
                    )
                },
            }
        ]
    )[0]
    mixed = build_leaderboard_rows(
        [
            {
                "run_id": "run_2",
                "model_id": "org/model2",
                "latency_results": {
                    "payload_json": json.dumps(
                        [
                            {"usage_source": "estimated", "effective_total_tokens": 10},
                            {"usage_source": "provider_reported", "effective_total_tokens": 10},
                        ]
                    )
                },
            }
        ]
    )[0]

    assert estimated["usage_source_summary"] == "estimated"
    assert mixed["usage_source_summary"] == "mixed"
