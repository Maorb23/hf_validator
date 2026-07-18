from __future__ import annotations

import json

from hf_model_watch.reporting.leaderboard import (
    build_leaderboard_rows,
    generate_leaderboard_reports,
)
from hf_model_watch.storage.sqlite_store import SQLiteStore


def test_leaderboard_sorts_by_best_quality() -> None:
    rows = build_leaderboard_rows(
        [
            {"run_id": "r1", "model_id": "org/a", "base_score": 0.4},
            {"run_id": "r2", "model_id": "org/b", "base_score": 0.8},
        ]
    )

    assert rows[0]["model_id"] == "org/b"
    assert rows[0]["rank"] == 1


def test_failed_runs_appear_with_failure_category(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    out_path = tmp_path / "reports" / "leaderboard.md"
    store = SQLiteStore(db_path)
    store.initialize()
    store.start_run("run_failed", "org/broken", "abc")
    store.record_failure("run_failed", "org/broken", "abc", "load_failed", "boom")

    generate_leaderboard_reports(db_path, out_path)

    text = out_path.read_text(encoding="utf-8")
    assert "org/broken" in text
    assert "load_failed" in text


def test_html_and_markdown_reports_are_generated(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    out_path = tmp_path / "reports" / "leaderboard.md"
    store = SQLiteStore(db_path)
    store.initialize()
    store.start_run("run_1", "org/model", "abc", model_family="Qwen", size_band="small")
    store.complete_run(
        "run_1",
        base_score=0.7,
        latency_summary=json.dumps({"total_tokens": 20, "successful_task_count": 2}),
    )

    generate_leaderboard_reports(db_path, out_path)

    assert out_path.exists()
    assert out_path.with_suffix(".html").exists()
    assert (out_path.parent / "runs" / "run_1.md").exists()
    assert (out_path.parent / "runs" / "run_1.html").exists()


def test_run_report_surfaces_eval_prompt_and_output(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    out_path = tmp_path / "reports" / "leaderboard.md"
    store = SQLiteStore(db_path)
    store.initialize()
    store.start_run("run_1", "org/model", "abc")
    store.record_quality_results(
        "run_1",
        "org/model",
        "abc",
        {"avg_quality_score": 1.0},
        [
            {
                "prompt_id": "qual_json",
                "task_type": "json_extraction",
                "prompt_text": "Extract JSON",
                "expected": {"ticker": "MSFT"},
                "generated_text": '{"ticker":"MSFT"}',
                "quality_score": 1.0,
                "success": True,
            }
        ],
    )
    store.complete_run("run_1", base_score=1.0)

    generate_leaderboard_reports(db_path, out_path)

    report = (out_path.parent / "runs" / "run_1.md").read_text(encoding="utf-8")
    assert "## Quality Prompts" in report
    assert "Extract JSON" in report
    assert '{"ticker":"MSFT"}' in report
