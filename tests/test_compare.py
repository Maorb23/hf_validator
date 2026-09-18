from __future__ import annotations

import json

from hf_model_watch.reporting.compare import compare_model, compare_run_to_prior
from hf_model_watch.storage.sqlite_store import SQLiteStore


def test_comparison_does_not_compare_against_untested_models(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    store = SQLiteStore(db_path)
    store.initialize()
    store.insert_scan_result(
        {"model_id": "org/scanned-only", "revision": "scan", "model_family": "Qwen"},
    )
    store.start_run("run_current", "org/current", "abc", model_family="Qwen", size_band="small")
    store.complete_run("run_current", base_score=0.7)

    result = compare_model(db_path, "org/current", tmp_path / "comparisons")

    assert result["comparison_status"] == "inconclusive"
    assert "same family" in result["reason"]


def test_comparison_inconclusive_without_prior_tested_model() -> None:
    current = {
        "run_id": "run_1",
        "model_id": "org/current",
        "base_score": 0.7,
        "model_family": "Qwen",
        "size_band": "small",
    }

    result = compare_run_to_prior(current, [])

    assert result["comparison_status"] == "inconclusive"


def test_comparison_similar_below_threshold() -> None:
    current = {
        "run_id": "run_current",
        "model_id": "org/current",
        "model_family": "Qwen",
        "size_band": "small",
        "base_score": 0.705,
        "latency_summary": json.dumps(
            {"avg_decode_tokens_per_second": 100.0, "total_tokens": 100}
        ),
        "context_drift_summary": json.dumps({"drift_score": 0.1}),
    }
    prior = {
        "run_id": "run_prior",
        "model_id": "org/prior",
        "model_family": "Qwen",
        "size_band": "small",
        "base_score": 0.7,
        "latency_summary": json.dumps(
            {"avg_decode_tokens_per_second": 101.0, "total_tokens": 100}
        ),
        "context_drift_summary": json.dumps({"drift_score": 0.1}),
        "status": "completed",
    }

    result = compare_run_to_prior(current, [prior])

    assert result["comparison_status"] == "similar"


def test_comparison_report_file_is_generated(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    store = SQLiteStore(db_path)
    store.initialize()
    store.start_run("run_prior", "org/prior", "aaa", model_family="Qwen", size_band="small")
    store.complete_run("run_prior", base_score=0.6)
    store.start_run("run_current", "org/current", "bbb", model_family="Qwen", size_band="small")
    store.complete_run("run_current", base_score=0.7)

    result = compare_model(db_path, "org/current", tmp_path / "comparisons")

    assert result["report_path"].endswith("org-current.md")
    assert (tmp_path / "comparisons" / "org-current.md").exists()
