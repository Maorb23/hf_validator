from __future__ import annotations

import sqlite3

from hf_model_watch.reporting.leaderboard import generate_leaderboard_markdown
from hf_model_watch.storage.sqlite_store import SQLiteStore


def test_db_initializes(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    store = SQLiteStore(db_path)

    store.initialize()

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "models" in tables
    assert "runs" in tables
    assert "artifacts" in tables
    assert "failures" in tables


def test_insert_and_read_run(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()

    store.start_run(
        run_id="run_1",
        model_id="org/model",
        revision="abc123",
        model_family="Qwen",
        size_band="medium",
        artifact_uri="artifacts/run_1",
    )
    run = store.fetch_run("run_1")

    assert run is not None
    assert run["model_id"] == "org/model"
    assert run["status"] == "started"
    assert run["artifact_uri"] == "artifacts/run_1"


def test_record_failure(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    store.start_run("run_1", "org/model", "abc123")

    store.record_failure("run_1", "org/model", "abc123", "load_failed", "boom")
    run = store.fetch_run("run_1")

    assert run is not None
    assert run["status"] == "failed"
    assert run["failure_category"] == "load_failed"
    assert run["failure_message"] == "boom"


def test_already_tested_revision_works(tmp_path) -> None:
    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    store.start_run("run_1", "org/model", "abc123")

    assert store.already_tested_revision("org/model", "abc123") is False

    store.complete_run("run_1")

    assert store.already_tested_revision("org/model", "abc123") is True
    assert store.already_tested_revision("org/model", "other") is False


def test_leaderboard_generates_markdown_file(tmp_path) -> None:
    db_path = tmp_path / "results.sqlite"
    out_path = tmp_path / "leaderboard.md"
    store = SQLiteStore(db_path)
    store.initialize()
    store.start_run("run_1", "org/model", "abc123", model_family="Qwen", size_band="medium")
    store.complete_run("run_1", base_score=0.7)

    generate_leaderboard_markdown(db_path, out_path)

    text = out_path.read_text(encoding="utf-8")
    assert "# hf-model-watch Leaderboard" in text
    assert "org/model" in text
    assert "0.7000" in text
