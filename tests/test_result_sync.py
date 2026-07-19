from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from hf_model_watch.reporting.result_sync import ingest_local_results, sync_s3_results
from hf_model_watch.storage.sqlite_store import SQLiteStore


def _write_run(root: Path, run_id: str = "run_sync_1", adapted_score: float = 0.8) -> Path:
    run_dir = root / run_id
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "model_id": "org/model",
        "revision": "abc123",
        "started_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:05:00+00:00",
        "status": "completed",
        "artifacts": {
            "runtime_result": f"artifacts/{run_id}/runtime_result.json",
            "adapter_json": f"artifacts/{run_id}/finetune/json/adapter",
        },
    }
    runtime = {
        "run_id": run_id,
        "revision": "abc123",
        "metadata": {
            "model_id": "org/model",
            "estimated_params_billion": 8.0,
            "config": {"model_type": "qwen"},
        },
        "decision": {"backend": "transformers", "dtype": "bf16", "quantization": None},
    }
    adaptation = {
        "run_id": run_id,
        "model_id": "org/model",
        "revision": "abc123",
        "created_at": "2026-01-01T00:10:00+00:00",
        "status": "training_completed",
        "training_completed": True,
        "overall": {
            "base_quality_score": 0.2,
            "adapted_quality_score": adapted_score,
            "adaptation_lift": adapted_score - 0.2,
        },
        "task_results": [],
    }
    latency = {
        "success": True,
        "effective_input_tokens": 10,
        "effective_output_tokens": 5,
        "effective_total_tokens": 15,
        "decode_tokens_per_second": 25.0,
        "ttft_ms": 0.0,
    }
    quality = {
        **latency,
        "quality_score": 0.6,
        "quality_per_1k_tokens": 40.0,
    }
    context = {
        **quality,
        "claimed_context": 2048,
        "actual_context_tested": 512,
    }
    files = {
        "run_manifest.json": manifest,
        "runtime_result.json": runtime,
        "adaptation_results.json": adaptation,
        "metadata/claims_vs_observed.json": {"claims": {}, "observed": {}},
    }
    for relative, payload in files.items():
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    for relative, payload in (
        ("latency_results.jsonl", latency),
        ("quality_results.jsonl", quality),
        ("context_drift_results.jsonl", context),
    ):
        (run_dir / relative).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return run_dir


def test_local_result_ingestion_is_idempotent_and_updates_leaderboard(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    _write_run(outputs)
    db = tmp_path / "results.sqlite"
    leaderboard = tmp_path / "reports" / "leaderboard.md"

    first = ingest_local_results(outputs, db, leaderboard)
    second = ingest_local_results(outputs, db, leaderboard)

    assert first.imported_runs == 1
    assert second.skipped_runs == 1
    run = SQLiteStore(db).fetch_reporting_run("run_sync_1")
    assert run is not None
    assert run["adapted_score"] == 0.8
    assert "org/model" in leaderboard.read_text(encoding="utf-8")
    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM adaptation_results WHERE run_id = 'run_sync_1'"
        ).fetchone()[0] == 1


def test_changed_local_run_replaces_previous_payload(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    run_dir = _write_run(outputs, adapted_score=0.8)
    db = tmp_path / "results.sqlite"
    leaderboard = tmp_path / "leaderboard.md"
    ingest_local_results(outputs, db, leaderboard)
    _write_run(outputs, adapted_score=0.9)

    summary = ingest_local_results(outputs, db, leaderboard)

    assert summary.imported_runs == 1
    assert SQLiteStore(db).fetch_run(run_dir.name)["adapted_score"] == 0.9


class _Paginator:
    def __init__(self, objects):
        self.objects = objects

    def paginate(self, **kwargs):
        yield {"Contents": self.objects}


class _FakeS3:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.downloaded = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        objects = [
            {"Key": key, "ETag": f'"{len(value)}"', "Size": len(value)}
            for key, value in self.files.items()
        ]
        return _Paginator(objects)

    def download_file(self, bucket, key, destination):
        self.downloaded.append(key)
        Path(destination).write_bytes(self.files[key])


def test_s3_sync_downloads_reporting_files_but_not_adapter_weights(tmp_path) -> None:
    local = _write_run(tmp_path / "source")
    files = {}
    for path in local.rglob("*"):
        if path.is_file():
            files[f"hf-model-watch/{local.name}/{path.relative_to(local).as_posix()}"] = (
                path.read_bytes()
            )
    files[f"hf-model-watch/{local.name}/finetune/json/adapter/adapter_model.safetensors"] = (
        b"large"
    )
    client = _FakeS3(files)
    kwargs = {
        "bucket": "bucket",
        "prefix": "hf-model-watch",
        "endpoint_url": "https://storage.example",
        "db_path": tmp_path / "results.sqlite",
        "cache_dir": tmp_path / "cache",
        "leaderboard_path": tmp_path / "leaderboard.md",
        "s3_client": client,
    }

    first = sync_s3_results(**kwargs)
    second = sync_s3_results(**kwargs)

    assert first.imported_runs == 1
    assert first.downloaded_files == 7
    assert second.skipped_runs == 1
    assert not any(key.endswith(".safetensors") for key in client.downloaded)
