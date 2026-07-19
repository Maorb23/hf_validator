from __future__ import annotations

import json
from pathlib import Path

from hf_model_watch.orchestration.prefect_flow import build_schedule, run_prefect_controller_flow
from hf_model_watch.orchestration.result_sync_flow import run_prefect_result_sync_flow
from hf_model_watch.reporting.result_sync import SyncSummary


def test_dry_run_prefect_flow_selects_mock_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(Path.cwd())
    submitted = []

    def submit_job(payload, config):
        from hf_model_watch.orchestration.nebius_jobs import NebiusJobResult

        submitted.append(payload)
        return NebiusJobResult(
            job_id=f"dry-run-{payload.run_id}",
            status="submitted_dry_run",
            payload=payload.model_dump(),
            command=["bash", "scripts/run_model_worker_job.sh"],
            env={},
            dry_run=True,
        )

    def poll_job(job_id, config):
        return "completed_dry_run"

    summary = run_prefect_controller_flow(
        "config",
        dry_run=True,
        submit_job=submit_job,
        poll_job=poll_job,
    )

    assert summary["dry_run"] is True
    assert summary["selected_candidates"] == 1
    assert len(submitted) == 1
    assert submitted[0].model_id == "sshleifer/tiny-gpt2"


def test_prefect_flow_uses_provided_candidates(tmp_path, capsys) -> None:
    submitted = []

    def submit_job(payload, config):
        from hf_model_watch.orchestration.nebius_jobs import NebiusJobResult

        submitted.append(payload)
        return NebiusJobResult(
            job_id="dry-run-run_custom",
            status="submitted_dry_run",
            payload=payload.model_dump(),
            command=[],
            env={},
            dry_run=True,
        )

    summary = run_prefect_controller_flow(
        "config",
        dry_run=True,
        candidates=[
            {"model_id": "org/a", "revision": "a", "eligible": False, "score": 99},
            {"model_id": "org/b", "revision": "b", "eligible": True, "score": 2},
        ],
        submit_job=submit_job,
        poll_job=lambda job_id, config: "completed_dry_run",
    )

    assert summary["selected_candidates"] == 1
    assert submitted[0].model_id == "org/b"
    events = [
        json.loads(line.removeprefix("TRACE_EVENT "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("TRACE_EVENT ")
    ]
    event_types = [event["event_type"] for event in events]
    assert event_types.index("model_selected") < event_types.index("nebius_job_submitting")
    selected = next(event for event in events if event["event_type"] == "model_selected")
    submitting = next(
        event for event in events if event["event_type"] == "nebius_job_submitting"
    )
    assert selected["metadata"]["model_id"] == "org/b"
    assert selected["metadata"]["revision"] == "b"
    assert submitting["metadata"]["model_id"] == "org/b"


def test_prefect_schedule_descriptor() -> None:
    assert build_schedule("0 7,19 * * *", "Asia/Jerusalem") == {
        "cron": "0 7,19 * * *",
        "timezone": "Asia/Jerusalem",
    }


def test_controller_script_has_safety_flags_and_trace_events() -> None:
    text = Path("scripts/run_prefect_controller_job.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "TRACE_EVENT" in text
    assert "controller_auth_failed" in text
    assert "DRY_RUN" in text


def test_worker_script_uses_immutable_finetune_environment() -> None:
    text = Path("scripts/run_model_worker_job.sh").read_text(encoding="utf-8")

    assert "HF_MODEL_WATCH_PREBUILT_ENV" in text
    assert "uv sync --locked --no-dev --extra finetune" in text
    assert "uv run --no-sync python -m hf_model_watch.runtime.capabilities --require-gpu" in text
    assert 'uv run --no-sync hf-watch worker --payload-json "${PAYLOAD_JSON}"' in text


def test_prefect_result_sync_flow_uses_storage_config(tmp_path) -> None:
    calls = []

    def sync_results(**kwargs):
        calls.append(kwargs)
        return SyncSummary(discovered_runs=2, imported_runs=1, skipped_runs=1)

    result = run_prefect_result_sync_flow(
        "config",
        cache_dir=tmp_path / "cache",
        sync_results=sync_results,
    )

    assert calls[0]["bucket"] == "hf-model-watch"
    assert calls[0]["prefix"] == "hf-model-watch"
    assert result["imported_runs"] == 1
    assert result["skipped_runs"] == 1
