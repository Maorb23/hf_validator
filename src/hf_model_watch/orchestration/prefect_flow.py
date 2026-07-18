from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hf_model_watch.config import AppConfig, load_config
from hf_model_watch.hub.scan import DEFAULT_SCAN_OUTPUT, scan_models
from hf_model_watch.hub.scoring import score_jsonl_file
from hf_model_watch.orchestration.nebius_jobs import (
    NebiusJobConfig,
    NebiusJobResult,
    build_payload_for_candidate,
    local_manifest_path,
    manifest_uri_for_job,
    poll_nebius_job,
    submit_nebius_job,
)
from hf_model_watch.reporting.leaderboard import generate_leaderboard_markdown
from hf_model_watch.storage.schemas import new_run_id
from hf_model_watch.storage.sqlite_store import SQLiteStore
from hf_model_watch.tracing import emit_trace_event


def run_prefect_controller_flow(
    config_dir: str | Path = "config",
    dry_run: bool = False,
    candidates: list[dict[str, Any]] | None = None,
    submit_job: Any = submit_nebius_job,
    poll_job: Any = poll_nebius_job,
) -> dict[str, Any]:
    config = load_config(config_dir)
    store = SQLiteStore(_db_path(config))
    store.initialize()
    emit_trace_event(
        "prefect_flow_started",
        "orchestration",
        "started",
        "Prefect controller flow started",
        metadata={"dry_run": dry_run},
    )
    selected = _select_candidates(config, dry_run, candidates)
    job_config = _job_config_from_app(config, dry_run)
    submitted: list[NebiusJobResult] = []
    for selection_index, candidate in enumerate(selected, start=1):
        run_id = candidate.get("run_id") or new_run_id()
        selection_metadata = {
            "run_id": run_id,
            "model_id": str(candidate["model_id"]),
            "revision": candidate.get("revision"),
            "score": candidate.get("score"),
            "selection_index": selection_index,
            "selected_candidates": len(selected),
        }
        emit_trace_event(
            "model_selected",
            "selection",
            "completed",
            "model selected for validation",
            metadata=selection_metadata,
        )
        payload = build_payload_for_candidate(
            candidate,
            run_id=run_id,
            config=_worker_config(config, dry_run),
            artifact_prefix=_artifact_prefix(config),
        )
        emit_trace_event(
            "nebius_job_submitting",
            "orchestration",
            "started",
            "submitting selected model to Nebius",
            metadata=selection_metadata,
        )
        result = submit_job(payload, job_config)
        submitted.append(result)
        status = poll_job(result.job_id, job_config)
        emit_trace_event(
            "nebius_job_polled",
            "orchestration",
            "completed",
            "Nebius job status polled",
            metadata={"run_id": run_id, "job_id": result.job_id, "status": status},
        )
        _ingest_manifest_reference(config, store, candidate, run_id, result, status)
    leaderboard_path = Path(str(config.storage.get("reports_dir", "reports"))) / "leaderboard.md"
    generate_leaderboard_markdown(_db_path(config), leaderboard_path)
    summary = {
        "dry_run": dry_run,
        "selected_candidates": len(selected),
        "submitted_jobs": [result.__dict__ for result in submitted],
        "leaderboard": str(leaderboard_path),
    }
    emit_trace_event(
        "prefect_flow_completed",
        "orchestration",
        "completed",
        "Prefect controller flow completed",
        metrics={"selected_candidates": len(selected), "submitted_jobs": len(submitted)},
        artifacts={"leaderboard": str(leaderboard_path)},
    )
    return summary


def build_schedule(cron: str = "0 7 * * *", timezone: str = "UTC") -> dict[str, str]:
    return {"cron": cron, "timezone": timezone}


try:
    from prefect import flow
except ImportError:  # Prefect is optional for local tests.
    flow = None


if flow is not None:
    prefect_controller_flow = flow(
        name="hf-model-watch-controller",
        log_prints=True,
    )(run_prefect_controller_flow)
else:
    prefect_controller_flow = run_prefect_controller_flow


def _select_candidates(
    config: AppConfig,
    dry_run: bool,
    candidates: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if candidates is not None:
        records = candidates
    elif dry_run:
        records = [_mock_candidate()]
    else:
        scan_records = scan_models(config.scan.model_dump(), DEFAULT_SCAN_OUTPUT)
        scored_path = Path("artifacts/scan/scored_candidates.jsonl")
        records = score_jsonl_file(
            DEFAULT_SCAN_OUTPUT,
            scored_path,
            config.scan.model_dump(),
            config.scoring,
        )
        if not records and scan_records:
            records = scan_records
    eligible = [record for record in records if record.get("eligible", True)]
    eligible.sort(key=lambda record: float(record.get("score") or 0.0), reverse=True)
    return eligible[: int(config.scan.max_candidates_to_run)]


def _mock_candidate() -> dict[str, Any]:
    return {
        "model_id": "sshleifer/tiny-gpt2",
        "revision": "dry-run",
        "eligible": True,
        "score": 1.0,
        "lineage": {"family": "GPT2", "size_band": "tiny"},
    }


def _job_config_from_app(config: AppConfig, dry_run: bool) -> NebiusJobConfig:
    serverless = config.serverless
    raw = config.serverless.model_dump()
    return NebiusJobConfig(
        project_id=raw.get("project_id"),
        profile_name=str(raw.get("profile_name") or "hf-model-watch-controller"),
        image=raw.get("image"),
        preset=raw.get("preset"),
        gpu_type=serverless.gpu_type,
        gpu_count=serverless.gpu_count,
        timeout_minutes=serverless.timeout_minutes,
        dry_run=dry_run,
        artifact_prefix=_artifact_prefix(config),
    )


def _worker_config(config: AppConfig, dry_run: bool) -> dict[str, Any]:
    payload = config.model_dump()
    payload["worker"] = {"dry_run": dry_run}
    payload["storage"] = {
        **payload.get("storage", {}),
        "s3": {**payload.get("storage", {}).get("s3", {}), "dry_run": dry_run},
    }
    return payload


def _artifact_prefix(config: AppConfig) -> str:
    storage = config.storage
    s3 = storage.get("s3") if isinstance(storage.get("s3"), dict) else {}
    return str(s3.get("prefix") or storage.get("artifact_prefix") or "hf-model-watch")


def _db_path(config: AppConfig) -> Path:
    return Path(str(config.storage.get("results_db", "results/hf_model_watch.sqlite")))


def _ingest_manifest_reference(
    config: AppConfig,
    store: SQLiteStore,
    candidate: dict[str, Any],
    run_id: str,
    result: NebiusJobResult,
    status: str,
) -> None:
    model_id = str(candidate["model_id"])
    revision = candidate.get("revision")
    storage = config.storage
    s3 = storage.get("s3") if isinstance(storage.get("s3"), dict) else {}
    bucket = str(s3.get("bucket") or "")
    manifest_uri = (
        manifest_uri_for_job(bucket, _artifact_prefix(config), run_id)
        if bucket
        else str(local_manifest_path(str(storage.get("local_artifact_dir", "artifacts")), run_id))
    )
    if store.fetch_run(run_id) is None:
        store.start_run(run_id, model_id, revision, artifact_uri=manifest_uri)
    store.complete_run(run_id, status="completed" if "completed" in status else "started")
    store.record_artifact(run_id, "worker_manifest", manifest_uri, model_id, revision)
    store.record_artifact(
        run_id,
        "nebius_job_result",
        json.dumps(result.payload),
        model_id,
        revision,
    )
