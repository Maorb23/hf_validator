from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from hf_model_watch.config import load_config, load_yaml_file
from hf_model_watch.errors import FailureCategory
from hf_model_watch.finetune.train_peft import run_peft_adaptation_suite
from hf_model_watch.hub.scan import DEFAULT_SCAN_OUTPUT, scan_models
from hf_model_watch.hub.scoring import score_jsonl_file
from hf_model_watch.orchestration.job_payloads import WorkerJobPayload, load_worker_payload
from hf_model_watch.orchestration.prefect_flow import run_prefect_controller_flow
from hf_model_watch.reporting.compare import compare_model
from hf_model_watch.reporting.leaderboard import generate_leaderboard_markdown
from hf_model_watch.runtime.smoke_tests import run_local_smoke_test
from hf_model_watch.storage.object_store import (
    ArtifactUploadError,
    object_store_config_from_mapping,
    preflight_object_store_upload,
    upload_run_artifacts,
)
from hf_model_watch.storage.schemas import (
    RunManifest,
    build_reproducibility_snapshot,
    initialize_run_artifacts,
    new_run_id,
    utc_now_iso,
)
from hf_model_watch.storage.sqlite_store import SQLiteStore
from hf_model_watch.tracing import emit_trace_event

app = typer.Typer(help="hf-model-watch command line interface.")
load_dotenv()


@app.command()
def scan(
    config: Annotated[Path, typer.Option("--config", help="Path to scan YAML config.")] = Path(
        "config/scan.yaml"
    ),
) -> None:
    scan_config = load_yaml_file(config)
    records = scan_models(scan_config, DEFAULT_SCAN_OUTPUT)
    emit_trace_event(
        "scan_cli_completed",
        "scan",
        "completed",
        "scan command completed",
        metrics={"models_written": len(records)},
        artifacts={"output_path": str(DEFAULT_SCAN_OUTPUT)},
    )


@app.command("score-candidates")
def score_candidates(
    input: Annotated[Path, typer.Option("--input", help="Candidate JSONL input path.")],
) -> None:
    config = load_config("config")
    output = Path("artifacts/scan/scored_candidates.jsonl")
    scored = score_jsonl_file(input, output, config.scan.model_dump(), config.scoring)
    emit_trace_event(
        "score_candidates_completed",
        "scoring",
        "completed",
        "candidate scoring completed",
        metrics={"candidates_written": len(scored)},
        artifacts={"input_path": str(input), "output_path": str(output)},
    )


@app.command("run-local")
def run_local(
    model_id: Annotated[str, typer.Option("--model-id", help="Hugging Face model id.")],
    config_dir: Annotated[Path, typer.Option("--config-dir", help="Config directory.")] = Path(
        "config"
    ),
    skip_finetune: Annotated[
        bool,
        typer.Option(
            "--skip-finetune",
            help="Skip PEFT adaptation and record a typed skip result.",
        ),
    ] = False,
    finetune_smoke_only: Annotated[
        bool,
        typer.Option(
            "--finetune-smoke-only",
            help="Run only PEFT smoke/capability checks after runtime/evals.",
        ),
    ] = False,
) -> None:
    config = load_config(config_dir)
    run_id = new_run_id()
    db_path = Path(str(config.storage.get("results_db", "results/hf_model_watch.sqlite")))
    store = SQLiteStore(db_path)
    store.initialize()
    run_dir = Path(str(config.storage.get("local_artifact_dir", "artifacts"))) / run_id
    try:
        result = run_local_smoke_test(run_id, model_id, config, store)
    except Exception as exc:
        store.record_failure(
            run_id,
            model_id,
            None,
            FailureCategory.LOAD_FAILED.value,
            f"{type(exc).__name__}: {exc}",
        )
        emit_trace_event(
            "run_local_failed",
            "local",
            "failed",
            "model-level local run failure recorded",
            metadata={
                "run_id": run_id,
                "model_id": model_id,
                "failure_category": FailureCategory.LOAD_FAILED.value,
                "error_type": type(exc).__name__,
            },
            artifacts={"db": str(db_path), "run_dir": str(run_dir)},
        )
        return
    if skip_finetune or finetune_smoke_only or config.finetune.get("enabled", True):
        finetune_result = run_peft_adaptation_suite(
            run_id=run_id,
            model_id=model_id,
            revision=str(result.get("revision")),
            run_dir=run_dir,
            config=config,
            store=store,
            smoke_only=finetune_smoke_only,
            skip=skip_finetune,
            metadata=(result.get("metadata") if isinstance(result.get("metadata"), dict) else None),
        )
    else:
        finetune_result = {"status": "disabled"}
    emit_trace_event(
        "run_local_completed",
        "local",
        "completed",
        "Phase 6 local runtime, eval, and adaptation flow completed",
        metadata={
            "run_id": run_id,
            "model_id": model_id,
            "revision": str(result.get("revision")),
        },
        artifacts={
            "manifest": str(run_dir / "run_manifest.json"),
            "claims_vs_observed": str(run_dir / "metadata" / "claims_vs_observed.json"),
            "runtime_result": str(run_dir / "runtime_result.json"),
            "raw_generations": str(run_dir / "raw_generations.jsonl"),
            "latency_results": str(run_dir / "latency_results.jsonl"),
            "quality_results": str(run_dir / "quality_results.jsonl"),
            "context_drift_results": str(run_dir / "context_drift_results.jsonl"),
            "finetune_config": str(run_dir / "finetune_config.json"),
            "adaptation_results": str(run_dir / "adaptation_results.json"),
            "adapter_eval": str(run_dir / "adapter_eval.json"),
            "db": str(db_path),
        },
        metrics={"finetune_status": finetune_result.get("status")},
    )


@app.command()
def worker(
    payload_json: Annotated[Path, typer.Option("--payload-json", help="Worker payload JSON path.")],
) -> None:
    payload = load_worker_payload(payload_json)
    config = load_config("config")
    _merge_mapping(config.storage, payload.config.get("storage"))
    _merge_mapping(config.finetune, payload.config.get("finetune"))
    output_root = Path(
        str(
            payload.config.get("local_artifact_dir")
            or payload.config.get("output_dir")
            or config.storage.get("local_artifact_dir")
            or os.getenv("HF_MODEL_WATCH_WORKER_OUTPUT_DIR")
            or "/workspace/output"
        )
    )
    config.storage["local_artifact_dir"] = str(output_root)
    config.storage["results_db"] = str(output_root / "results" / "hf_model_watch.sqlite")
    db_path = Path(str(config.storage["results_db"]))
    store = SQLiteStore(db_path)
    store.initialize()
    run_dir = output_root / payload.run_id
    dry_run = _worker_dry_run(payload)

    emit_trace_event(
        "worker_started",
        "worker",
        "started",
        "model worker started",
        metadata={"run_id": payload.run_id, "model_id": payload.model_id, "dry_run": dry_run},
    )
    upload_config = object_store_config_from_mapping(
        config.storage,
        artifact_prefix=payload.artifact_prefix,
        dry_run=_object_store_dry_run(payload),
    )
    preflight_object_store_upload(payload.run_id, upload_config)
    try:
        if dry_run:
            _run_worker_dry_run(payload, config.model_dump(), store, run_dir)
        else:
            result = run_local_smoke_test(payload.run_id, payload.model_id, config, store)
            if payload.skip_finetune or payload.finetune_smoke_only or config.finetune.get(
                "enabled", True
            ):
                run_peft_adaptation_suite(
                    run_id=payload.run_id,
                    model_id=payload.model_id,
                    revision=str(result.get("revision") or payload.revision or ""),
                    run_dir=run_dir,
                    config=config,
                    store=store,
                    smoke_only=payload.finetune_smoke_only,
                    skip=payload.skip_finetune,
                    metadata=(
                        result.get("metadata") if isinstance(result.get("metadata"), dict) else None
                    ),
                )
    except Exception as exc:
        _record_worker_model_failure(payload, store, run_dir, exc)

    upload_result = _upload_worker_artifacts(payload, config.storage, run_dir, store)
    emit_trace_event(
        "worker_completed",
        "worker",
        "completed",
        "model worker completed",
        metadata={"run_id": payload.run_id, "upload_dry_run": upload_result.dry_run},
        artifacts={"run_dir": str(run_dir), "artifact_uri": upload_result.uri},
    )


@app.command()
def leaderboard(
    db: Annotated[Path, typer.Option("--db", help="Results SQLite database path.")],
    out: Annotated[Path, typer.Option("--out", help="Output markdown report path.")],
) -> None:
    output = generate_leaderboard_markdown(db, out)
    emit_trace_event(
        "leaderboard_created",
        "reporting",
        "completed",
        "leaderboard markdown generated",
        artifacts={"db": str(db), "output": str(output)},
    )


@app.command()
def compare(
    model_id: Annotated[str, typer.Option("--model-id", help="Hugging Face model id.")],
    db: Annotated[Path, typer.Option("--db", help="Results SQLite database path.")],
) -> None:
    result = compare_model(db, model_id)
    emit_trace_event(
        "comparison_created",
        "reporting",
        "completed",
        "comparison report generated",
        metadata={
            "model_id": model_id,
            "comparison_status": str(result.get("comparison_status")),
        },
        artifacts={"db": str(db), "output": str(result.get("report_path"))},
    )
    typer.echo(
        f"{model_id}: {result.get('comparison_status')} "
        f"({result.get('reason')}); report={result.get('report_path')}"
    )


@app.command("prefect-flow")
def prefect_flow(
    config_dir: Annotated[Path, typer.Option("--config-dir", help="Config directory.")] = Path(
        "config"
    ),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run orchestration without submitting real Nebius jobs."),
    ] = False,
) -> None:
    summary = run_prefect_controller_flow(config_dir, dry_run=dry_run)
    emit_trace_event(
        "prefect_flow_cli_completed",
        "orchestration",
        "completed",
        "prefect-flow command completed",
        metadata={"config_dir": str(config_dir), "dry_run": dry_run},
        metrics={
            "selected_candidates": summary["selected_candidates"],
            "submitted_jobs": len(summary["submitted_jobs"]),
        },
        artifacts={"leaderboard": summary["leaderboard"]},
    )

def _run_worker_dry_run(
    payload: WorkerJobPayload,
    config_snapshot: dict[str, object],
    store: SQLiteStore,
    run_dir: Path,
) -> None:
    manifest = RunManifest(
        run_id=payload.run_id,
        model_id=payload.model_id,
        revision=payload.revision,
        status="completed",
        completed_at=utc_now_iso(),
        config_snapshot=config_snapshot,
        reproducibility=build_reproducibility_snapshot(
            payload.model_id,
            payload.revision,
            config_snapshot,
        ),
        artifacts={},
    )
    initialize_run_artifacts(run_dir.parent, manifest)
    result_path = run_dir / "worker_result.json"
    result_path.write_text(
        json.dumps(
            {
                "run_id": payload.run_id,
                "model_id": payload.model_id,
                "revision": payload.revision,
                "status": "dry_run_completed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _update_manifest_artifacts(
        run_dir,
        {
            "worker_result": str(result_path),
            "trace_events": str(run_dir / "trace_events.jsonl"),
        },
    )
    store.start_run(
        payload.run_id,
        payload.model_id,
        payload.revision,
        artifact_uri=str(run_dir),
    )
    store.complete_run(payload.run_id, status="completed")
    store.record_artifact(
        payload.run_id,
        "worker_result",
        str(result_path),
        payload.model_id,
        payload.revision,
    )


def _record_worker_model_failure(
    payload: WorkerJobPayload,
    store: SQLiteStore,
    run_dir: Path,
    exc: Exception,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    failure_path = run_dir / "worker_failure.json"
    failure_payload = {
        "run_id": payload.run_id,
        "model_id": payload.model_id,
        "revision": payload.revision,
        "failure_category": "unknown_failure",
        "failure_message": f"{type(exc).__name__}: {exc}",
    }
    failure_path.write_text(
        json.dumps(failure_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    store.record_failure(
        payload.run_id,
        payload.model_id,
        payload.revision,
        failure_payload["failure_category"],
        failure_payload["failure_message"],
    )
    store.record_artifact(
        payload.run_id,
        "worker_failure",
        str(failure_path),
        payload.model_id,
        payload.revision,
    )
    emit_trace_event(
        "worker_model_failed",
        "worker",
        "failed",
        "model-level worker failure recorded",
        metadata=failure_payload,
    )


def _upload_worker_artifacts(
    payload: WorkerJobPayload,
    storage_config: dict[str, object],
    run_dir: Path,
    store: SQLiteStore,
):
    dry_run = _object_store_dry_run(payload)
    upload_config = object_store_config_from_mapping(
        storage_config,
        artifact_prefix=payload.artifact_prefix,
        dry_run=dry_run,
    )
    try:
        result = upload_run_artifacts(run_dir, payload.run_id, upload_config)
    except ArtifactUploadError as exc:
        store.record_failure(
            payload.run_id,
            payload.model_id,
            payload.revision,
            exc.failure_category,
            str(exc),
        )
        raise typer.Exit(code=1) from exc
    if result.uri:
        store.record_artifact(
            payload.run_id,
            "uploaded_artifact_prefix",
            result.uri,
            payload.model_id,
            payload.revision,
        )
        _update_manifest_artifacts(run_dir, {"uploaded_artifact_prefix": result.uri})
    return result


def _worker_dry_run(payload: WorkerJobPayload) -> bool:
    worker = payload.config.get("worker") if isinstance(payload.config.get("worker"), dict) else {}
    raw = os.getenv("HF_MODEL_WATCH_WORKER_DRY_RUN")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(worker.get("dry_run", False))


def _object_store_dry_run(payload: WorkerJobPayload) -> bool:
    raw = os.getenv("HF_MODEL_WATCH_OBJECT_STORE_DRY_RUN")
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    storage = (
        payload.config.get("storage") if isinstance(payload.config.get("storage"), dict) else {}
    )
    s3 = storage.get("s3") if isinstance(storage.get("s3"), dict) else {}
    return bool(s3.get("dry_run", _worker_dry_run(payload)))


def _merge_mapping(target: dict[str, object], source: object) -> None:
    if isinstance(source, dict):
        target.update(source)


def _update_manifest_artifacts(run_dir: Path, artifacts: dict[str, str]) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["artifacts"] = {**data.get("artifacts", {}), **artifacts}
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
