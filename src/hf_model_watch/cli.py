from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from dotenv import load_dotenv

from hf_model_watch.tracing import emit_trace_event

if TYPE_CHECKING:
    from hf_model_watch.orchestration.job_payloads import WorkerJobPayload
    from hf_model_watch.storage.sqlite_store import SQLiteStore

app = typer.Typer(help="hf-model-watch command line interface.")
load_dotenv()


class FrameworkMode(StrEnum):
    fast = "fast"
    sweep = "sweep"


@app.command()
def recommend(
    request: Annotated[str | None, typer.Argument(help="Natural-language task request.")] = None,
    gpu_memory_gb: Annotated[float | None, typer.Option("--gpu-memory-gb")] = None,
    gpu_count: Annotated[int, typer.Option("--gpu-count")] = 1,
    budget_usd_per_hour: Annotated[float | None, typer.Option("--budget-usd-per-hour")] = None,
    gpu_hourly_cost_usd: Annotated[float | None, typer.Option("--gpu-hourly-cost-usd")] = None,
    license: Annotated[str | None, typer.Option("--license")] = None,
    framework_mode: Annotated[FrameworkMode | None, typer.Option("--framework-mode")] = None,
    model: Annotated[str | None, typer.Option("--nemotron-model")] = None,
    model_id: Annotated[
        str | None, typer.Option("--model-id", help="Explicit Hugging Face model ID.")
    ] = None,
    candidate: Annotated[list[str] | None, typer.Option("--candidate")] = None,
    json_mode: Annotated[
        bool, typer.Option("--json", help="Machine-readable, non-interactive output.")
    ] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("config/recommender.yaml"),
) -> None:
    from hf_model_watch.recommender.core import (
        RequestSpec,
        load_settings,
    )
    from hf_model_watch.recommender.core import (
        recommend as run_recommend,
    )
    from hf_model_watch.recommender.discovery import discover_models, discovery_inputs
    from hf_model_watch.storage.schemas import new_run_id

    if request is None and not json_mode:
        request = typer.prompt("Describe what model you want")
    if model_id is None and request:
        values, missing = discovery_inputs(
            request,
            {
                "gpu_memory_gb": gpu_memory_gb,
                "budget_usd_per_hour": budget_usd_per_hour,
                "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
                "license": license,
                "framework_mode": framework_mode.value if framework_mode else None,
            },
        )
        if json_mode and missing:
            typer.echo(
                json.dumps({"status": "needs_input", "needs_input": missing}, sort_keys=True)
            )
            return
        defaults = {"license": "any", "framework_mode": "sweep"}
        for key in missing:
            values[key] = typer.prompt(key.replace("_", " "), default=defaults.get(key))
    else:
        values = {
            "request": request,
            "gpu_memory_gb": gpu_memory_gb,
            "budget_usd_per_hour": budget_usd_per_hour,
            "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
            "license": license,
            "framework_mode": framework_mode.value if framework_mode else None,
        }
        missing = [key for key, value in values.items() if value is None]
        if json_mode and missing:
            typer.echo(
                json.dumps({"status": "needs_input", "needs_input": missing}, sort_keys=True)
            )
            return
        for key in missing:
            values[key] = typer.prompt(key.replace("_", " "))
    try:
        spec = RequestSpec(
            **values,
            gpu_count=gpu_count,
            model=model,
            candidates=([model_id] if model_id else candidate or []),
        )
        settings = load_settings(config)
        destination = output_dir or Path("artifacts/recommendations") / new_run_id()
        if model_id:
            result = run_recommend(spec, settings, destination)
        elif json_mode:
            # The scanner emits TRACE_EVENT lines on stdout; reserve stdout for JSON.
            with redirect_stdout(sys.stderr):
                result = discover_models(
                    spec,
                    settings,
                    destination,
                    progress=lambda message: typer.echo(message, err=True),
                )
        else:
            result = discover_models(
                spec,
                settings,
                destination,
                progress=lambda message: typer.echo(message, err=True),
            )
    except Exception as exc:
        if json_mode:
            typer.echo(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
            raise typer.Exit(code=1) from None
        raise typer.BadParameter(str(exc)) from exc
    if model_id is None and not json_mode:
        gemma = result.get("scan_diagnostics", {}).get("gemma", {})
        if gemma and not gemma.get("eligible"):
            blockers = sorted(gemma.get("blockers", {}).items(), key=lambda item: -item[1])
            reasons = ", ".join(f"{reason} ({count})" for reason, count in blockers[:3])
            typer.echo(
                f"Gemma scan: {gemma.get('found', 0)} found, "
                f"{gemma.get('size_matched', 0)} near target size, none eligible. "
                f"Main blockers: {reasons or 'no size match'}"
            )
    if model_id is None and not json_mode and result["recommendations"]:
        typer.echo("Recommended models")
        for number, item in enumerate(result["recommendations"], 1):
            label = " (strongest recommendation)" if number == 1 else ""
            typer.echo(f"{number}. {item['model_id']}{label}")
            if item.get("reasoning"):
                typer.echo(f"   Decision reasoning: {item['reasoning']}")
            typer.echo(f"   HF score: {item['hf_score']:.3f}")
            for reason in item["why_it_fits"]:
                typer.echo(f"   Why: {reason}")
            for field, status in item.get("task_fit", {}).items():
                typer.echo(f"   {field.replace('_', ' ')}: {status}")
            for limitation in item["limitations"]:
                typer.echo(f"   Limitation: {limitation}")
        choice = typer.prompt("Select model", type=int, default=1)
        if choice not in range(1, len(result["recommendations"]) + 1):
            raise typer.BadParameter("Select a listed model number")
        result["selected_model_id"] = result["recommendations"][choice - 1]["model_id"]
        (destination / "discovery.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
        )
        typer.echo(f"Selected model: {result['selected_model_id']}")
        typer.echo("[ Optimize Inference + Benchmark ] (placeholder)")
    if json_mode:
        typer.echo(
            json.dumps(
                {
                    **result,
                    "artifact_path": str(
                        destination / ("recommendation.json" if model_id else "discovery.json")
                    ),
                },
                sort_keys=True,
            )
        )
    elif model_id is not None:
        chosen = result["recommendation"]
        typer.echo(f"{result['status']}: {chosen['model_id'] if chosen else 'none'}")
        typer.echo(f"Saved: {destination / 'recommendation.json'}")


@app.command("benchmark-recommendation")
def benchmark_recommendation_cli(
    artifact: Annotated[Path, typer.Argument(help="Saved recommendation.json artifact.")],
    confirm: Annotated[
        bool, typer.Option("--confirm", help="Authorize Nebius submission.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    config_dir: Annotated[Path, typer.Option("--config-dir")] = Path("config"),
) -> None:
    from hf_model_watch.recommender.benchmark import benchmark_recommendation

    try:
        result = benchmark_recommendation(
            artifact, confirm=confirm, dry_run=dry_run, config_dir=config_dir
        )
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, sort_keys=True))


@app.command("benchmark-report")
def benchmark_report_cli(
    artifact: Annotated[Path, typer.Argument(help="Saved recommendation.json artifact.")],
    results_dir: Annotated[Path, typer.Option("--results-dir", help="Synchronized run folders.")],
    db: Annotated[Path, typer.Option("--db")] = Path("results/hf_model_watch.sqlite"),
) -> None:
    from hf_model_watch.recommender.report import build_benchmark_report

    try:
        result = build_benchmark_report(artifact, results_dir, db)
    except (ValueError, FileNotFoundError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, sort_keys=True))


@app.command()
def scan(
    config: Annotated[Path, typer.Option("--config", help="Path to scan YAML config.")] = Path(
        "config/scan.yaml"
    ),
) -> None:
    from hf_model_watch.config import load_yaml_file
    from hf_model_watch.hub.scan import DEFAULT_SCAN_OUTPUT, scan_models

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


@app.command()
def lookup(
    model_id: Annotated[
        str | None, typer.Option("--model-id", help="Exact Hugging Face model ID.")
    ] = None,
    family: Annotated[
        str | None, typer.Option("--family", help="Case-insensitive model family search.")
    ] = None,
    query: Annotated[str | None, typer.Option("--query", help="Model name or tag search.")] = None,
    fuzzy: Annotated[
        bool, typer.Option("--fuzzy", help="Allow approximate local matching.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", help="Maximum results (1-100).")] = 20,
    output: Annotated[Path, typer.Option("--output", help="JSONL result path.")] = Path(
        "artifacts/scan/lookup_models.jsonl"
    ),
) -> None:
    emit_trace_event(
        "lookup_started",
        "lookup",
        "started",
        "Hugging Face model lookup started",
        metadata={
            "selector": "model_id" if model_id else "family" if family else "query",
            "fuzzy": fuzzy,
            "limit": limit,
        },
        artifacts={"output_path": str(output)},
    )
    from hf_model_watch.hub.lookup import lookup_models

    try:
        records = lookup_models(
            model_id=model_id,
            family=family,
            query=query,
            fuzzy=fuzzy,
            limit=limit,
            output_path=output,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    emit_trace_event(
        "lookup_completed",
        "lookup",
        "completed",
        "Hugging Face model lookup completed",
        metrics={"models_written": len(records)},
        artifacts={"output_path": str(output)},
    )


@app.command("score-candidates")
def score_candidates(
    input: Annotated[Path, typer.Option("--input", help="Candidate JSONL input path.")],
) -> None:
    from hf_model_watch.config import load_config
    from hf_model_watch.hub.scoring import score_jsonl_file

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
    from hf_model_watch.config import load_config
    from hf_model_watch.errors import FailureCategory
    from hf_model_watch.finetune.train_peft import run_peft_adaptation_suite
    from hf_model_watch.runtime.smoke_tests import run_local_smoke_test
    from hf_model_watch.storage.schemas import new_run_id
    from hf_model_watch.storage.sqlite_store import SQLiteStore

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
    from hf_model_watch.config import load_config
    from hf_model_watch.finetune.train_peft import run_peft_adaptation_suite
    from hf_model_watch.orchestration.job_payloads import load_worker_payload
    from hf_model_watch.runtime.smoke_tests import run_local_smoke_test
    from hf_model_watch.storage.object_store import (
        object_store_config_from_mapping,
        preflight_object_store_upload,
    )
    from hf_model_watch.storage.sqlite_store import SQLiteStore

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
        elif payload.config.get("benchmark_recommendation"):
            from hf_model_watch.recommender.benchmark_run import run_fixture_benchmark

            run_fixture_benchmark(payload, run_dir, store)
        else:
            result = run_local_smoke_test(payload.run_id, payload.model_id, config, store)
            if (
                payload.skip_finetune
                or payload.finetune_smoke_only
                or config.finetune.get("enabled", True)
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
    from hf_model_watch.reporting.leaderboard import generate_leaderboard_markdown

    output = generate_leaderboard_markdown(db, out)
    emit_trace_event(
        "leaderboard_created",
        "reporting",
        "completed",
        "leaderboard markdown generated",
        artifacts={"db": str(db), "output": str(output)},
    )


@app.command("ingest-results")
def ingest_results(
    input: Annotated[
        Path,
        typer.Option("--input", help="Directory containing downloaded run directories."),
    ] = Path("outputs"),
    db: Annotated[
        Path,
        typer.Option("--db", help="Application results SQLite database."),
    ] = Path("results/hf_model_watch.sqlite"),
    out: Annotated[
        Path,
        typer.Option("--out", help="Leaderboard Markdown output path."),
    ] = Path("reports/leaderboard.md"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Reimport runs even when their fingerprint is unchanged."),
    ] = False,
) -> None:
    from hf_model_watch.reporting.result_sync import ingest_local_results

    summary = ingest_local_results(input, db, out, force=force)
    typer.echo(json.dumps(summary.as_dict(), sort_keys=True))


@app.command("sync-nebius-results")
def sync_nebius_results(
    config_dir: Annotated[
        Path,
        typer.Option("--config-dir", help="Configuration directory."),
    ] = Path("config"),
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Local cache for downloaded reporting artifacts."),
    ] = Path("outputs/synced-nebius"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Redownload and reimport unchanged runs."),
    ] = False,
) -> None:
    from hf_model_watch.config import load_config
    from hf_model_watch.reporting.result_sync import sync_s3_results

    config = load_config(config_dir)
    s3 = config.storage.get("s3") or {}
    summary = sync_s3_results(
        bucket=str(s3.get("bucket") or ""),
        prefix=str(s3.get("prefix") or "hf-model-watch"),
        endpoint_url=s3.get("endpoint_url"),
        db_path=Path(str(config.storage.get("results_db", "results/hf_model_watch.sqlite"))),
        cache_dir=cache_dir,
        leaderboard_path=(
            Path(str(config.storage.get("reports_dir", "reports"))) / "leaderboard.md"
        ),
        force=force,
    )
    typer.echo(json.dumps(summary.as_dict(), sort_keys=True))


@app.command()
def compare(
    model_id: Annotated[str, typer.Option("--model-id", help="Hugging Face model id.")],
    db: Annotated[Path, typer.Option("--db", help="Results SQLite database path.")],
) -> None:
    from hf_model_watch.reporting.compare import compare_model

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
    from hf_model_watch.orchestration.prefect_flow import run_prefect_controller_flow

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


@app.command("prefect-sync-results")
def prefect_sync_results(
    config_dir: Annotated[
        Path,
        typer.Option("--config-dir", help="Configuration directory."),
    ] = Path("config"),
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Local cache for synchronized artifacts."),
    ] = Path("outputs/synced-nebius"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Reimport unchanged S3 runs."),
    ] = False,
) -> None:
    from hf_model_watch.orchestration.result_sync_flow import run_prefect_result_sync_flow

    summary = run_prefect_result_sync_flow(config_dir, cache_dir, force)
    typer.echo(json.dumps(summary, sort_keys=True))


def _run_worker_dry_run(
    payload: WorkerJobPayload,
    config_snapshot: dict[str, object],
    store: SQLiteStore,
    run_dir: Path,
) -> None:
    from hf_model_watch.storage.schemas import (
        RunManifest,
        build_reproducibility_snapshot,
        initialize_run_artifacts,
        utc_now_iso,
    )

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
    from hf_model_watch.storage.object_store import (
        ArtifactUploadError,
        object_store_config_from_mapping,
        upload_run_artifacts,
    )

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
