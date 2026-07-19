from __future__ import annotations

from pathlib import Path
from typing import Any

from hf_model_watch.config import load_config
from hf_model_watch.reporting.result_sync import sync_s3_results
from hf_model_watch.tracing import emit_trace_event


def run_prefect_result_sync_flow(
    config_dir: str | Path = "config",
    cache_dir: str | Path = "outputs/synced-nebius",
    force: bool = False,
    sync_results: Any | None = None,
) -> dict[str, Any]:
    config = load_config(config_dir)
    s3 = config.storage.get("s3") or {}
    db_path = Path(str(config.storage.get("results_db", "results/hf_model_watch.sqlite")))
    leaderboard_path = (
        Path(str(config.storage.get("reports_dir", "reports"))) / "leaderboard.md"
    )
    emit_trace_event(
        "prefect_result_sync_started",
        "result_sync",
        "started",
        "Prefect Nebius result synchronization started",
        metadata={"bucket": s3.get("bucket"), "prefix": s3.get("prefix")},
    )
    sync_function = sync_results or sync_s3_results
    summary = sync_function(
        bucket=str(s3.get("bucket") or ""),
        prefix=str(s3.get("prefix") or "hf-model-watch"),
        endpoint_url=s3.get("endpoint_url"),
        db_path=db_path,
        cache_dir=cache_dir,
        leaderboard_path=leaderboard_path,
        force=force,
    )
    result = summary.as_dict() if hasattr(summary, "as_dict") else dict(summary)
    result.update(
        {
            "db": str(db_path),
            "leaderboard": str(leaderboard_path),
            "cache_dir": str(cache_dir),
        }
    )
    emit_trace_event(
        "prefect_result_sync_completed",
        "result_sync",
        "completed" if not result.get("failed_runs") else "failed",
        "Prefect Nebius result synchronization completed",
        metrics={
            key: result[key]
            for key in (
                "discovered_runs",
                "imported_runs",
                "skipped_runs",
                "downloaded_files",
                "failed_runs",
            )
            if key in result
        },
        artifacts={"db": str(db_path), "leaderboard": str(leaderboard_path)},
    )
    return result


try:
    from prefect import flow
except ImportError:  # Prefect is optional for local tests.
    flow = None


if flow is not None:
    prefect_result_sync_flow = flow(
        name="hf-model-watch-result-sync",
        log_prints=True,
    )(run_prefect_result_sync_flow)
else:
    prefect_result_sync_flow = run_prefect_result_sync_flow
