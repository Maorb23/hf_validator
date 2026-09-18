from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from hf_model_watch.storage.sqlite_store import SQLiteStore


def build_benchmark_report(
    recommendation_path: Path,
    results_dir: Path,
    db_path: Path = Path("results/hf_model_watch.sqlite"),
) -> dict[str, Any]:
    submission_path = recommendation_path.parent / "benchmark_submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    jobs = submission.get("submitted_jobs") or []
    if not jobs:
        raise ValueError("No benchmark jobs were submitted for this recommendation")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending = []
    for job in jobs:
        path = results_dir / job["run_id"] / "benchmark_results.json"
        if not path.exists():
            pending.append(job["run_id"])
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        if (
            summary.get("model_id") != job["model_id"]
            or summary.get("framework") != job["framework"]
        ):
            raise ValueError(f"Benchmark result does not match submitted job: {job['run_id']}")
        groups[job["model_id"]].append(summary)
    comparisons = []
    for model_id, summaries in groups.items():
        comparisons.append(
            {
                "model_id": model_id,
                "frameworks": sorted(summaries, key=lambda row: row["framework"]),
                "fastest_framework": min(
                    (row for row in summaries if row.get("latency_seconds_mean") is not None),
                    key=lambda row: row["latency_seconds_mean"],
                    default={},
                ).get("framework"),
            }
        )
    report = {
        "status": "partial" if pending else "complete",
        "pending_run_ids": pending,
        "comparisons": comparisons,
    }
    output = recommendation_path.parent / "framework_comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    store = SQLiteStore(db_path)
    store.initialize()
    for job in jobs:
        summary = next(
            (row for row in groups.get(job["model_id"], []) if row["run_id"] == job["run_id"]), None
        )
        if summary is None:
            continue
        store.record_comparison_result(
            job["run_id"],
            job["model_id"],
            summary.get("revision"),
            job["model_id"],
            summary.get("revision"),
            {
                "benchmark_framework": job["framework"],
                "summary": summary,
                "comparison_artifact": str(output),
            },
        )
        store.record_artifact(
            job["run_id"],
            "framework_comparison",
            str(output),
            job["model_id"],
            summary.get("revision"),
        )
    return report
