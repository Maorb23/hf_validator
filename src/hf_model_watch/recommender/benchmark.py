from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hf_model_watch.config import load_config
from hf_model_watch.orchestration.job_payloads import WorkerJobPayload
from hf_model_watch.orchestration.nebius_jobs import NebiusJobConfig, submit_nebius_job
from hf_model_watch.recommender.core import load_settings
from hf_model_watch.storage.schemas import new_run_id
from hf_model_watch.storage.sqlite_store import SQLiteStore


def benchmark_recommendation(
    path: Path,
    *,
    confirm: bool = False,
    dry_run: bool = False,
    config_dir: Path = Path("config"),
    submit: Any = submit_nebius_job,
) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("status") != "recommended" or not artifact.get("recommendation"):
        raise ValueError("Artifact has no confirmed recommendation")
    plan = artifact.get("benchmark_plan") or []
    if not plan:
        raise ValueError("Recommendation has no benchmark plan")
    config = load_config(config_dir)
    recommender_settings = load_settings(config_dir / "recommender.yaml")
    framework_mode = artifact.get("framework_mode")
    if framework_mode not in {"fast", "sweep"}:
        raise ValueError("Recommendation has no valid framework mode")
    choices = plan if framework_mode == "sweep" else plan[:1]
    preflight: list[dict[str, Any]] = []
    problems: list[str] = []
    candidate_index = {item["model_id"]: item for item in artifact.get("evaluated_candidates", [])}
    for item in choices:
        candidate = candidate_index.get(item["model_id"], {})
        frameworks = item.get("frameworks") or []
        if not frameworks:
            problems.append(f"{item['model_id']}: no framework selected")
        for framework in frameworks:
            reasons = []
            if not candidate.get("revision") or item.get("revision") != candidate.get("revision"):
                reasons.append("pinned revision is missing or inconsistent")
            if item.get("task_fixture") not in {
                "pr_review",
                "social_scoring",
                "long_context_analysis",
            }:
                reasons.append("benchmark task fixture is missing or unsupported")
            image = (
                config.serverless.image
                if framework == "transformers"
                else recommender_settings.get("framework_images", {}).get(framework)
            )
            if not image:
                reasons.append("framework worker image is not configured")
            if framework not in candidate.get("compatible_frameworks", []):
                reasons.append("framework compatibility is not evidenced")
            if candidate.get("blocking_reasons") or not candidate.get("license"):
                reasons.append("candidate constraints or license are unresolved")
            memory = candidate.get("estimated_bf16_memory_gb")
            usable_gpu_count = config.serverless.gpu_count if framework != "transformers" else 1
            available = (
                config.serverless.gpu_memory_gb
                * usable_gpu_count
                * config.serverless.memory_safety_margin
            )
            if memory is None or float(memory) > available:
                reasons.append("configured GPU memory is insufficient or unverified")
            preflight.append(
                {
                    "model_id": item["model_id"],
                    "framework": framework,
                    "ready": not reasons,
                    "reasons": reasons,
                }
            )
            problems.extend(reasons)
    outcome: dict[str, Any] = {
        "status": "preflight_failed" if problems else "awaiting_confirmation",
        "confirmed": confirm,
        "dry_run": dry_run,
        "preflight": preflight,
        "submitted_jobs": [],
    }
    if problems or not confirm:
        return outcome
    store = SQLiteStore(
        Path(str(config.storage.get("results_db", "results/hf_model_watch.sqlite")))
    )
    store.initialize()
    for item in preflight:
        run_id = new_run_id()
        payload = WorkerJobPayload(
            run_id=run_id,
            model_id=item["model_id"],
            revision=candidate_index[item["model_id"]].get("revision"),
            skip_finetune=True,
            config={
                "benchmark_recommendation": {
                    "framework": item["framework"],
                    "gpu_count": config.serverless.gpu_count,
                    "task_fixture": next(
                        (
                            entry.get("task_fixture")
                            for entry in choices
                            if entry["model_id"] == item["model_id"]
                        ),
                        None,
                    ),
                    "source_artifact": str(path),
                }
            },
        )
        job_config = NebiusJobConfig(
            project_id=config.serverless.project_id,
            image=(
                config.serverless.image
                if item["framework"] == "transformers"
                else recommender_settings["framework_images"][item["framework"]]
            ),
            preset=config.serverless.preset,
            gpu_type=config.serverless.gpu_type,
            gpu_count=config.serverless.gpu_count,
            timeout_minutes=config.serverless.timeout_minutes,
            dry_run=dry_run,
        )
        result = submit(payload, job_config)
        store.start_run(run_id, payload.model_id, payload.revision, artifact_uri=str(path))
        store.record_artifact(
            run_id, "recommendation", str(path), payload.model_id, payload.revision
        )
        outcome["submitted_jobs"].append(
            {
                "run_id": run_id,
                "job_id": result.job_id,
                "model_id": payload.model_id,
                "framework": item["framework"],
                "status": result.status,
            }
        )
    outcome["status"] = "submitted_dry_run" if dry_run else "submitted"
    output = path.parent / "benchmark_submission.json"
    output.write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outcome
