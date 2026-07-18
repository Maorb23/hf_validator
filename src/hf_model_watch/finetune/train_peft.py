from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.finetune.datasets import count_training_tokens, load_all_adaptation_datasets
from hf_model_watch.finetune.eval_adaptation import (
    compute_adaptation_metrics,
    weighted_overall_lift,
)
from hf_model_watch.finetune.peft_config_resolver import resolve_peft_config
from hf_model_watch.finetune.qlora_training import run_qlora_training
from hf_model_watch.storage.schemas import utc_now_iso
from hf_model_watch.storage.sqlite_store import SQLiteStore
from hf_model_watch.tracing import emit_trace_event


def run_peft_adaptation_suite(
    run_id: str,
    model_id: str,
    revision: str | None,
    run_dir: str | Path,
    config: Any,
    store: SQLiteStore,
    smoke_only: bool = False,
    skip: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    peft_cfg = resolve_peft_config(
        model_size_billion=float((metadata or {}).get("estimated_params_billion") or 0.1),
        architecture=(metadata or {}).get("architecture"),
        available_vram_gb=float(config.serverless.gpu_memory_gb),
        task="multi_task_adaptation",
        context_length=int((metadata or {}).get("context_length") or 2048),
        quantization_mode="4bit",
        finetune_config=config.finetune,
    ).as_dict()
    datasets = load_all_adaptation_datasets(config.finetune.get("datasets"), split="train")
    eval_datasets = load_all_adaptation_datasets(config.finetune.get("datasets"), split="eval")
    token_counts_by_task = {
        task: count_training_tokens(examples) for task, examples in datasets.items()
    }
    method = str(peft_cfg.get("method") or peft_cfg.get("peft_method") or "lora").lower()
    emit_trace_event(
        "finetune_started",
        "finetune",
        "started",
        "QLoRA adaptation started",
        metadata={
            "run_id": run_id,
            "model_id": model_id,
            "revision": revision,
            "method": method,
            "max_steps": peft_cfg.get("max_steps"),
            "smoke_only": smoke_only,
            "skip": skip,
        },
    )
    peft_available = _module_available("peft")
    bitsandbytes_required = method == "qlora"
    bitsandbytes_available = _module_available("bitsandbytes")
    missing_dependencies = [
        name
        for name, required, available in (
            ("peft", True, peft_available),
            ("bitsandbytes", bitsandbytes_required, bitsandbytes_available),
        )
        if required and not available
    ]
    dependencies_ready = not missing_dependencies
    emit_trace_event(
        "finetune_dependencies",
        "finetune",
        "completed" if dependencies_ready or skip else "failed",
        (
            "fine-tune dependencies are available"
            if dependencies_ready
            else f"missing required fine-tune dependencies: {', '.join(missing_dependencies)}"
        ),
        metadata={
            "run_id": run_id,
            "model_id": model_id,
            "method": method,
            "peft_available": peft_available,
            "bitsandbytes_available": bitsandbytes_available,
            "bitsandbytes_required": bitsandbytes_required,
            "missing_dependencies": missing_dependencies,
        },
    )
    training_details = None
    if skip:
        status = "skipped"
        failure_category = None
    elif not dependencies_ready:
        status = "smoke_unavailable"
        failure_category = FailureCategory.FINETUNE_SMOKE_FAILED.value
    else:
        task_runs: dict[str, dict[str, Any]] = {}
        for task_type, task_examples in datasets.items():
            if not task_examples:
                continue
            emit_trace_event(
                "finetune_task_started",
                "finetune",
                "started",
                f"starting independent QLoRA run for {task_type}",
                metadata={
                    "run_id": run_id,
                    "model_id": model_id,
                    "task_type": task_type,
                    "training_examples": len(task_examples),
                },
            )
            try:
                details = run_qlora_training(
                    run_id=run_id,
                    model_id=model_id,
                    revision=revision,
                    run_dir=root / "finetune" / task_type,
                    train_datasets={task_type: task_examples},
                    eval_datasets={task_type: eval_datasets.get(task_type, [])},
                    peft_config=peft_cfg,
                    smoke_only=smoke_only,
                    task_type=task_type,
                )
            except Exception as exc:
                details = {
                    "success": False,
                    "message": f"QLoRA training failed: {type(exc).__name__}: {exc}",
                }
            task_runs[task_type] = details
            emit_trace_event(
                "finetune_task_finished",
                "finetune",
                "completed" if details["success"] else "failed",
                details["message"],
                metadata={"run_id": run_id, "model_id": model_id, "task_type": task_type},
            )
        successful_runs = {task: item for task, item in task_runs.items() if item["success"]}
        training_details = {
            "success": bool(task_runs) and len(successful_runs) == len(task_runs),
            "message": (
                f"completed {len(successful_runs)}/{len(task_runs)} independent QLoRA runs"
            ),
            "task_runs": task_runs,
            "base_scores": {
                task: float(item.get("base_scores", {}).get(task, 0.0))
                for task, item in successful_runs.items()
            },
            "adapted_scores": {
                task: float(item.get("adapted_scores", {}).get(task, 0.0))
                for task, item in successful_runs.items()
            },
        }
        if training_details["success"]:
            status = "smoke_completed" if smoke_only else "training_completed"
            failure_category = None
        else:
            status = "smoke_failed" if smoke_only else "training_failed"
            failure_category = (
                FailureCategory.FINETUNE_OOM.value
                if "out of memory" in training_details["message"].lower()
                else FailureCategory.FINETUNE_FAILED.value
            )
    smoke = {
        "status": status,
        "peft_available": peft_available,
        "smoke_only": smoke_only,
        "failure_category": failure_category,
        "message": (
            "fine-tune skipped by CLI flag"
            if skip
            else f"missing required dependencies: {', '.join(missing_dependencies)}"
            if not dependencies_ready
            else training_details["message"]
        ),
        "details": training_details,
        "steps": {
            "load_model": bool(training_details and training_details["success"]),
            "attach_adapter": bool(training_details and training_details["success"]),
            "forward_backward_optimizer_step": bool(
                training_details and training_details["success"]
            ),
            "save_adapter": bool(
                training_details
                and any(
                    item.get("adapter_dir")
                    for item in training_details.get("task_runs", {}).values()
                )
            ),
            "evaluate_adapter": bool(training_details and training_details.get("adapted_scores")),
        },
    }
    task_results = []
    training_completed = status == "training_completed"
    base_scores = (training_details or {}).get("base_scores") or {}
    adapted_scores = (training_details or {}).get("adapted_scores") or {}
    for task in sorted(set(base_scores) & set(adapted_scores)):
        token_counts = token_counts_by_task[task]
        task_training = training_details["task_runs"][task]
        task_results.append(
            compute_adaptation_metrics(
                model_id=model_id,
                run_id=run_id,
                task_type=task,
                peft_config=peft_cfg,
                base_quality_score=float(base_scores[task]),
                adapted_quality_score=adapted_scores[task],
                token_counts=token_counts,
                trainable_params=int(task_training["trainable_params"]),
                total_params=int(task_training["total_params"]),
            )
        )
    overall = weighted_overall_lift(task_results)
    result = {
        "run_id": run_id,
        "model_id": model_id,
        "revision": revision,
        "status": status,
        "training_completed": training_completed,
        "created_at": utc_now_iso(),
        "peft_config": peft_cfg,
        "smoke": smoke,
        "task_results": task_results,
        "overall": overall,
    }
    paths = _write_finetune_artifacts(root, peft_cfg, result, smoke)
    _update_manifest(root, paths)
    for artifact_type, artifact_uri in paths.items():
        store.record_artifact(run_id, artifact_type, artifact_uri, model_id, revision)
    store.record_finetune_run(run_id, model_id, revision, status, result)
    store.record_adaptation_result(run_id, model_id, revision, result)
    store.complete_run(
        run_id,
        status=(
            "completed"
            if status in {"training_completed", "smoke_completed", "skipped"}
            else "failed"
        ),
        adapted_score=overall["adapted_quality_score"],
        adaptation_lift=overall["adaptation_lift"],
    )
    emit_trace_event(
        "finetune_finished",
        "finetune",
        (
            "skipped"
            if status == "skipped"
            else "completed"
            if status in {"smoke_completed", "training_completed"}
            else "failed"
        ),
        (
            "QLoRA adaptation completed"
            if status in {"smoke_completed", "training_completed"}
            else smoke["message"]
        ),
        metadata={
            "run_id": run_id,
            "model_id": model_id,
            "method": method,
            "finetune_status": status,
            "training_completed": training_completed,
        },
        metrics={"adaptation_lift": overall["adaptation_lift"]},
    )
    return result


def _write_finetune_artifacts(
    run_dir: Path,
    peft_config: dict[str, Any],
    adaptation_result: dict[str, Any],
    adapter_eval: dict[str, Any],
) -> dict[str, str]:
    paths = {
        "finetune_config": run_dir / "finetune_config.json",
        "adaptation_results": run_dir / "adaptation_results.json",
        "adapter_eval": run_dir / "adapter_eval.json",
    }
    details = (adaptation_result.get("smoke") or {}).get("details") or {}
    for task_type, task_details in details.get("task_runs", {}).items():
        if task_details.get("adapter_dir"):
            paths[f"adapter_{task_type}"] = Path(task_details["adapter_dir"])
    paths["finetune_config"].write_text(
        json.dumps(peft_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["adaptation_results"].write_text(
        json.dumps(adaptation_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["adapter_eval"].write_text(
        json.dumps(adapter_eval, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def _update_manifest(run_dir: Path, paths: dict[str, str]) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["artifacts"] = {**data.get("artifacts", {}), **paths}
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
