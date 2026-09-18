from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.finetune.attempt_subprocess import run_training_attempt_subprocess
from hf_model_watch.finetune.datasets import count_training_tokens, load_all_adaptation_datasets
from hf_model_watch.finetune.eval_adaptation import (
    compute_adaptation_metrics,
    weighted_overall_lift,
)
from hf_model_watch.finetune.model_profile import profile_model_config
from hf_model_watch.finetune.policy import (
    DatasetProfile,
    HardwareProfile,
    ResolvedFinetunePolicy,
    build_dataset_profile,
    resolve_finetune_policy,
)
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
    finetune_config = dict(config.finetune)
    model_metadata = metadata or {}
    hardware = HardwareProfile(
        gpu_name=str(getattr(config.serverless, "gpu_type", "configured GPU")),
        total_vram_bytes=int(float(config.serverless.gpu_memory_gb) * 2**30),
        bf16_supported=bool(finetune_config.get("bf16", True)),
        gpu_count=1,
        flash_attention_supported=_module_available("flash_attn"),
    )
    requested_method = str(
        finetune_config.get("method") or finetune_config.get("default_method") or "auto"
    ).lower()
    peft_available = _module_available("peft")
    trl_available = _module_available("trl")
    bitsandbytes_required = requested_method in {"auto", "qlora"}
    bitsandbytes_available = _module_available("bitsandbytes")
    wandb_required = bool(finetune_config.get("wandb_enabled", True))
    wandb_available = _module_available("wandb")
    missing_dependencies = [
        name
        for name, required, available in (
            ("peft", True, peft_available),
            ("trl", True, trl_available),
            ("bitsandbytes", bitsandbytes_required, bitsandbytes_available),
            ("wandb", wandb_required, wandb_available),
        )
        if required and not available
    ]
    dependencies_ready = not missing_dependencies
    emit_trace_event(
        "finetune_started",
        "finetune",
        "started",
        "automatic PEFT adaptation started",
        metadata={
            "run_id": run_id,
            "model_id": model_id,
            "revision": revision,
            "requested_method": requested_method,
            "smoke_only": smoke_only,
            "skip": skip,
            "single_gpu": True,
            "maximum_attempts": 2,
        },
    )
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
            "requested_method": requested_method,
            "peft_available": peft_available,
            "trl_available": trl_available,
            "bitsandbytes_available": bitsandbytes_available,
            "bitsandbytes_required": bitsandbytes_required,
            "wandb_available": wandb_available,
            "wandb_required": wandb_required,
            "missing_dependencies": missing_dependencies,
        },
    )

    datasets: dict[str, list[Any]] = {}
    eval_datasets: dict[str, list[Any]] = {}
    token_counts_by_task: dict[str, dict[str, int]] = {}
    dataset_profiles: dict[str, DatasetProfile] = {}
    task_policies: dict[str, ResolvedFinetunePolicy] = {}
    if not skip and dependencies_ready:
        datasets = load_all_adaptation_datasets(finetune_config.get("datasets"), split="train")
        eval_datasets = load_all_adaptation_datasets(
            finetune_config.get("datasets"), split="eval"
        )
        tokenizer = _load_profile_tokenizer(model_id, revision)
        for task, examples in datasets.items():
            task_dataset_config = (finetune_config.get("datasets") or {}).get(task) or {}
            profile = build_dataset_profile(task, examples, task_dataset_config, tokenizer)
            dataset_profiles[task] = profile
            token_counts_by_task[task] = count_training_tokens(examples)
            task_policies[task] = _resolve_task_policy(
                task=task,
                profile=profile,
                hardware=hardware,
                finetune_config=finetune_config,
                model_metadata=model_metadata,
            )
    else:
        placeholder = DatasetProfile(
            task="multi_task_adaptation",
            task_breadth="general",
            examples=0,
            total_tokens=0,
            mean_tokens=128,
            p95_tokens=min(int(model_metadata.get("context_length") or 2048), 2048),
            p99_tokens=min(int(model_metadata.get("context_length") or 2048), 2048),
        )
        task_policies[placeholder.task] = _resolve_task_policy(
            task=placeholder.task,
            profile=placeholder,
            hardware=hardware,
            finetune_config=finetune_config,
            model_metadata=model_metadata,
        )

    peft_cfg = {
        "method": (
            "per_task"
            if len(task_policies) > 1
            else next(iter(task_policies.values())).method
        ),
        "requested_method": requested_method,
        "hardware": hardware.as_dict(),
        "dataset_profiles": {task: profile.as_dict() for task, profile in dataset_profiles.items()},
        "metadata_model_profile": profile_model_config(
            model_metadata.get("config") or {},
            estimated_params_billion=float(
                model_metadata.get("estimated_params_billion") or 0.1
            ),
        ).as_dict(),
        "task_policies": {task: policy.as_dict() for task, policy in task_policies.items()},
    }
    training_details: dict[str, Any] | None = None
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
            policy = task_policies[task_type]
            emit_trace_event(
                "finetune_task_started",
                "finetune",
                "started",
                f"starting independent {policy.method.upper()} run for {task_type}",
                metadata={
                    "run_id": run_id,
                    "model_id": model_id,
                    "task_type": task_type,
                    "training_examples": len(task_examples),
                    "method": policy.method,
                    "microbatch_size": policy.microbatch_size,
                    "reasons": list(policy.reasons),
                },
            )
            details = _run_task_with_single_fallback(
                run_id=run_id,
                model_id=model_id,
                revision=revision,
                task_type=task_type,
                task_examples=task_examples,
                eval_examples=eval_datasets.get(task_type, []),
                task_dir=root / "finetune" / task_type,
                initial_policy=policy,
                dataset_profile=dataset_profiles[task_type],
                hardware=hardware,
                finetune_config=_task_policy_config(finetune_config, task_type),
                model_metadata=model_metadata,
                smoke_only=smoke_only,
            )
            task_runs[task_type] = details
            emit_trace_event(
                "finetune_task_finished",
                "finetune",
                "completed" if details.get("success") else "failed",
                str(details.get("message") or "fine-tune task finished"),
                metadata={
                    "run_id": run_id,
                    "model_id": model_id,
                    "task_type": task_type,
                    "attempts": len(details.get("attempts") or []),
                },
            )
        peft_cfg["task_policies"] = {
            task: task_runs.get(task, {}).get("resolved_policy", policy.as_dict())
            for task, policy in task_policies.items()
        }
        peft_cfg["model_profiles"] = {
            task: details.get("model_profile")
            for task, details in task_runs.items()
            if details.get("model_profile")
        }
        successful_runs = {task: item for task, item in task_runs.items() if item.get("success")}
        training_details = {
            "success": bool(task_runs) and len(successful_runs) == len(task_runs),
            "message": f"completed {len(successful_runs)}/{len(task_runs)} independent PEFT runs",
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
                if _has_oom_failure(task_runs)
                else FailureCategory.FINETUNE_FAILED.value
            )

    smoke = {
        "status": status,
        "peft_available": peft_available,
        "trl_available": trl_available,
        "wandb_available": wandb_available,
        "wandb_enabled": wandb_required,
        "smoke_only": smoke_only,
        "failure_category": failure_category,
        "message": (
            "fine-tune skipped by CLI flag"
            if skip
            else f"missing required dependencies: {', '.join(missing_dependencies)}"
            if not dependencies_ready
            else str((training_details or {}).get("message") or "fine-tune did not run")
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
        task_training = training_details["task_runs"][task]
        task_results.append(
            compute_adaptation_metrics(
                model_id=model_id,
                run_id=run_id,
                task_type=task,
                peft_config=task_training["resolved_policy"],
                base_quality_score=float(base_scores[task]),
                adapted_quality_score=adapted_scores[task],
                token_counts=token_counts_by_task[task],
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
            "automatic PEFT adaptation completed"
            if status in {"smoke_completed", "training_completed"}
            else smoke["message"]
        ),
        metadata={
            "run_id": run_id,
            "model_id": model_id,
            "requested_method": requested_method,
            "finetune_status": status,
            "training_completed": training_completed,
        },
        metrics={"adaptation_lift": overall["adaptation_lift"]},
    )
    return result


def _run_task_with_single_fallback(
    *,
    run_id: str,
    model_id: str,
    revision: str | None,
    task_type: str,
    task_examples: list[Any],
    eval_examples: list[Any],
    task_dir: Path,
    initial_policy: ResolvedFinetunePolicy,
    dataset_profile: DatasetProfile,
    hardware: HardwareProfile,
    finetune_config: dict[str, Any],
    model_metadata: dict[str, Any],
    smoke_only: bool,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    policy = initial_policy
    for attempt_number in (1, 2):
        try:
            attempt = run_training_attempt_subprocess(
                attempt_number=attempt_number,
                run_id=run_id,
                model_id=model_id,
                revision=revision,
                run_dir=task_dir,
                train_datasets={task_type: task_examples},
                eval_datasets={task_type: eval_examples},
                peft_config=policy.as_dict(),
                smoke_only=smoke_only,
                task_type=task_type,
            )
        except Exception as exc:
            attempt = {
                "success": False,
                "retry_recommended": False,
                "retry_reason": "attempt_process_failed",
                "message": f"training attempt process failed: {type(exc).__name__}: {exc}",
                "attempt_number": attempt_number,
            }
        attempt["resolved_policy"] = policy.as_dict()
        attempts.append(attempt)
        if attempt.get("success"):
            result = dict(attempt)
            result["attempts"] = [dict(item) for item in attempts]
            return result
        if attempt_number == 2 or not attempt.get("retry_recommended"):
            break
        try:
            policy = resolve_finetune_policy(
                model_size_billion=float(
                    model_metadata.get("estimated_params_billion") or 0.1
                ),
                active_model_size_billion=_active_model_size_billion(model_metadata),
                architecture=model_metadata.get("architecture"),
                context_length=int(model_metadata.get("context_length") or 2048),
                dataset=dataset_profile,
                hardware=hardware,
                config=finetune_config,
                retry_of=policy,
            )
        except ValueError as exc:
            attempt["message"] = f"{attempt.get('message')}; fallback blocked: {exc}"
            break
    final = dict(attempts[-1])
    final["attempts"] = attempts
    final["resolved_policy"] = policy.as_dict()
    return final


def _resolve_task_policy(
    *,
    task: str,
    profile: DatasetProfile,
    hardware: HardwareProfile,
    finetune_config: dict[str, Any],
    model_metadata: dict[str, Any],
) -> ResolvedFinetunePolicy:
    return resolve_finetune_policy(
        model_size_billion=float(model_metadata.get("estimated_params_billion") or 0.1),
        active_model_size_billion=_active_model_size_billion(model_metadata),
        architecture=model_metadata.get("architecture"),
        context_length=int(model_metadata.get("context_length") or 2048),
        dataset=profile,
        hardware=hardware,
        config=_task_policy_config(finetune_config, task),
    )


def _task_policy_config(config: dict[str, Any], task: str) -> dict[str, Any]:
    merged = {key: value for key, value in config.items() if key != "datasets"}
    task_overrides = (config.get("task_overrides") or {}).get(task) or {}
    merged.update(task_overrides)
    return merged


def _active_model_size_billion(model_metadata: dict[str, Any]) -> float | None:
    profile = profile_model_config(
        model_metadata.get("config") or {},
        estimated_params_billion=float(model_metadata.get("estimated_params_billion") or 0.1),
    )
    if profile.active_parameters_estimate is None:
        return None
    return profile.active_parameters_estimate / 1_000_000_000


def _load_profile_tokenizer(model_id: str, revision: str | None) -> Any | None:
    try:
        from transformers import AutoTokenizer

        model_revision = revision if revision and not revision.startswith("local-phase") else None
        return AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    except Exception:
        return None


def _has_oom_failure(task_runs: dict[str, dict[str, Any]]) -> bool:
    return any(
        not result.get("success") and result.get("retry_reason") == "cuda_oom"
        for result in task_runs.values()
    )


def _write_finetune_artifacts(
    run_dir: Path,
    peft_config: dict[str, Any],
    adaptation_result: dict[str, Any],
    adapter_eval: dict[str, Any],
) -> dict[str, str]:
    paths = {
        "finetune_config": run_dir / "finetune_config.json",
        "finetune_plan": run_dir / "finetune_plan.json",
        "model_profiles": run_dir / "model_profiles.json",
        "adaptation_results": run_dir / "adaptation_results.json",
        "adapter_eval": run_dir / "adapter_eval.json",
    }
    details = (adaptation_result.get("smoke") or {}).get("details") or {}
    for task_type, task_details in details.get("task_runs", {}).items():
        if task_details.get("adapter_dir"):
            paths[f"adapter_{task_type}"] = Path(task_details["adapter_dir"])
        for attempt in task_details.get("attempts", []):
            attempt_number = int(attempt.get("attempt_number") or 0)
            attempt_log = attempt.get("attempt_log")
            if attempt_number and attempt_log:
                paths[f"finetune_attempt_{task_type}_{attempt_number}_log"] = Path(attempt_log)
                result_path = Path(attempt_log).with_name("result.json")
                if result_path.exists():
                    paths[f"finetune_attempt_{task_type}_{attempt_number}"] = result_path
    paths["finetune_config"].write_text(
        json.dumps(peft_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["finetune_plan"].write_text(
        json.dumps(peft_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["model_profiles"].write_text(
        json.dumps(peft_config.get("model_profiles") or {}, indent=2, sort_keys=True) + "\n",
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
