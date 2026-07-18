from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from hf_model_watch.errors import FailureCategory
from hf_model_watch.evals.context_drift import run_context_drift_benchmark
from hf_model_watch.evals.latency_bench import run_latency_benchmark
from hf_model_watch.evals.metrics import summarize_records
from hf_model_watch.evals.quality_eval import run_quality_benchmark
from hf_model_watch.hub.lineage import extract_lineage
from hf_model_watch.hub.metadata import (
    extract_architectures,
    extract_context_length,
    extract_license,
    has_safetensors_file,
    sibling_names,
)
from hf_model_watch.runtime.backend_transformers import (
    GenerationResult,
    TransformersSession,
    smoke_generate,
)
from hf_model_watch.runtime.runtime_resolver import resolve_runtime
from hf_model_watch.storage.schemas import (
    CLAIMS_VS_OBSERVED_PATH,
    RunManifest,
    build_reproducibility_snapshot,
    initialize_run_artifacts,
    utc_now_iso,
    write_run_manifest,
)
from hf_model_watch.storage.sqlite_store import SQLiteStore
from hf_model_watch.tracing import emit_trace_event

GenerateFn = Callable[[str, str | None, dict[str, Any]], GenerationResult]


def run_local_smoke_test(
    run_id: str,
    model_id: str,
    config: Any,
    store: SQLiteStore,
    generate_fn: GenerateFn = smoke_generate,
    eval_generate_fn: Callable[[str, int], GenerationResult] | None = None,
) -> dict[str, Any]:
    artifact_root = Path(str(config.storage.get("local_artifact_dir", "artifacts")))
    run_dir = artifact_root / run_id
    metadata = collect_metadata_claims(model_id)
    revision = metadata.get("revision") or "local-phase4-unresolved"
    lineage = extract_lineage(metadata)
    manifest = RunManifest(
        run_id=run_id,
        model_id=model_id,
        revision=revision,
        artifacts={},
        config_snapshot=config.model_dump(),
        reproducibility=build_reproducibility_snapshot(
            model_id,
            revision,
            config.model_dump(),
        ),
    )
    initialize_run_artifacts(artifact_root, manifest)
    store.start_run(
        run_id=run_id,
        model_id=model_id,
        revision=revision,
        model_family=lineage.get("family"),
        size_band=lineage.get("size_band"),
        artifact_uri=str(run_dir),
    )

    decision = resolve_runtime(metadata, config.scan.model_dump(), config.serverless.model_dump())
    manifest = manifest.model_copy(
        update={
            "reproducibility": build_reproducibility_snapshot(
                model_id,
                revision,
                config.model_dump(),
                decision.as_dict(),
            )
        }
    )
    write_run_manifest(run_dir, manifest)
    if decision.fail_fast:
        store.record_failure(
            run_id,
            model_id,
            revision,
            decision.failure_category or FailureCategory.LOAD_FAILED.value,
            decision.failure_message or "Runtime resolution failed",
        )
        _write_claims(run_dir, metadata, decision.as_dict(), observed={})
        raise RuntimeError(decision.failure_message or "Runtime resolution failed")

    emit_trace_event(
        "runtime_resolved",
        "runtime",
        "completed",
        "runtime decision selected",
        metadata={"run_id": run_id, "backend": decision.backend, "dtype": decision.dtype},
    )
    _append_run_event(
        run_dir,
        {
            "event_type": "runtime_resolved",
            "stage": "runtime",
            "status": "completed",
            "message": "runtime decision selected",
            "metadata": {"run_id": run_id, "backend": decision.backend, "dtype": decision.dtype},
            "timestamp": utc_now_iso(),
        },
    )

    try:
        generation = generate_fn(model_id, revision, decision.selected_load_args)
    except Exception as exc:
        store.record_failure(
            run_id,
            model_id,
            revision,
            FailureCategory.LOAD_FAILED.value,
            f"{type(exc).__name__}: {exc}",
        )
        _write_claims(run_dir, metadata, decision.as_dict(), observed={"load_success": False})
        raise

    observed = {
        "tokenizer_load_success": True,
        "transformers_load_support": True,
        "bf16_load_worked": _bf16_available() and decision.dtype == "bf16",
        "four_bit_load_worked": decision.quantization == "4bit",
        "generation": generation.as_dict(),
    }
    claims_path = _write_claims(run_dir, metadata, decision.as_dict(), observed)
    raw_generation_path = run_dir / "raw_generations.jsonl"
    runtime_result_path = run_dir / "runtime_result.json"
    raw_generation_path.write_text(
        json.dumps(generation.as_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime_result = {
        "run_id": run_id,
        "model_id": model_id,
        "revision": revision,
        "metadata": metadata,
        "decision": decision.as_dict(),
        "observed": observed,
    }
    runtime_result_path.write_text(
        json.dumps(runtime_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = manifest.model_copy(
        update={
            "completed_at": utc_now_iso(),
            "status": "completed",
            "artifacts": {
                "claims_vs_observed": str(claims_path),
                "raw_generations": str(raw_generation_path),
                "runtime_result": str(runtime_result_path),
                "trace_events": str(run_dir / "trace_events.jsonl"),
            },
        }
    )
    write_run_manifest(run_dir, manifest)
    _append_run_event(
        run_dir,
        {
            "event_type": "smoke_generation_completed",
            "stage": "runtime",
            "status": "completed",
            "message": "tiny local generation smoke test completed",
            "metadata": {"run_id": run_id, "model_id": model_id, "revision": revision},
            "timestamp": utc_now_iso(),
        },
    )
    store.record_artifact(run_id, "claims_vs_observed", str(claims_path), model_id, revision)
    store.record_artifact(run_id, "raw_generations", str(raw_generation_path), model_id, revision)
    store.record_artifact(run_id, "runtime_result", str(runtime_result_path), model_id, revision)
    store.record_artifact(
        run_id,
        "trace_events",
        str(run_dir / "trace_events.jsonl"),
        model_id,
        revision,
    )
    store.record_runtime_result(run_id, model_id, revision, decision.as_dict(), runtime_result)
    store.record_metadata_result(run_id, model_id, revision, claims_path)
    store.complete_run(
        run_id,
        status="completed",
        selected_backend=decision.backend,
        selected_dtype=decision.dtype,
        selected_quantization=decision.quantization,
    )
    eval_artifacts = _run_phase5_evals(
        run_id=run_id,
        model_id=model_id,
        revision=revision,
        run_dir=run_dir,
        config=config,
        store=store,
        decision=decision.as_dict(),
        eval_generate_fn=eval_generate_fn,
    )
    _update_manifest_eval_artifacts(run_dir, eval_artifacts)
    return runtime_result


def collect_metadata_claims(model_id: str) -> dict[str, Any]:
    api = HfApi()
    info = api.model_info(model_id, files_metadata=True)
    tags = [str(tag) for tag in (getattr(info, "tags", None) or [])]
    siblings = getattr(info, "siblings", None) or []
    files = sibling_names(siblings)
    weight_files = [
        sibling
        for sibling in siblings
        if str(getattr(sibling, "rfilename", "")).endswith((".safetensors", ".bin"))
    ]
    total_weight_size = sum(
        int(getattr(sibling, "size", 0) or 0)
        for sibling in weight_files
        if isinstance(getattr(sibling, "size", None), int | float)
    )
    config = getattr(info, "config", None) or {}
    if not isinstance(config, dict):
        config = {}
    card_data = getattr(info, "cardData", None) or getattr(info, "card_data", None)
    safetensors_metadata = getattr(info, "safetensors", None) or {}
    if hasattr(safetensors_metadata, "model_dump"):
        safetensors_metadata = safetensors_metadata.model_dump()
    parameter_count = (
        safetensors_metadata.get("total") if isinstance(safetensors_metadata, dict) else None
    )
    estimated_params_billion = (
        float(parameter_count) / 1_000_000_000
        if isinstance(parameter_count, int | float) and parameter_count > 0
        else total_weight_size / 2_000_000_000
        if total_weight_size > 0
        else None
    )
    return {
        "model_id": model_id,
        "revision": getattr(info, "sha", None),
        "tags": tags,
        "license": extract_license(card_data, tags),
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "config": config,
        "architecture": extract_architectures(config),
        "context_length": extract_context_length(config, tags),
        "has_safetensors": has_safetensors_file(files),
        "shard_count": len([file for file in files if file.endswith(".safetensors")]),
        "weight_file_count": len(weight_files),
        "total_weight_file_size_bytes": total_weight_size,
        "estimated_params_billion": estimated_params_billion,
        "sibling_files": files,
    }


def _write_claims(
    run_dir: Path,
    claims: dict[str, Any],
    decision: dict[str, Any],
    observed: dict[str, Any],
) -> Path:
    path = run_dir / CLAIMS_VS_OBSERVED_PATH
    payload = {
        "claims": claims,
        "runtime_decision": decision,
        "observed": observed,
        "verification": {
            "remote_code_required": decision.get("trust_remote_code_required"),
            "vllm_support_attempt": decision.get("vllm"),
            "selected_backend": decision.get("backend"),
            "selected_dtype": decision.get("dtype"),
            "selected_quantization": decision.get("quantization"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _append_run_event(run_dir: Path, event: dict[str, Any]) -> None:
    with (run_dir / "trace_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _run_phase5_evals(
    run_id: str,
    model_id: str,
    revision: str | None,
    run_dir: Path,
    config: Any,
    store: SQLiteStore,
    decision: dict[str, Any],
    eval_generate_fn: Callable[[str, int], GenerationResult] | None = None,
) -> dict[str, str]:
    if eval_generate_fn is None:
        session = TransformersSession(model_id, revision, decision.get("selected_load_args") or {})
        generate = session.generate
    else:
        generate = eval_generate_fn
    backend = str(decision.get("backend") or "transformers")
    latency_path = run_dir / "latency_results.jsonl"
    quality_path = run_dir / "quality_results.jsonl"
    context_path = run_dir / "context_drift_results.jsonl"
    latency_records, latency_summary = run_latency_benchmark(
        model_id=model_id,
        backend=backend,
        dtype=decision.get("dtype"),
        quantization=decision.get("quantization"),
        generate_fn=generate,
        output_path=latency_path,
        dataset_config=_ground_truth_dataset_config(config),
    )
    quality_records, quality_summary = run_quality_benchmark(
        model_id=model_id,
        backend=backend,
        generate_fn=generate,
        output_path=quality_path,
        dataset_config=_ground_truth_dataset_config(config),
    )
    context_records, context_summary = run_context_drift_benchmark(
        model_id=model_id,
        backend=backend,
        claimed_context=decision.get("max_context"),
        max_context_tokens=_context_max_tokens(config),
        generate_fn=generate,
        output_path=context_path,
    )
    quality_summary["avg_quality_score"] = _avg_quality_score(quality_records)
    combined_summary = summarize_records([*latency_records, *quality_records, *context_records])
    store.record_latency_results(run_id, model_id, revision, latency_summary, latency_records)
    store.record_quality_results(run_id, model_id, revision, quality_summary, quality_records)
    store.record_context_drift_results(run_id, model_id, revision, context_summary, context_records)
    store.record_artifact(run_id, "latency_results", str(latency_path), model_id, revision)
    store.record_artifact(run_id, "quality_results", str(quality_path), model_id, revision)
    store.record_artifact(run_id, "context_drift_results", str(context_path), model_id, revision)
    store.complete_run(
        run_id,
        status="completed",
        base_score=quality_summary["avg_quality_score"],
        latency_summary=json.dumps(combined_summary, sort_keys=True),
        context_drift_summary=json.dumps(context_summary, sort_keys=True),
    )
    _append_run_event(
        run_dir,
        {
            "event_type": "evals_completed",
            "stage": "evals",
            "status": "completed",
            "message": "latency, quality, and context drift benchmarks completed",
            "metadata": {"run_id": run_id, "model_id": model_id},
            "timestamp": utc_now_iso(),
        },
    )
    return {
        "latency_results": str(latency_path),
        "quality_results": str(quality_path),
        "context_drift_results": str(context_path),
    }


def _update_manifest_eval_artifacts(run_dir: Path, eval_artifacts: dict[str, str]) -> None:
    manifest_path = run_dir / "run_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["artifacts"] = {**data.get("artifacts", {}), **eval_artifacts}
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _avg_quality_score(records: list[dict[str, Any]]) -> float:
    values = [float(record.get("quality_score") or 0.0) for record in records]
    return sum(values) / max(len(values), 1)


def _context_max_tokens(config: Any) -> int:
    eval_tasks = getattr(config, "eval_tasks", {})
    if not isinstance(eval_tasks, dict):
        return 4096
    context_config = eval_tasks.get("context", {})
    if not isinstance(context_config, dict):
        return 4096
    return int(context_config.get("max_tokens", 4096))


def _ground_truth_dataset_config(config: Any) -> dict[str, Any]:
    eval_tasks = getattr(config, "eval_tasks", {})
    if not isinstance(eval_tasks, dict):
        return {}
    dataset_config = eval_tasks.get("ground_truth_dataset", {})
    return dataset_config if isinstance(dataset_config, dict) else {}


def _bf16_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
