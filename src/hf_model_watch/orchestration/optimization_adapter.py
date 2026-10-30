"""Additive bridge to llm-optimizer; its controller owns all remote resources."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Protocol

from hf_model_watch.optimization.catalog import incompatibilities
from hf_model_watch.optimization.contracts import (
    Candidate,
    NewStudyIntent,
    StudyContext,
    StudySummary,
    TrialRecord,
)


def _client_args(context: StudyContext) -> list[str]:
    w = context.workload
    return [
        "dataset_name=random",
        f"random_input_len={w.input_tokens}",
        f"random_output_len={w.output_tokens}",
        "random_range_ratio=1.0",
        f"num_prompts={w.num_prompts}",
        f"seed={w.seed}",
        f"max_concurrency={context.concurrency}",
    ]


def _constraint_text(context: StudyContext) -> str | None:
    aliases = {
        "mean_e2e_latency_ms": "e2e_latency:mean",
        "mean_ttft_ms": "ttft:mean",
        "p95_e2e_latency_ms": "e2e_latency:p95",
        "p95_ttft_ms": "ttft:p95",
        "output_throughput": "output_throughput",
    }
    return (
        ";".join(
            f"{aliases[b.metric]}{b.operator}{b.value}" + ("ms" if b.metric.endswith("_ms") else "")
            for b in context.constraints
        )
        or None
    )


class OptimizerController(Protocol):
    def submit(
        self, data: dict[str, Any], idempotency_key: str | None = None
    ) -> dict[str, Any]: ...
    def status(self, run_id: str) -> dict[str, Any]: ...
    def results(self, run_id: str) -> dict[str, Any]: ...
    def cancel(self, run_id: str, framework: str | None = None) -> dict[str, Any]: ...


class OptimizationAdapter:
    def __init__(self, controller: OptimizerController, execution: dict[str, Any]):
        self.controller = controller
        self.execution = execution

    def build_request(self, intent: NewStudyIntent) -> dict[str, Any]:
        """Build without submitting; shared by preparation and execution."""
        c = intent.context
        manifest = None
        if c.workload.dataset is None and self.execution.get('workload_manifest') is not None:
            raise ValueError('Custom workload manifest requires a dataset identity')
        if c.workload.dataset is not None:
            from llm_optimizer.hf_workload import prepare_hf_workload, validate_manifest
            w = c.workload
            dataset = w.dataset.model_dump(exclude={'schema_version'})
            if w.dataset.data_file is None:
                # Older manifests have no selected-file field in their identity.
                dataset.pop('data_file')
            cached = self.execution.get('workload_manifest')
            if cached is not None:
                manifest = validate_manifest(cached)
                # Source adapters add hashed provenance, not source-selection fields.
                source_selection = {k: v for k, v in manifest['dataset'].items()
                                    if k not in {'source_format', 'adapter_version'}}
                if (source_selection != dataset or manifest['tokenizer'] != c.model_id
                        or manifest['tokenizer_revision'] != c.revision
                        or manifest['seed'] != w.seed
                        or len(manifest['requests']) != w.num_prompts
                        or any(r['prompt_len'] != w.input_tokens
                               or r['output_len'] != w.output_tokens
                               for r in manifest['requests'])):
                    raise ValueError('Prepared custom workload differs from intent')
            else:
                if w.dataset.data_file is not None:
                    raise ValueError(
                        'Selected dataset data_file requires a frozen workload manifest')
                manifest = prepare_hf_workload(dataset, c.model_id, c.revision,
                    w.input_tokens, w.output_tokens, w.num_prompts, w.seed)
            if w.workload_sha256 and w.workload_sha256 != manifest['sha256']:
                raise ValueError('Custom workload changed since preparation')
            w.dataset.revision = manifest['dataset']['revision']
            w.workload_sha256 = manifest['sha256']
            self.execution['workload_manifest'] = manifest
        from hf_model_watch.optimization.intake import execution_defaults
        execution = execution_defaults(c.model_dump(), self.execution, intent.frameworks)
        # Explicit semantic fields override execution defaults so workloads cannot drift.
        payload = {
            **execution,
            "model": c.model_id,
            "revision": c.revision,
            "frameworks": intent.frameworks,
            "gpu_type": c.gpu_type,
            "gpu_count": c.gpu_count,
            "search_strategy": "optuna",
            "constraints": _constraint_text(c),
            "client_args": _client_args(c),
            "optuna": {**self.execution.get("optuna", {}), "objectives": c.objectives},
            "semantic_context": c.model_dump(mode="json"),
        }
        if manifest is not None:
            payload['workload_manifest'] = manifest
        from llm_optimizer.search_profiles import expanded_server_args, merge_server_args
        from hf_model_watch.optimization.runtime_constraints import (
            GEMMA,
            constrain_trt_files,
            overrides as runtime_overrides,
            validate_request,
        )
        # The optimizer owns parameter domains and framework-native rendering.
        payload["optuna"] = {"trials": 30, "sampler": "TPESampler", **payload["optuna"]}
        payload["server_args"] = dict(execution.get("server_args", {}))
        payload["server_config_files"] = dict(execution.get("server_config_files", {}))
        for framework in intent.frameworks:
            if framework not in {"vllm", "sglang", "trtllm"}:
                continue
            defaults, files = expanded_server_args(
                framework, c.concurrency, c.workload.input_tokens)
            defaults = merge_server_args(defaults, runtime_overrides(framework, c))
            if c.model_id == GEMMA and framework == "trtllm":
                defaults, files = constrain_trt_files(defaults, files)
            supplied = payload["server_args"].get(framework, [])
            # Optuna-specific overrides must not collide with profile dimensions.
            tuned = {
                **payload["optuna"],
                **execution.get("optuna_by_framework", {}).get(framework, {})
            }.get("server_args", [])
            merged = merge_server_args(defaults, [*supplied, *tuned])
            payload["server_args"][framework] = merged[:-len(tuned)] if tuned else merged
            if files and any(
                "@LLM_OPTIMIZER_CONFIG@/" in arg for arg in payload["server_args"][framework]
            ):
                payload["server_config_files"][framework] = {
                    **files, **payload["server_config_files"].get(framework, {})}
        overrides = self.execution.get("optuna_by_framework", {})
        if any(set(v) & {"objective", "objectives", "client_args"} for v in overrides.values()):
            raise ValueError("Framework overrides cannot change objectives or workload")
        if payload["optuna"].get("client_args") or "objective" in payload["optuna"]:
            raise ValueError("Execution settings cannot override fixed workload or objectives")
        for framework in intent.frameworks:
            settings = {**payload["optuna"], **overrides.get(framework, {})}
            if settings.get("pruner", "NopPruner") != "NopPruner":
                if len(c.objectives) != 1 or c.constraints:
                    raise ValueError("Pruning requires an unconstrained scalar objective")
        deferred = {"server_ready_timeout_seconds", "cache_pinned_model_assets"} & payload.keys()
        if deferred:
            raise ValueError(
                "Worker extensions are deferred for existing images; remove: "
                + ", ".join(sorted(deferred))
            )
        validate_request(payload)
        return payload

    def submit(self, intent: NewStudyIntent) -> dict[str, Any]:
        return self.controller.submit(
            self.build_request(intent), idempotency_key=intent.idempotency_key)

    def status(self, reference: str) -> dict[str, Any]:
        return self.controller.status(reference)

    def cancel(self, reference: str) -> dict[str, Any]:
        return self.controller.cancel(reference)

    def results(self, reference: str, context: StudyContext) -> StudySummary:
        from hf_model_watch.optimization.timing import framework_timings, trial_timings

        report = self.controller.results(reference)
        if report.get("run_id") != reference:
            raise ValueError("Controller returned a different study")
        summary = StudySummary(study_reference=reference, context=context, state=report["state"])
        if report.get("report_uri"):
            summary.artifacts["report"] = report["report_uri"]
        for job in report.get("jobs", []):
            if job.get("error") or job.get("state") == "failed":
                summary.framework_errors[job["framework"]] = job.get("error") or "Framework failed"
        for framework, manifest in report.get("manifests", {}).items():
            try:
                if manifest.get("run_id") != reference or manifest.get("framework") != framework:
                    raise ValueError("Manifest identity mismatch")
                measured = StudyContext.model_validate(manifest.get("semantic_context"))
                # Check execution metadata as well as the semantic envelope.
                if (
                    manifest.get("model") != measured.model_id
                    or manifest.get("gpu_type") != measured.gpu_type
                    or manifest.get("gpu_count") != measured.gpu_count
                    or sorted(manifest.get("workload", [])) != sorted(_client_args(measured))
                    or (measured.workload.dataset is not None
                        and manifest.get('workload_sha256') != measured.workload.workload_sha256)
                    or manifest.get("objectives") != measured.objectives
                    or manifest.get("constraints") != _constraint_text(measured)
                ):
                    raise ValueError("Incomparable measurements: execution metadata mismatch")
                measured.revision = (
                    manifest.get("resolved_revision") or manifest.get("revision") or ""
                )
                mismatch = incompatibilities(context, measured)
                if mismatch:
                    raise ValueError(f"Incomparable measurements: {', '.join(mismatch)}")
            except ValueError as exc:
                summary.framework_errors[framework] = str(exc)
                continue
            artifacts = manifest.get("artifacts", {})
            if manifest.get("manifest_uri"):
                summary.artifacts[framework] = manifest["manifest_uri"]
            summary.importance_per_objective[framework] = manifest.get("study_summary", {}).get(
                "importance_per_objective", {}
            )
            stats = manifest.get("study_summary", {})
            summary.trial_state_counts[framework] = stats.get("trial_state_counts", {})
            summary.failed_trials[framework] = stats.get("failed_trials", [])
            summary.gpu_seconds[framework] = stats.get("gpu_seconds", 0)
            seen_trials = set()
            # Full measurements, including unsuccessful attempts for inspection.
            for row in manifest.get("test_results", []):
                trial = row.get("optimization", {})
                state = stats.get("trial_states", {}).get(
                    str(trial.get("trial_number")), trial.get("state", "COMPLETE")
                )
                trial_id = (
                    str(trial["trial_number"]) if trial.get("trial_number") is not None
                    else f"unreported-{len(seen_trials)}"
                )
                seen_trials.add(trial_id)
                metrics = row.get("results", {})
                config = row.get("config", {})
                identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                numeric = {
                    k: v for k, v in metrics.items()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool) and math.isfinite(v)
                }
                timings = trial_timings(row.get("metadata", {}).get("execution", {}))
                record = TrialRecord(framework=framework, trial_id=trial_id, status=state,
                    configuration=config, configuration_id=identity,
                    metrics=numeric, timings=timings)
                summary.trials.append(record)
                if state != "COMPLETE":
                    continue
                expected = context.workload.num_prompts
                if (metrics.get("total_requests", expected) != expected
                    or metrics.get("completed", expected) != expected):
                    record.status = "INVALID"
                    record.reason = "Incomplete request count"
                    continue
                if metrics.get("benchmark_valid") is False or metrics.get("failure_rate", 0) != 0:
                    record.status = "INVALID"
                    record.reason = "Benchmark invalid or requests failed"
                    continue
                config = row.get("config", {})
                identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
                try:
                    summary.candidates.append(
                        Candidate(
                            configuration_id=identity,
                            framework=framework,
                            configuration=config,
                            metrics={
                                k: v
                                for k, v in metrics.items()
                                if isinstance(v, (int, float))
                                and not isinstance(v, bool)
                                and math.isfinite(v)
                            },
                            artifacts=artifacts,
                            trial_id=trial_id,
                            timings=timings,
                        )
                    )
                except ValueError:
                    continue
            failures = {str(t["number"]): t for t in stats.get("failed_trials", [])}
            states = {**{n: "FAIL" for n in failures}, **stats.get("trial_states", {})}
            for number, state in states.items():
                if str(number) not in seen_trials:
                    failure = failures.get(str(number), {})
                    summary.trials.append(TrialRecord(framework=framework, trial_id=str(number),
                        status=state, reason=failure.get("reason"),
                        timings=trial_timings(failure.get("execution", {}))))
            framework_trials = [t for t in summary.trials if t.framework == framework]
            totals = framework_timings(framework_trials)
            if sum(stats.get("trial_state_counts", {}).values()) > len(framework_trials):
                # Older workers can report counts without exporting every attempt.
                totals = {key: None for key in totals}
            summary.framework_timings[framework] = totals
        return summary
