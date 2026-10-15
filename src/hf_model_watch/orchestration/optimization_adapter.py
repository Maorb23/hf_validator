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

    def submit(self, intent: NewStudyIntent) -> dict[str, Any]:
        c = intent.context
        # Explicit semantic fields override execution defaults so workloads cannot drift.
        payload = {
            **self.execution,
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
        overrides = self.execution.get("optuna_by_framework", {})
        if any(set(v) & {"objective", "objectives", "client_args"} for v in overrides.values()):
            raise ValueError("Framework overrides cannot change objectives or workload")
        if payload["optuna"].get("client_args") or "objective" in payload["optuna"]:
            raise ValueError("Execution settings cannot override fixed workload or objectives")
        return self.controller.submit(payload, idempotency_key=intent.idempotency_key)

    def status(self, reference: str) -> dict[str, Any]:
        return self.controller.status(reference)

    def cancel(self, reference: str) -> dict[str, Any]:
        return self.controller.cancel(reference)

    def results(self, reference: str, context: StudyContext) -> StudySummary:
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
            # Full measurements, including successful trials from failed frameworks.
            for row in manifest.get("test_results", []):
                metrics = row.get("results", {})
                if metrics.get("benchmark_valid") is False or metrics.get("failure_rate", 0) != 0:
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
                        )
                    )
                except ValueError:
                    continue
        return summary
