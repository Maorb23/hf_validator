"""Deterministic selection of comparable measurements, retaining every Pareto point."""

from __future__ import annotations

import math

from .contracts import Candidate, OptimizationResult, SelectionPolicy, StudySummary


def _identity(candidate: Candidate) -> tuple[str, str]:
    return candidate.framework, candidate.configuration_id


def _valid(candidate: Candidate, objectives: list[str]) -> bool:
    return all(
        metric in candidate.metrics and math.isfinite(candidate.metrics[metric])
        for metric in objectives
    )


def _loss(candidate: Candidate, metric: str) -> float:
    from llm_optimizer.optimization import objective_direction

    return candidate.metrics[metric] * (-1 if objective_direction(metric) == "maximize" else 1)


def pareto_front(candidates: list[Candidate], objectives: list[str]) -> list[Candidate]:
    valid = [c for c in candidates if _valid(c, objectives)]
    return sorted(
        [
            c
            for c in valid
            if not any(
                all(_loss(other, m) <= _loss(c, m) for m in objectives)
                and any(_loss(other, m) < _loss(c, m) for m in objectives)
                for other in valid
            )
        ],
        key=_identity,
    )


def analyze(summary: StudySummary, policy: SelectionPolicy) -> OptimizationResult:
    objectives = summary.context.objectives
    frameworks = (
        {candidate.framework for candidate in summary.candidates}
        | set(summary.framework_errors)
        | set(summary.importance_per_objective)
    )
    importance = {
        framework: {
            metric: summary.importance_per_objective.get(framework, {}).get(
                metric,
                {
                    "status": "unavailable",
                    "values": {},
                    "reason": "No stored parameter importance for this objective",
                },
            )
            for metric in objectives
        }
        for framework in sorted(frameworks)
    }
    result = OptimizationResult(
        study_reference=summary.study_reference,
        policy=policy,
        importance_per_objective=importance,
        framework_errors=summary.framework_errors,
        artifacts=summary.artifacts,
    )
    valid = [c for c in summary.candidates if _valid(c, objectives)]
    result.pareto_front = pareto_front(valid, objectives)
    result.best_per_objective = (
        {m: min(valid, key=lambda c: (_loss(c, m), _identity(c))) for m in objectives}
        if valid
        else {}
    )
    bounds = summary.context.constraints + policy.constraints
    feasible = [
        c
        for c in valid
        if all(
            b.metric in c.metrics
            and math.isfinite(c.metrics[b.metric])
            and (
                c.metrics[b.metric] <= b.value
                if b.operator == "<="
                else c.metrics[b.metric] >= b.value
            )
            for b in bounds
        )
    ]
    if not feasible:
        result.reason = "No feasible result satisfies the required metrics and constraints"
        return result
    priority = {
        "latency_first": "mean_e2e_latency_ms",
        "ttft_first": "mean_ttft_ms",
        "throughput_first": "output_throughput",
    }.get(policy.name)
    if priority:
        percentile = priority.replace("mean_", "p95_", 1)
        if percentile in objectives:
            priority = percentile
        if priority not in objectives:
            result.reason = f"Policy {policy.name} requires objective {priority}"
            return result
        result.selected = min(
            feasible,
            key=lambda c: (
                tuple(_loss(c, m) for m in [priority] + [o for o in objectives if o != priority]),
                _identity(c),
            ),
        )
        return result
    if set(policy.weights) - set(objectives):
        raise ValueError("Policy weights must refer to study objectives")
    frontier = pareto_front(feasible, objectives)
    ranges = {
        m: (min(_loss(c, m) for c in feasible), max(_loss(c, m) for c in feasible))
        for m in objectives
    }

    def score(candidate: Candidate) -> tuple[float, tuple[str, str]]:
        losses = [
            (
                0.0 if hi == lo else (_loss(candidate, m) - lo) / (hi - lo),
                policy.weights.get(m, 1.0),
            )
            for m, (lo, hi) in ranges.items()
        ]
        if policy.name == "knee_point":
            # Minimax normalized regret selects the bend nearest the ideal point.
            value = max(loss * weight for loss, weight in losses)
        else:
            value = sum(loss * weight for loss, weight in losses) / sum(w for _, w in losses)
        return value, _identity(candidate)

    result.selected = min(frontier, key=score)
    return result
