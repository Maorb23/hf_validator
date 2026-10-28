from itertools import combinations
from unittest.mock import Mock

import pytest
from llm_optimizer.optimization import extract_objective, objective_direction

from hf_model_watch.optimization.analysis import analyze
from hf_model_watch.optimization.contracts import (
    Candidate,
    NewStudyIntent,
    SelectionPolicy,
    StudyContext,
    StudySummary,
)
from hf_model_watch.optimization.objectives import LABELS, objective_options
from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter


@pytest.mark.parametrize("concurrency", [1, 24, 128])
def test_every_p95_objective_combination_reaches_native_optimizer(concurrency):
    options = objective_options()
    assert len(options) == 7
    for option in options:
        names = option["value"].split(",")
        context = StudyContext(model_id="org/model", revision="a" * 40,
            gpu_type="gpu-h100-sxm", concurrency=concurrency,
            workload={"input_tokens": 16, "output_tokens": 8}, objectives=names)
        intent = NewStudyIntent(context=context, frameworks=["vllm"], idempotency_key="test")
        payload = OptimizationAdapter(Mock(), {}).build_request(intent)
        assert payload["optuna"]["objectives"] == names
        assert payload["semantic_context"]["objectives"] == names
        assert option["directions"] == [objective_direction(m) for m in names]
        for metric in names:
            assert LABELS[metric] in option["label"]
        metrics = {m: i + 1 for i, m in enumerate(names)}
        assert extract_objective(metrics, option["value"]) == (
            list(metrics.values()) if len(names) > 1 else metrics[names[0]])


def test_p95_frontier_selection_and_constraints_do_not_use_means():
    names = list(LABELS)
    context = StudyContext(model_id="org/model", revision="a" * 40,
        gpu_type="gpu-h100-sxm", concurrency=1,
        workload={"input_tokens": 16, "output_tokens": 8}, objectives=names,
        constraints=[{"metric": "p95_ttft_ms", "operator": "<=", "value": 20}])
    def candidate(name, throughput, e2e, ttft):
        return Candidate(configuration_id=name, framework="vllm", configuration={},
            metrics=dict(zip(names, [throughput, e2e, ttft], strict=True)) | {
                "mean_e2e_latency_ms": 999 - e2e, "mean_ttft_ms": 999 - ttft})
    candidates = [candidate("a", 100, 10, 30), candidate("b", 90, 20, 10),
                  candidate("dominated", 80, 25, 15)]
    summary = StudySummary(study_reference="run", context=context, state="completed",
                           candidates=candidates)
    for policy in ["balanced", "knee_point", "latency_first", "ttft_first", "throughput_first"]:
        result = analyze(summary, SelectionPolicy(name=policy))
        assert {c.configuration_id for c in result.pareto_front} == {"a", "b"}
        assert result.selected.configuration_id == "b"
    payload = OptimizationAdapter(Mock(), {}).build_request(NewStudyIntent(
        context=context, frameworks=["vllm"], idempotency_key="constraints"))
    assert payload["constraints"] == "ttft:p95<=20.0ms"


def test_duplicates_and_native_pruning_restrictions():
    base = dict(model_id="org/model", revision="a" * 40, gpu_type="gpu-h100-sxm",
                concurrency=1, workload={"input_tokens": 16, "output_tokens": 8})
    with pytest.raises(ValueError, match="unique"):
        StudyContext(**base, objectives=["p95_ttft_ms", "p95_ttft_ms"])
    for pair in combinations(LABELS, 2):
        context = StudyContext(**base, objectives=list(pair))
        with pytest.raises(ValueError, match="Pruning requires"):
            OptimizationAdapter(Mock(), {"optuna": {"pruner": "MedianPruner"}}).build_request(
                NewStudyIntent(context=context, frameworks=["vllm"], idempotency_key="prune"))
