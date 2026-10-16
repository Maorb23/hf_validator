from __future__ import annotations

from unittest.mock import Mock, call

import pytest

from hf_model_watch.optimization.analysis import analyze
from hf_model_watch.optimization.catalog import FixtureStudyCatalog, incompatibilities
from hf_model_watch.optimization.contracts import (
    Candidate,
    NewStudyIntent,
    ReuseIntent,
    SelectionPolicy,
    StudyContext,
    StudySummary,
    objectives_for,
)
from hf_model_watch.optimization.workflow import OptimizationWorkflow
from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter
from hf_model_watch.recommender.run_state import load_run, run_dir, save_run


def context(**changes):
    return StudyContext.model_validate(
        {
            "model_id": "org/model",
            "revision": "abc123",
            "gpu_type": "gpu-h100-sxm",
            "workload": {"input_tokens": 128, "output_tokens": 32},
            "concurrency": 4,
            **changes,
        }
    )


def candidate(identity, throughput, latency, framework="vllm", **extra):
    return Candidate(
        configuration_id=identity,
        framework=framework,
        configuration={"batch": identity},
        metrics={"output_throughput": throughput, "mean_e2e_latency_ms": latency, **extra},
    )


def study():
    return StudySummary(
        study_reference="prior",
        context=context(),
        state="partial",
        candidates=[
            candidate("a", 100, 10),
            candidate("b", 200, 30),
            candidate("c", 150, 15),
            candidate("d", 90, 40),
        ],
        framework_errors={"sglang": "failed"},
        artifacts={"report": "s3://bucket/report"},
    )


def selected_run(root):
    run = {
        "run_id": "run_opt",
        "status": "recommended",
        "selected_model_id": "org/model",
        "selection": {"model_id": "org/model", "revision": "abc123"},
        "recommendations": [{"model_id": "org/model"}],
    }
    save_run(run_dir(root, "run_opt"), run)
    return load_run(root, "run_opt")[1]


def test_skip_leaves_recommendation_unchanged_and_catalog_is_empty(tmp_path):
    before = selected_run(tmp_path)
    workflow = OptimizationWorkflow(tmp_path)
    assert workflow.status("run_opt")["status"] == "not_requested"
    assert workflow.list_prior_studies("run_opt", context()) == {
        "studies": [],
        "message": "no prior studies available",
    }
    assert load_run(tmp_path, "run_opt")[1] == before


def test_objective_mapping_and_validation():
    assert objectives_for(1) == ["mean_e2e_latency_ms", "mean_ttft_ms"]
    assert objectives_for(4) == ["output_throughput", "mean_e2e_latency_ms"]
    assert context(objectives=["output_throughput"]).objectives == ["output_throughput"]
    with pytest.raises(ValueError):
        context(objectives=["invalid"])
    with pytest.raises(ValueError):
        objectives_for(0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "other"),
        ("model_id", "other/model"),
        ("gpu_type", "gpu-a100"),
        ("gpu_count", 2),
        ("concurrency", 5),
        ("workload", {"input_tokens": 256, "output_tokens": 32}),
        ("constraints", [{"metric": "mean_e2e_latency_ms", "operator": "<=", "value": 20}]),
    ],
)
def test_compatibility_checks(field, value):
    assert field in incompatibilities(context(), context(**{field: value}))


def test_reuse_only_reads_catalog_and_preserves_failed_framework(tmp_path):
    selected_run(tmp_path)
    adapter = Mock()
    workflow = OptimizationWorkflow(tmp_path, adapter, FixtureStudyCatalog([study()]))
    intent = ReuseIntent(study_reference="prior", context=context())
    result = workflow.reuse("run_opt", intent)
    assert result["selected"]["configuration_id"] == "c"
    assert result["framework_errors"] == {"sglang": "failed"}
    assert {c["configuration_id"] for c in result["pareto_front"]} == {"a", "b", "c"}
    assert workflow.status("run_opt")["status"] == "analyzed"
    with pytest.raises(ValueError, match="read-only"):
        workflow.cancel("run_opt")
    assert not adapter.mock_calls
    with pytest.raises(ValueError, match="incomparable"):
        workflow.reuse(
            "run_opt", ReuseIntent(study_reference="prior", context=context(concurrency=5))
        )


@pytest.mark.parametrize(
    "policy,identity",
    [
        ("latency_first", "a"),
        ("throughput_first", "b"),
        ("balanced", "c"),
        ("knee_point", "c"),
    ],
)
def test_policies_are_deterministic_and_preserve_complete_front(policy, identity):
    summary = study()
    first = analyze(summary, SelectionPolicy(name=policy))
    summary.candidates.reverse()
    assert analyze(summary, SelectionPolicy(name=policy)) == first
    assert first.selected.configuration_id == identity
    assert len(first.pareto_front) == 3
    assert first.best_per_objective["output_throughput"].configuration_id == "b"


def test_ttft_and_constrained_selection_and_invalid_metrics():
    summary = study()
    summary.context = context(concurrency=1)
    summary.candidates = [
        candidate("a", 100, 10, mean_ttft_ms=5),
        candidate("b", 200, 30, mean_ttft_ms=1),
    ]
    assert analyze(summary, SelectionPolicy(name="ttft_first")).selected.configuration_id == "b"
    unavailable = analyze(summary, SelectionPolicy(name="throughput_first"))
    assert unavailable.selected is None and "requires objective" in unavailable.reason
    summary = study()
    policy = SelectionPolicy(
        name="constrained",
        constraints=[{"metric": "mean_e2e_latency_ms", "operator": "<=", "value": 12}],
    )
    assert analyze(summary, policy).selected.configuration_id == "a"
    policy.constraints[0].value = 1
    assert analyze(summary, policy).selected is None
    summary.candidates[0].metrics = {}
    summary.candidates[1].metrics["output_throughput"] = float("nan")
    assert {x.configuration_id for x in analyze(summary, SelectionPolicy()).pareto_front} == {"c"}


def test_submission_is_idempotent_and_locks_selected_intent(tmp_path):
    before = selected_run(tmp_path)
    controller = Mock()
    controller.submit.return_value = controller.status.return_value = {
        "run_id": "remote",
        "state": "running",
    }
    adapter = OptimizationAdapter(
        controller, {"images": {"vllm": "image"}, "bucket": "bucket", "dry_run": True}
    )
    workflow = OptimizationWorkflow(tmp_path, adapter)
    intent = NewStudyIntent(context=context(), frameworks=["vllm"], idempotency_key="request-1")
    assert workflow.start("run_opt", intent) == workflow.start("run_opt", intent)
    controller.submit.assert_called_once()
    payload = controller.submit.call_args.args[0]
    assert controller.submit.call_args.kwargs == {"idempotency_key": "request-1"}
    assert payload["optuna"]["objectives"] == context().objectives
    assert "max_concurrency=4" in payload["client_args"]
    after = load_run(tmp_path, "run_opt")[1]
    assert after["recommendations"] == before["recommendations"]
    assert after["optimization_reference"]["study_reference"] == "remote"
    with pytest.raises(ValueError, match="different optimization intent"):
        workflow.start("run_opt", intent.model_copy(update={"idempotency_key": "other"}))


def test_controller_results_keep_successful_trials_from_failed_framework():
    controller = Mock()
    manifests = {}
    for framework in ["vllm", "sglang"]:
        manifests[framework] = {
            "run_id": "remote",
            "framework": framework,
            "semantic_context": context().model_dump(),
            "model": context().model_id,
            "gpu_type": context().gpu_type,
            "gpu_count": 1,
            "workload": [
                "dataset_name=random",
                "random_input_len=128",
                "random_output_len=32",
                "random_range_ratio=1.0",
                "num_prompts=100",
                "seed=1",
                "max_concurrency=4",
            ],
            "objectives": context().objectives,
            "constraints": None,
            "resolved_revision": "abc123",
            "status": "failed" if framework == "sglang" else "completed",
            "artifacts": {"results.json": f"s3://bucket/{framework}/results.json"},
            "test_results": [
                {"config": {"batch": 3}, "optimization": {"trial_number": 3},
                 "results": {"output_throughput": 999, "mean_e2e_latency_ms": 1}},
                {"config": {"batch": 4}, "optimization": {"state": "PRUNED"},
                 "results": {"output_throughput": 999, "mean_e2e_latency_ms": 1}},
                {"config": {"batch": 5},
                 "results": {"completed": 50, "total_requests": 50,
                             "output_throughput": 999, "mean_e2e_latency_ms": 1}},
                {
                    "config": {"batch": 1},
                    "results": {"output_throughput": 100, "mean_e2e_latency_ms": 10},
                },
                {
                    "config": {"batch": 2},
                    "results": {
                        "benchmark_valid": False,
                        "output_throughput": 200,
                        "mean_e2e_latency_ms": 5,
                    },
                },
            ],
            "study_summary": {
                "trial_states": {"3": "FAIL"},
                "importance_per_objective": {
                    "output_throughput": {"status": "available", "values": {"batch": 1}}
                }
            },
        }
    controller.results.return_value = {
        "run_id": "remote",
        "state": "partial",
        "manifests": manifests,
        "jobs": [{"framework": "sglang", "state": "failed", "error": "exit 1"}],
    }
    adapter = OptimizationAdapter(controller, {})
    summary = adapter.results("remote", context())
    assert len(summary.candidates) == 2
    assert summary.framework_errors == {"sglang": "exit 1"}
    assert summary.importance_per_objective["vllm"]["output_throughput"]["values"] == {"batch": 1}
    manifests["sglang"]["resolved_revision"] = "different"
    summary = adapter.results("remote", context())
    assert len(summary.candidates) == 1
    assert "revision" in summary.framework_errors["sglang"]


def test_nanoclaw_optional_actions_use_shared_workflow(tmp_path):
    import json

    from hf_model_watch.nanoclaw_mcp import NanoClawMcpServer
    from hf_model_watch.openclaw_adapter import OpenClawToolAdapter

    selected_run(tmp_path)
    workflow = OptimizationWorkflow(tmp_path, catalog=FixtureStudyCatalog([study()]))
    server = NanoClawMcpServer(
        OpenClawToolAdapter(state_root=tmp_path, optimization_workflow=workflow)
    )
    response = server.handle(
        {
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "reuse_optimization_study",
                "arguments": {
                    "run_id": "run_opt",
                    "intent": ReuseIntent(context=context(), study_reference="prior").model_dump(
                        mode="json"
                    ),
                },
            },
        }
    )
    assert (
        json.loads(response["result"]["content"][0]["text"])["selected"]["configuration_id"] == "c"
    )


def test_known_controller_study_reuse_only_reads_results(tmp_path):
    selected_run(tmp_path)
    adapter = Mock()
    adapter.results.return_value = study()
    workflow = OptimizationWorkflow(tmp_path, adapter)
    result = workflow.reuse("run_opt", ReuseIntent(study_reference="prior", context=context()))
    assert result["study_reference"] == "prior"
    assert adapter.mock_calls == [call.results("prior", context())]


def test_retry_after_reference_write_failure_uses_controller_idempotency(tmp_path, monkeypatch):
    from hf_model_watch.optimization import workflow as workflow_module

    selected_run(tmp_path)
    controller = Mock()
    stored = {}

    def submit(payload, idempotency_key=None):
        stored.setdefault(idempotency_key, {"run_id": "remote", "state": "running"})
        return stored[idempotency_key]

    controller.submit.side_effect = submit
    workflow = OptimizationWorkflow(tmp_path, OptimizationAdapter(controller, {}))
    intent = NewStudyIntent(context=context(), frameworks=["vllm"], idempotency_key="stable")
    real_save = workflow_module.save_run
    monkeypatch.setattr(workflow_module, "save_run", Mock(side_effect=OSError("interrupted write")))
    with pytest.raises(OSError):
        workflow.start("run_opt", intent)
    monkeypatch.setattr(workflow_module, "save_run", real_save)
    assert workflow.start("run_opt", intent)["run_id"] == "remote"
    assert len(stored) == 1
    assert controller.submit.call_args_list[0] == controller.submit.call_args_list[1]


def test_cli_optional_status_and_empty_list_are_available(tmp_path):
    import json

    from typer.testing import CliRunner

    from hf_model_watch.cli import app

    before = selected_run(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app, ["optimization", "status", "--run-id", "run_opt", "--state-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "not_requested"
    request = tmp_path / "context.json"
    request.write_text(context().model_dump_json())
    result = runner.invoke(
        app,
        [
            "optimization",
            "list-prior-studies",
            "--run-id",
            "run_opt",
            "--request-file",
            str(request),
            "--state-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["message"] == "no prior studies available"
    assert load_run(tmp_path, "run_opt")[1] == before


def test_submission_contract_matches_optimizer_and_constraint_units():
    remote = pytest.importorskip("llm_optimizer.remote.contracts")
    from llm_optimizer.optimization import constraint_violations
    from llm_optimizer.performance import parse_slo_constraints

    controller = Mock()
    controller.submit.return_value = {"run_id": "remote", "state": "running"}
    fixed = context(
        constraints=[
            {"metric": "mean_e2e_latency_ms", "operator": "<=", "value": 20},
            {"metric": "output_throughput", "operator": ">=", "value": 150},
        ]
    )
    adapter = OptimizationAdapter(controller, {"images": {"vllm": "image"}, "bucket": "bucket"})
    adapter.submit(NewStudyIntent(context=fixed, frameworks=["vllm"], idempotency_key="stable"))
    request = remote.OptimizationRequest.parse(controller.submit.call_args.args[0])
    bounds = parse_slo_constraints(request.constraints)
    assert constraint_violations({"mean_e2e_latency_ms": 15, "output_throughput": 200}, bounds) == [
        -5,
        -50,
    ]
    assert request.semantic_context == fixed.model_dump()


def test_version_and_objective_compatibility_are_explicit():
    assert "objectives" in incompatibilities(context(), context(concurrency=1))
    with pytest.raises(ValueError):
        context(schema_version=2)


def test_scalar_pruning_payload_and_pareto_rejection():
    controller = Mock()
    controller.submit.return_value = {"run_id": "scalar"}
    adapter = OptimizationAdapter(controller, {"optuna": {
        "pruner": "MedianPruner", "min_completed_requests": 64}})
    intent = NewStudyIntent(context=context(objectives=["output_throughput"]),
                            frameworks=["vllm"], idempotency_key="scalar")
    adapter.submit(intent)
    payload = controller.submit.call_args.args[0]
    assert payload["optuna"]["objectives"] == ["output_throughput"]
    assert payload["optuna"]["min_completed_requests"] == 64
    with pytest.raises(ValueError, match="unconstrained scalar"):
        adapter.submit(intent.model_copy(update={"context": context()}))
