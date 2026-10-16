from __future__ import annotations

import json
from pathlib import Path

from hf_model_watch.nanoclaw_mcp import NanoClawMcpServer, PROTOCOL_VERSION
from hf_model_watch.openclaw_adapter import OpenClawToolAdapter
from hf_model_watch.recommender.run_state import run_dir, save_run


def _recommended_run(root: Path, run_id: str = "run_mcp") -> dict:
    run = {
        "status": "recommended",
        "run_id": run_id,
        "task_request": "coding model",
        "recommendations": [{"model_id": "org/model", "evidence": {"status": "verified"}}],
        "selected_model_id": None,
        "action": "Optimize Inference + Benchmark",
        "action_status": "placeholder",
    }
    save_run(run_dir(root, run_id), run)
    return run


def test_mcp_initialization_and_tool_inventory_are_restricted(tmp_path):
    server = NanoClawMcpServer(OpenClawToolAdapter(state_root=tmp_path, cli_runner=lambda *_: {}))
    initialized = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert initialized["result"]["protocolVersion"] == PROTOCOL_VERSION
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert names == {
        "lookup_task_size",
        "recommend_models",
        "provide_recommendation_input",
        "select_model",
        "record_model_feedback",
        "revise_recommendation",
        "get_run_status",
        "start_optimization",
        "get_optimization_status",
        "list_prior_studies",
        "reuse_optimization_study",
        "cancel_optimization",
        "get_conversation_context",
        "update_workload",
        "start_recommendation",
        "provide_active_input",
        "get_active_model_evidence",
        "get_optimization_connection",
    }
    assert "run_optimization" not in names
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_mcp_tool_call_uses_adapter_validation_and_bounded_safe_view(tmp_path):
    _recommended_run(tmp_path)
    server = NanoClawMcpServer(OpenClawToolAdapter(state_root=tmp_path, cli_runner=lambda *_: {}))
    result = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_run_status", "arguments": {"run_id": "run_mcp"}},
        }
    )
    payload = json.loads(result["result"]["content"][0]["text"])
    assert payload["selected_model_id"] is None
    assert payload["optimization_available"] is False

    invalid = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "run_optimization", "arguments": {}},
        }
    )
    assert invalid["result"]["isError"] is True

    untrusted = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "get_run_status", "arguments": {"run_id": "../../etc/passwd"}},
        }
    )
    assert untrusted["result"]["isError"] is True


def test_recommend_select_handoff_uses_saved_run(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    from hf_model_watch.conversation import ConversationAdapter, write_json
    from hf_model_watch.recommender import discovery

    monkeypatch.delenv("HF_MODEL_WATCH_OPTIMIZER_DATABASE", raising=False)
    monkeypatch.setattr(discovery, "discover_models", lambda *args, **kwargs: {
        "status": "recommended", "task_request": "coding model",
        "recommendations": [
            {"model_id": "org/alpha", "revision": "a" * 40},
            {"model_id": "org/beta", "revision": "b" * 40},
            {"model_id": "other/alpha", "revision": "c" * 40},
        ], "selected_model_id": None,
    })

    def cli(arguments, timeout):
        result = CliRunner().invoke(app, list(arguments))
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    adapter = ConversationAdapter(state_root=tmp_path, cli_runner=cli)
    server = NanoClawMcpServer(adapter)

    def call(name, **arguments):
        return server.handle({"id": 1, "method": "tools/call", "params": {
            "name": name, "arguments": arguments,
        }})["result"]

    def payload(name, **arguments):
        result = call(name, **arguments)
        assert not result.get("isError"), result
        return json.loads(result["content"][0]["text"])

    recommended = payload("recommend_models", request="coding model", gpu_memory_gb=80,
                          gpu_count=2, budget_usd_per_hour=8,
                          gpu_hourly_cost_usd=3, license="any", framework_mode="sweep",
                          speculative_decoding="off")
    run_id = recommended["run_id"]
    assert recommended["selection_prompt"] == "Which model would you like to select?"
    assert recommended["selection_choices"][1] == {"number": 2, "model_id": "org/beta"}
    for choice in ("alpha", "0", "4", "1 or 2", "unknown"):
        assert call("select_model", run_id=run_id, model_id=choice, confirmed=True)["isError"]
    assert call("select_model", run_id=run_id, model_id="2", confirmed=False)["isError"]
    assert payload("get_run_status", run_id=run_id)["selected_model_id"] is None
    # A unique short name and a numbered answer resolve to the same persisted choice.
    selected = payload("select_model", run_id=run_id, model_id="beta", confirmed=True)
    assert selected["model_id"] == "org/beta"
    payload("select_model", run_id=run_id, model_id="2", confirmed=True)
    # Newer remembered defaults must never leak into the selected run's handoff.
    write_json(adapter.context_path, {"active_run_id": run_id, "workload": {"gpu_count": 8}})
    saved = payload("get_run_status", run_id=run_id)
    assert "selection_prompt" not in saved
    handoff = saved["optimization_handoff"]
    assert handoff["run_id"] == run_id
    assert handoff["model_id"] == "org/beta"
    assert handoff["revision"] == "b" * 40
    assert handoff["workload"] == "coding model"
    assert handoff["gpu_count"] == 2
    assert handoff["gpu_memory_gb"] == 80
    assert handoff["budget_usd_per_hour"] == 8
    assert handoff["gpu_hourly_cost_usd"] == 3
    assert handoff["missing_saved_fields"] == []
    assert handoff["controller_status"] == "not_connected"
    assert handoff["execution_ready"] is False
    assert handoff["requires_explicit_approval"] is True
    assert "optimization_reference" not in saved
    assert payload("get_optimization_status", run_id=run_id)["status"] == "not_requested"
    # A different run cannot accept a model that only belonged to the old shortlist.
    _recommended_run(tmp_path, "run_other")
    assert call("select_model", run_id="run_other", model_id="org/beta", confirmed=True)["isError"]


def test_handoff_reports_missing_fields_and_does_not_claim_connection(tmp_path, monkeypatch):
    run = _recommended_run(tmp_path)
    run["selected_model_id"] = "org/model"
    save_run(run_dir(tmp_path, run["run_id"]), run)
    monkeypatch.setenv("HF_MODEL_WATCH_OPTIMIZER_DATABASE", "/missing/controller.db")
    adapter = OpenClawToolAdapter(state_root=tmp_path, cli_runner=lambda *_: {})
    handoff = adapter.get_run_status({"run_id": run["run_id"]})["optimization_handoff"]
    assert "revision" in handoff["missing_saved_fields"]
    assert "budget_usd_per_hour" in handoff["missing_saved_fields"]
    assert handoff["controller_status"] == "configured_unverified"
    assert handoff["execution_ready"] is False


def test_optimization_submission_requires_approval_at_mcp_boundary(tmp_path):
    calls = []

    class ControllerFixture:
        def start(self, run_id, intent):
            calls.append((run_id, intent))
            return {"status": "queued"}

    adapter = OpenClawToolAdapter(state_root=tmp_path, cli_runner=lambda *_: {},
                                 optimization_workflow=ControllerFixture())
    # Approval forwarding is isolated here; spending checks have separate tests.
    adapter.validate_optimization_approval = lambda approved: None
    server = NanoClawMcpServer(adapter)
    arguments = {"run_id": "run_mcp", "max_cost_usd": 5, "approved_timeout_minutes": 60,
                 "approved_gpu_type": "gpu-h100-sxm", "approved_gpu_hourly_cost_usd": 3, "intent": {
        "idempotency_key": "approved-study", "frameworks": ["vllm"],
        "context": {"model_id": "org/model", "revision": "a" * 40,
                    "gpu_type": "gpu-h100-sxm", "gpu_count": 1,
                    "workload": {"input_tokens": 128, "output_tokens": 32}, "concurrency": 1},
    }}
    for approval in ({}, {"user_approved": False}):
        response = server.handle({"id": 1, "method": "tools/call", "params": {
            "name": "start_optimization", "arguments": {**arguments, **approval},
        }})
        assert response["result"]["isError"]
    assert calls == []
    response = server.handle({"id": 1, "method": "tools/call", "params": {
        "name": "start_optimization", "arguments": {**arguments, "user_approved": True},
    }})
    assert not response["result"].get("isError")
    assert len(calls) == 1
    assert calls[0][0] == "run_mcp"


def test_nanoclaw_spending_gate_checks_revision_hardware_and_cost(tmp_path, monkeypatch):
    import pytest
    from hf_model_watch.nanoclaw_mcp import ApprovedOptimizationInputs
    run = _recommended_run(tmp_path)
    run.update(selected_model_id="org/model", selection={"revision": "a"*40}, request_spec={
        "gpu_count": 2, "gpu_hourly_cost_usd": 1, "budget_usd_per_hour": 2.5})
    save_run(run_dir(tmp_path, run["run_id"]), run)
    monkeypatch.setenv("HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON", '{"timeout_minutes":60}')
    adapter = OpenClawToolAdapter(state_root=tmp_path, cli_runner=lambda *_: {})
    adapter.get_optimization_connection = lambda _: {"connected": True}
    values = {"run_id": run["run_id"], "user_approved": True, "max_cost_usd": 2,
              "approved_timeout_minutes": 60, "approved_gpu_type": "gpu-l40s-d",
              "approved_gpu_hourly_cost_usd": 1, "intent": {
                  "frameworks": ["vllm"], "idempotency_key": "budget-test", "context": {
                      "model_id": "org/model", "revision": "a"*40, "gpu_count":2,
                      "gpu_type":"gpu-l40s-d", "workload":{"input_tokens":128,"output_tokens":32},
                      "concurrency":1}}}
    adapter.validate_optimization_approval(ApprovedOptimizationInputs.model_validate(values))
    with pytest.raises(ValueError, match="study budget"):
        adapter.validate_optimization_approval(ApprovedOptimizationInputs.model_validate({**values, "max_cost_usd":1}))
    with pytest.raises(ValueError, match="exact optimizer GPU"):
        adapter.validate_optimization_approval(ApprovedOptimizationInputs.model_validate({**values, "approved_gpu_type":"gpu-h100-sxm"}))
    values["intent"]["frameworks"].append("sglang")
    with pytest.raises(ValueError, match="concurrent framework"):
        adapter.validate_optimization_approval(ApprovedOptimizationInputs.model_validate(values))


def test_operator_optimizer_config_preserves_explicit_environment(tmp_path, monkeypatch):
    from hf_model_watch.nanoclaw_mcp import load_optimizer_config
    monkeypatch.delenv("HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON", raising=False)
    monkeypatch.setenv("HF_MODEL_WATCH_OPTIMIZER_DATABASE", "/explicit/controller.sqlite")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/optimizer.json").write_text(json.dumps({
        "database": "/mounted/controller.sqlite", "execution": {"dry_run": True, "timeout_minutes": 60}}))
    load_optimizer_config(tmp_path)
    import os
    assert os.environ["HF_MODEL_WATCH_OPTIMIZER_DATABASE"] == "/explicit/controller.sqlite"
    assert json.loads(os.environ["HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON"])["dry_run"]
