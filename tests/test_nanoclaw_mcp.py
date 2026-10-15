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
