"""NanoClaw stdio MCP facade for the restricted recommendation adapter.

This module deliberately contains no recommendation logic.  It turns the
allowlisted operations already enforced by :mod:`openclaw_adapter` into MCP
tools, which NanoClaw can start inside its isolated agent container.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from hf_model_watch.conversation import (
    ConversationAdapter, ContextInputs, WorkloadInputs, ActiveAnswersInputs, EvidenceInputs,
)

from hf_model_watch.openclaw_adapter import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TOOL_OUTPUT_BYTES,
    TOOL_MODELS,
    OpenClawToolAdapter,
    RecommendationInputs,
    OptimizationStartInputs,
    dispatch,
)

SERVER_NAME = "hf-model-watch"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2025-03-26"

CONVERSATION_TOOLS = {
    "get_conversation_context": ContextInputs,
    "update_workload": WorkloadInputs,
    "start_recommendation": RecommendationInputs,
    "provide_active_input": ActiveAnswersInputs,
    "get_active_model_evidence": EvidenceInputs,
    "get_optimization_connection": ContextInputs,
}
class ApprovedOptimizationInputs(OptimizationStartInputs):
    user_approved: Literal[True]
    approved_gpu_type: str = Field(pattern=r"^gpu-")
    approved_gpu_hourly_cost_usd: float = Field(gt=0)
    max_cost_usd: float = Field(gt=0)
    approved_timeout_minutes: int = Field(ge=60, le=10080)


MCP_TOOL_MODELS = {**TOOL_MODELS, **CONVERSATION_TOOLS,
                   "start_optimization": ApprovedOptimizationInputs}


def _tool_definition(name: str) -> dict[str, Any]:
    model = MCP_TOOL_MODELS[name]
    return {
        "name": name,
        "description": {
            "lookup_task_size": "Return model-independent heuristic size ranges for a task. Use when size is unspecified; clarify ambiguous matches. Returns no model names and starts no research.",
            "recommend_models": "Discover evidence-backed Hugging Face model recommendations.",
            "provide_recommendation_input": "Answer outstanding clarification fields for a run.",
            "select_model": "Persist the user choice for the exact shown run. Then read get_run_status and ask whether to optimize and benchmark; selection never approves GPU work.",
            "record_model_feedback": "Persist user-reported feedback for the selected model.",
            "revise_recommendation": "Create a bounded, traceable revised recommendation run.",
            "get_run_status": "Retrieve the canonical persisted recommendation run.",
            "start_optimization": "Start an optional optimization study for the selected model.",
            "get_optimization_status": "Read study status and analyze available measurements.",
            "list_prior_studies": "List prior studies and their compatibility; initially empty.",
            "reuse_optimization_study": "Reanalyze stored measurements without running trials.",
            "cancel_optimization": "Cancel the selected model's new optimization study.",
            "get_conversation_context": "Read active run and progress. For a recommended unselected run, give a concise shortlist and end with its selection_prompt verbatim. Never starts research.",
            "get_optimization_connection": "Read-only check of the optimizer connection and execution mode. Never submits or ticks GPU jobs.",
            "update_workload": "Save explicitly supplied workload defaults; null removes an optional remembered value.",
            "start_recommendation": "Start one background recommendation using remembered workload and explicit overrides; returns immediately. Poll get_conversation_context.",
            "provide_active_input": "Answer only the active run's outstanding questions, without requiring the user to supply a run ID. Returns immediately.",
            "get_active_model_evidence": "Read sourced evidence and assessments for one active shortlisted model without new research.",
        }[name],
        "inputSchema": model.model_json_schema(),
    }


class NanoClawMcpServer:
    """Small JSON-RPC dispatcher used by a NanoClaw stdio MCP process."""

    def __init__(self, adapter: OpenClawToolAdapter) -> None:
        self.adapter = adapter

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle one MCP message; notifications intentionally have no reply."""
        message_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str):
            return self._error(message_id, -32600, "JSON-RPC method must be a string")
        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return self._result(
                message_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": "Keep replies conversational and concise. After successful unselected recommendations ask selection_prompt verbatim. After saving selection verify the run, check optimizer connection, and ask whether to optimize and benchmark. Never start GPU work without explicit study approval.",
                },
            )
        if method == "tools/list":
            return self._result(
                message_id, {"tools": [_tool_definition(name) for name in MCP_TOOL_MODELS]}
            )
        if method == "tools/call":
            return self._call_tool(message_id, request.get("params"))
        return self._error(message_id, -32601, f"Unsupported MCP method: {method}")

    def _call_tool(self, message_id: Any, params: Any) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            return self._error(message_id, -32602, "tools/call params must be an object")
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(name, str) or name not in MCP_TOOL_MODELS:
            return self._tool_error(message_id, "Unsupported hf-model-watch operation")
        if not isinstance(arguments, Mapping):
            return self._tool_error(message_id, "Tool arguments must be an object")
        try:
            if name == "start_optimization":
                approved = ApprovedOptimizationInputs.model_validate(arguments)
                self.adapter.validate_optimization_approval(approved)
                arguments = approved.model_dump(exclude={"user_approved", "max_cost_usd", "approved_timeout_minutes", "approved_gpu_type", "approved_gpu_hourly_cost_usd"})
            if name in CONVERSATION_TOOLS:
                validated = CONVERSATION_TOOLS[name].model_validate(arguments)
                result = getattr(self.adapter, name)(validated.model_dump(exclude_unset=True))
            else:
                result = dispatch(self.adapter, {"operation": name, "arguments": arguments})
            encoded = json.dumps(result, default=str, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES:
                raise RuntimeError("hf-model-watch response exceeded the tool output limit")
            return self._result(message_id, {"content": [{"type": "text", "text": encoded}]})
        except (ValueError, FileNotFoundError, ValidationError, RuntimeError, KeyError) as exc:
            return self._tool_error(message_id, str(exc))

    @staticmethod
    def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    @staticmethod
    def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}

    @classmethod
    def _tool_error(cls, message_id: Any, message: str) -> dict[str, Any]:
        return cls._result(
            message_id,
            {"content": [{"type": "text", "text": message}], "isError": True},
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hf-model-watch NanoClaw MCP server")
    parser.add_argument(
        "--state-root",
        default=os.environ.get("HF_MODEL_WATCH_STATE_ROOT", "artifacts/recommendations"),
    )
    parser.add_argument("--project-root", default=os.environ.get("HF_MODEL_WATCH_PROJECT_ROOT"))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def load_optimizer_config(project_root: Path) -> None:
    """Load optional operator-owned defaults from the read-only installed runtime."""
    path = project_root / "config" / "optimizer.json"
    if not path.is_file():
        return
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) != {"database", "execution"}:
        raise ValueError("Optimizer config requires database and execution only")
    if not isinstance(config["database"], str) or not Path(config["database"]).is_absolute():
        raise ValueError("Optimizer database path must be absolute")
    if not isinstance(config["execution"], dict):
        raise ValueError("Optimizer execution defaults must be an object")
    os.environ.setdefault("HF_MODEL_WATCH_OPTIMIZER_DATABASE", config["database"])
    os.environ.setdefault("HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON", json.dumps(config["execution"]))


def main() -> int:
    args = _parse_args()
    if args.project_root:
        load_optimizer_config(Path(args.project_root))
    adapter = ConversationAdapter(
        state_root=Path(args.state_root),
        project_root=Path(args.project_root) if args.project_root else None,
        timeout_seconds=args.timeout,
    )
    server = NanoClawMcpServer(adapter)
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > MAX_TOOL_OUTPUT_BYTES:
            response = NanoClawMcpServer._error(
                None, -32600, "MCP request exceeded the input limit"
            )
        else:
            try:
                request = json.loads(raw_line)
                if not isinstance(request, dict):
                    raise ValueError("JSON-RPC request must be an object")
                response = server.handle(request)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                response = NanoClawMcpServer._error(None, -32700, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response, default=str, separators=(",", ":")) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
