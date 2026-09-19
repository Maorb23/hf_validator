from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hf_model_watch.openclaw_adapter import OpenClawToolAdapter, dispatch
from hf_model_watch.recommender.run_state import add_feedback, load_run, run_dir, save_run


class FakeCli:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[list[str]] = []
        self.counter = 0

    def __call__(self, arguments, _timeout):
        args = list(arguments)
        self.calls.append(args)
        command = args[0]
        if command == "recommend" and "--run-id" not in args and "--parent-run-id" not in args:
            self.counter += 1
            run_id = f"run_fake{self.counter}"
            request = args[1]
            if "--gpu-memory-gb" not in args:
                state = {
                    "status": "needs_input",
                    "run_id": run_id,
                    "task_request": request,
                    "needs_input": ["gpu_memory_gb"],
                    "questions": [{"field": "gpu_memory_gb", "type": "number"}],
                    "recommendations": [],
                    "action": "Optimize Inference + Benchmark",
                    "action_status": "placeholder",
                }
            else:
                state = self._recommended(run_id, request)
            save_run(run_dir(self.root, run_id), state)
            return state
        if command == "recommend" and "--run-id" in args:
            run_id = args[args.index("--run-id") + 1]
            _, old = load_run(self.root, run_id)
            state = self._recommended(run_id, old["task_request"])
            save_run(run_dir(self.root, run_id), state)
            return state
        if command == "recommend" and "--parent-run-id" in args:
            self.counter += 1
            run_id = f"run_fake{self.counter}"
            parent = args[args.index("--parent-run-id") + 1]
            state = self._recommended(run_id, args[1])
            state["parent_run_id"] = parent
            state["revision_reason"] = args[args.index("--revision-reason") + 1]
            save_run(run_dir(self.root, run_id), state)
            return state
        if command == "select":
            run_id = args[args.index("--run-id") + 1]
            model_id = args[args.index("--model-id") + 1]
            directory, state = load_run(self.root, run_id)
            state["selected_model_id"] = model_id
            state["selection"] = {"status": "selected", "run_id": run_id, "model_id": model_id}
            save_run(directory, state)
            return state["selection"]
        if command == "feedback":
            run_id = args[args.index("--run-id") + 1]
            directory, state = load_run(self.root, run_id)
            return add_feedback(
                directory,
                state,
                model_id=args[args.index("--model-id") + 1],
                category=args[args.index("--category") + 1],
                details=args[args.index("--details") + 1],
                source="user",
                context=json.loads(args[args.index("--context-json") + 1]),
            )
        raise AssertionError(args)

    @staticmethod
    def _recommended(run_id: str, request: str):
        return {
            "status": "recommended",
            "run_id": run_id,
            "task_request": request,
            "recommendations": [
                {"model_id": "org/model", "evidence": {"status": "verified"}},
            ],
            "selected_model_id": None,
            "action": "Optimize Inference + Benchmark",
            "action_status": "placeholder",
        }


@pytest.fixture
def adapter(tmp_path):
    cli = FakeCli(tmp_path)
    return OpenClawToolAdapter(state_root=tmp_path, cli_runner=cli), cli


def test_recommendation_arguments_are_typed_and_not_shell_interpreted(adapter):
    tool, cli = adapter
    request = "coding model; touch /tmp/should-not-exist && $(id)"
    result = tool.recommend_models({"request": request, "gpu_memory_gb": 80})
    assert result["status"] == "recommended"
    assert cli.calls[0][1] == request
    assert cli.calls[0][0] == "recommend"
    with pytest.raises(ValidationError):
        tool.recommend_models({"request": "x", "gpu_memory_gb": -1})
    with pytest.raises(ValidationError):
        tool.recommend_models({"request": "x", "shell": "rm -rf /"})


def test_clarification_resumes_same_run_and_rejects_unrequested_fields(adapter):
    tool, _ = adapter
    initial = tool.recommend_models({"request": "find a coding model"})
    with pytest.raises(ValueError, match="not requested"):
        tool.provide_recommendation_input(
            {"run_id": initial["run_id"], "answers": {"license": "mit"}}
        )
    result = tool.provide_recommendation_input(
        {"run_id": initial["run_id"], "answers": {"gpu_memory_gb": 80}}
    )
    assert result["run_id"] == initial["run_id"]
    assert result["status"] == "recommended"


def test_selection_requires_explicit_confirmation_and_feedback_targets_selection(adapter):
    tool, _ = adapter
    run = tool.recommend_models({"request": "coding", "gpu_memory_gb": 80})
    with pytest.raises(ValidationError):
        tool.select_model({"run_id": run["run_id"], "model_id": "org/model"})
    selected = tool.select_model(
        {"run_id": run["run_id"], "model_id": "org/model", "confirmed": True}
    )
    assert selected["status"] == "selected"
    with pytest.raises(ValueError, match="selected model"):
        tool.record_model_feedback(
            {
                "run_id": run["run_id"], "model_id": "org/other",
                "category": "quality", "details": "wrong answer",
            }
        )
    feedback = tool.record_model_feedback(
        {
            "run_id": run["run_id"], "model_id": "org/model",
            "category": "out_of_memory", "details": "OOM at 32k",
            "context": {"gpu": "A100", "memory_gb": 80},
        }
    )
    assert feedback["feedback_status"] == "reported"
    assert feedback["context"]["gpu"] == "A100"


def test_revision_and_retrieval_preserve_canonical_history(adapter):
    tool, _ = adapter
    parent = tool.recommend_models({"request": "coding", "gpu_memory_gb": 80})
    child = tool.revise_recommendation(
        {
            "parent_run_id": parent["run_id"],
            "request": "a smaller coding model",
            "gpu_memory_gb": 40,
            "revision_reason": "fit a smaller GPU",
        }
    )
    assert child["parent_run_id"] == parent["run_id"]
    restored = tool.get_run_status({"run_id": child["run_id"]})
    assert restored["task_request"] == "a smaller coding model"
    with pytest.raises(ValidationError, match="optimization"):
        tool.revise_recommendation(
            {
                "parent_run_id": parent["run_id"], "request": "x",
                "revision_reason": "run optimization now",
            }
        )


def test_status_is_bounded_and_does_not_expose_artifact_paths(tmp_path):
    recommendations = [{"model_id": f"org/model-{n}"} for n in range(5)]
    save_run(
        run_dir(tmp_path, "run_many"),
        {
            "status": "recommended", "run_id": "run_many",
            "recommendations": recommendations, "artifact_path": "/secret/path",
        },
    )
    tool = OpenClawToolAdapter(state_root=tmp_path, cli_runner=lambda *_: {})
    result = tool.get_run_status({"run_id": "run_many"})
    assert len(result["recommendations"]) == 3
    assert result["recommendation_count"] == 5
    assert "artifact_path" not in result


def test_invalid_and_unauthorized_operations_are_rejected(adapter):
    tool, _ = adapter
    with pytest.raises(ValueError, match="Unsupported"):
        dispatch(tool, {"operation": "run_optimization", "arguments": {}})
    with pytest.raises(ValidationError):
        tool.get_run_status({"run_id": "../../etc/passwd"})
    with pytest.raises(FileNotFoundError):
        tool.get_run_status({"run_id": "run_not_owned_by_local_state"})


def test_insufficient_evidence_cannot_be_selected(tmp_path):
    save_run(
        run_dir(tmp_path, "run_empty"),
        {
            "status": "insufficient_evidence",
            "run_id": "run_empty",
            "recommendations": [],
            "action_status": "placeholder",
        },
    )
    tool = OpenClawToolAdapter(state_root=tmp_path, cli_runner=lambda *_: pytest.fail())
    status = tool.get_run_status({"run_id": "run_empty"})
    assert status["recommendations"] == []
    assert status["optimization_available"] is False
    with pytest.raises(ValueError, match="no selectable"):
        tool.select_model(
            {"run_id": "run_empty", "model_id": "org/guessed", "confirmed": True}
        )


@pytest.mark.parametrize("message", ["malformed JSON", "timed out after 1 seconds"])
def test_cli_protocol_failures_are_bounded_errors(tmp_path, message):
    def failing_runner(_args, _timeout):
        raise RuntimeError(message)

    tool = OpenClawToolAdapter(state_root=tmp_path, cli_runner=failing_runner)
    with pytest.raises(RuntimeError, match=message):
        tool.recommend_models({"request": "coding"})


def test_end_to_end_conversational_tool_sequence_survives_restart(tmp_path):
    cli = FakeCli(tmp_path)
    first_session = OpenClawToolAdapter(state_root=tmp_path, cli_runner=cli)
    initial = dispatch(
        first_session,
        {"operation": "recommend_models", "arguments": {"request": "coding assistant"}},
    )
    recommended = dispatch(
        first_session,
        {
            "operation": "provide_recommendation_input",
            "arguments": {"run_id": initial["run_id"], "answers": {"gpu_memory_gb": 80}},
        },
    )
    dispatch(
        first_session,
        {
            "operation": "select_model",
            "arguments": {
                "run_id": recommended["run_id"], "model_id": "org/model", "confirmed": True,
            },
        },
    )
    feedback = dispatch(
        first_session,
        {
            "operation": "record_model_feedback",
            "arguments": {
                "run_id": recommended["run_id"], "model_id": "org/model",
                "category": "quality", "details": "weak code review",
            },
        },
    )
    restarted = OpenClawToolAdapter(state_root=tmp_path, cli_runner=cli)
    restored = restarted.get_run_status({"run_id": recommended["run_id"]})
    assert restored["selected_model_id"] == "org/model"
    revised = restarted.revise_recommendation(
        {
            "parent_run_id": recommended["run_id"], "request": "stronger code review",
            "gpu_memory_gb": 80, "revision_reason": "reported quality issue",
            "trigger_feedback_id": feedback["feedback_id"],
        }
    )
    assert revised["parent_run_id"] == recommended["run_id"]
