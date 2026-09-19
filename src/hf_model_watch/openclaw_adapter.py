from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from hf_model_watch.recommender.run_state import (
    MAX_FEEDBACK_CHARS,
    MAX_REQUEST_CHARS,
    load_feedback,
    load_run,
    validate_run_id,
)

MAX_TOOL_OUTPUT_BYTES = 2_000_000
DEFAULT_TIMEOUT_SECONDS = 300
MAX_REVISION_DEPTH = 5
ALLOWED_FEEDBACK_CATEGORIES = {
    "out_of_memory",
    "latency",
    "throughput",
    "quality",
    "capability",
    "other",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RecommendationInputs(StrictModel):
    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    gpu_memory_gb: float | None = Field(default=None, gt=0)
    gpu_count: int = Field(default=1, ge=1, le=64)
    budget_usd_per_hour: float | None = Field(default=None, gt=0)
    gpu_hourly_cost_usd: float | None = Field(default=None, gt=0)
    license: str | None = Field(default=None, min_length=1, max_length=100)
    framework_mode: Literal["fast", "sweep"] | None = None


class ClarificationInputs(StrictModel):
    run_id: str
    answers: dict[str, str | int | float]

    @field_validator("run_id")
    @classmethod
    def valid_run_id(cls, value: str) -> str:
        return validate_run_id(value)

    @field_validator("answers")
    @classmethod
    def has_answers(cls, value: dict[str, str | int | float]) -> dict[str, str | int | float]:
        if not value:
            raise ValueError("At least one clarification answer is required")
        return value


class SelectionInputs(StrictModel):
    run_id: str
    model_id: str = Field(min_length=1, max_length=500)
    confirmed: Literal[True]

    @field_validator("run_id")
    @classmethod
    def valid_run_id(cls, value: str) -> str:
        return validate_run_id(value)


class FeedbackInputs(StrictModel):
    run_id: str
    model_id: str = Field(min_length=1, max_length=500)
    category: Literal[
        "out_of_memory", "latency", "throughput", "quality", "capability", "other"
    ]
    details: str = Field(min_length=1, max_length=MAX_FEEDBACK_CHARS)
    context: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("run_id")
    @classmethod
    def valid_run_id(cls, value: str) -> str:
        return validate_run_id(value)


class RevisionInputs(RecommendationInputs):
    parent_run_id: str
    revision_reason: str = Field(min_length=1, max_length=1000)
    trigger_feedback_id: str | None = None

    @field_validator("parent_run_id")
    @classmethod
    def valid_parent_run_id(cls, value: str) -> str:
        return validate_run_id(value)

    @model_validator(mode="after")
    def no_optimization_request(self) -> RevisionInputs:
        if "run optimization" in self.revision_reason.lower():
            raise ValueError("Inference optimization is unavailable in Phase 2")
        return self


class RunStatusInputs(StrictModel):
    run_id: str

    @field_validator("run_id")
    @classmethod
    def valid_run_id(cls, value: str) -> str:
        return validate_run_id(value)


CliRunner = Callable[[Sequence[str], float], dict[str, Any]]


def _default_cli_runner(project_root: Path) -> CliRunner:
    allowed_env_names = {
        "PATH",
        "PYTHONPATH",
        "HF_HOME",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "NEBIUS_API_KEY",
        "TAVILY_API_KEY",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
    }

    def run(arguments: Sequence[str], timeout: float) -> dict[str, Any]:
        env = {name: value for name, value in os.environ.items() if name in allowed_env_names}
        source_root = str(project_root / "src")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "hf_model_watch", *arguments],
                cwd=project_root,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"hf-model-watch timed out after {timeout:g} seconds") from exc
        if len(completed.stdout) > MAX_TOOL_OUTPUT_BYTES:
            raise RuntimeError("hf-model-watch response exceeded the tool output limit")
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("hf-model-watch returned malformed JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("hf-model-watch returned a non-object response")
        if completed.returncode != 0 or payload.get("status") == "error":
            message = payload.get("error") or "hf-model-watch operation failed"
            raise ValueError(str(message))
        return payload

    return run


class OpenClawToolAdapter:
    """Restricted OpenClaw-facing facade over the authoritative CLI."""

    def __init__(
        self,
        *,
        state_root: Path,
        project_root: Path | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cli_runner: CliRunner | None = None,
    ) -> None:
        self.state_root = state_root.resolve()
        self.project_root = (project_root or Path.cwd()).resolve()
        self.timeout_seconds = timeout_seconds
        self._run_cli = cli_runner or _default_cli_runner(self.project_root)

    def recommend_models(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = RecommendationInputs.model_validate(raw)
        return self._invoke_recommend(inputs)

    def provide_recommendation_input(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = ClarificationInputs.model_validate(raw)
        _, run = load_run(self.state_root, inputs.run_id)
        if run.get("status") != "needs_input":
            raise ValueError("Only a run awaiting clarification can accept input")
        outstanding = set(run.get("needs_input") or [])
        unexpected = set(inputs.answers) - outstanding
        if unexpected:
            raise ValueError(
                f"Answers contain fields not requested by the run: {sorted(unexpected)}"
            )
        arguments = ["recommend", "--run-id", inputs.run_id]
        arguments.extend(self._answer_arguments(inputs.answers))
        return self._invoke(arguments)

    def select_model(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = SelectionInputs.model_validate(raw)
        _, run = load_run(self.state_root, inputs.run_id)
        if run.get("status") != "recommended":
            raise ValueError("Run has no selectable recommendations")
        candidates = {row.get("model_id") for row in run.get("recommendations", [])}
        if inputs.model_id not in candidates:
            raise ValueError("Selected model does not belong to this recommendation run")
        return self._invoke(
            ["select", "--run-id", inputs.run_id, "--model-id", inputs.model_id]
        )

    def record_model_feedback(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = FeedbackInputs.model_validate(raw)
        _, run = load_run(self.state_root, inputs.run_id)
        if run.get("selected_model_id") != inputs.model_id:
            raise ValueError("Feedback must target the explicitly selected model for this run")
        return self._invoke(
            [
                "feedback",
                "--run-id",
                inputs.run_id,
                "--model-id",
                inputs.model_id,
                "--category",
                inputs.category,
                "--details",
                inputs.details,
                "--context-json",
                json.dumps(inputs.context, separators=(",", ":")),
            ]
        )

    def revise_recommendation(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = RevisionInputs.model_validate(raw)
        _, parent = load_run(self.state_root, inputs.parent_run_id)
        if self._revision_depth(parent) >= MAX_REVISION_DEPTH:
            raise ValueError(f"Revision chain is limited to {MAX_REVISION_DEPTH} revisions")
        if inputs.trigger_feedback_id:
            feedback = load_feedback(
                self.state_root / inputs.parent_run_id, inputs.trigger_feedback_id
            )
            if feedback.get("run_id") != inputs.parent_run_id:
                raise ValueError("Feedback does not belong to the parent run")
        arguments = self._recommend_arguments(inputs)
        arguments.extend(
            [
                "--parent-run-id",
                inputs.parent_run_id,
                "--revision-reason",
                inputs.revision_reason,
            ]
        )
        if inputs.trigger_feedback_id:
            arguments.extend(["--trigger-feedback-id", inputs.trigger_feedback_id])
        return self._invoke(arguments)

    def get_run_status(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = RunStatusInputs.model_validate(raw)
        _, run = load_run(self.state_root, inputs.run_id)
        return self._safe_run_view(run)

    def _invoke_recommend(self, inputs: RecommendationInputs) -> dict[str, Any]:
        return self._invoke(self._recommend_arguments(inputs))

    def _invoke(self, arguments: list[str]) -> dict[str, Any]:
        payload = self._run_cli(
            [*arguments, "--state-root", str(self.state_root), "--json"],
            self.timeout_seconds,
        )
        return self._safe_run_view(payload)

    @staticmethod
    def _recommend_arguments(inputs: RecommendationInputs) -> list[str]:
        arguments = ["recommend", inputs.request, "--gpu-count", str(inputs.gpu_count)]
        fields = {
            "gpu_memory_gb": "--gpu-memory-gb",
            "budget_usd_per_hour": "--budget-usd-per-hour",
            "gpu_hourly_cost_usd": "--gpu-hourly-cost-usd",
            "license": "--license",
            "framework_mode": "--framework-mode",
        }
        for field, option in fields.items():
            value = getattr(inputs, field)
            if value is not None:
                arguments.extend([option, str(value)])
        return arguments

    @staticmethod
    def _answer_arguments(answers: Mapping[str, str | int | float]) -> list[str]:
        options = {
            "request": None,
            "gpu_memory_gb": "--gpu-memory-gb",
            "budget_usd_per_hour": "--budget-usd-per-hour",
            "gpu_hourly_cost_usd": "--gpu-hourly-cost-usd",
            "license": "--license",
            "framework_mode": "--framework-mode",
        }
        arguments: list[str] = []
        request = answers.get("request")
        if request is not None:
            arguments.append(str(request))
        for field, value in answers.items():
            if field not in options:
                raise ValueError(f"Unsupported clarification field: {field}")
            option = options[field]
            if option:
                arguments.extend([option, str(value)])
        return arguments

    def _revision_depth(self, run: Mapping[str, Any]) -> int:
        depth = 0
        current = run
        seen: set[str] = set()
        while current.get("parent_run_id"):
            parent_id = validate_run_id(str(current["parent_run_id"]))
            if parent_id in seen:
                raise ValueError("Recommendation revision history contains a cycle")
            seen.add(parent_id)
            _, current = load_run(self.state_root, parent_id)
            depth += 1
        return depth

    @staticmethod
    def _safe_run_view(run: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "status",
            "run_id",
            "parent_run_id",
            "task_request",
            "constraints",
            "recommendations",
            "selected_model_id",
            "selection",
            "needs_input",
            "questions",
            "action",
            "action_status",
            "revision_reason",
            "trigger_feedback_id",
            "feedback_id",
            "model_id",
            "category",
            "details",
            "context",
            "feedback_status",
            "created_at",
            "updated_at",
        }
        result = {key: value for key, value in run.items() if key in allowed}
        recommendations = result.get("recommendations")
        if isinstance(recommendations, list):
            result["recommendations"] = recommendations[:3]
            result["recommendation_count"] = len(recommendations)
        # The optimization action remains visible for compatibility but cannot be invoked.
        if result.get("action_status") == "placeholder":
            result["optimization_available"] = False
        return result


TOOL_MODELS: dict[str, type[StrictModel]] = {
    "recommend_models": RecommendationInputs,
    "provide_recommendation_input": ClarificationInputs,
    "select_model": SelectionInputs,
    "record_model_feedback": FeedbackInputs,
    "revise_recommendation": RevisionInputs,
    "get_run_status": RunStatusInputs,
}


def dispatch(adapter: OpenClawToolAdapter, request: Mapping[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    arguments = request.get("arguments", {})
    if operation not in TOOL_MODELS:
        raise ValueError("Unsupported OpenClaw tool operation")
    if not isinstance(arguments, Mapping):
        raise ValueError("Tool arguments must be an object")
    method = getattr(adapter, str(operation))
    return method(arguments)


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_TOOL_OUTPUT_BYTES + 1)
        if len(raw) > MAX_TOOL_OUTPUT_BYTES:
            raise ValueError("Tool request exceeded the input limit")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("Tool request must be a JSON object")
        state_root = Path(os.environ.get("HF_MODEL_WATCH_STATE_ROOT", "artifacts/recommendations"))
        adapter = OpenClawToolAdapter(state_root=state_root)
        result = dispatch(adapter, request)
        sys.stdout.write(json.dumps({"ok": True, "result": result}, default=str))
        return 0
    except (ValueError, FileNotFoundError, json.JSONDecodeError, ValidationError) as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    except RuntimeError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
