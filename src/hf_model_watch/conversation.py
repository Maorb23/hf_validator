"""Group-local conversation state and detached recommendation jobs.

The agent translates natural language; Python validates and persists its inputs.
No model selection, benchmark, or spending is triggered by remembering a workload.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from pydantic import Field

from hf_model_watch.openclaw_adapter import (
    OpenClawToolAdapter, RecommendationInputs, ClarificationInputs, SelectionInputs, StrictModel, _default_cli_runner,
)
from hf_model_watch.optimization.contracts import HFDataset
from hf_model_watch.recommender.run_state import load_run, locked_run, save_run
from hf_model_watch.recommender.utilities import extract_constraints, speculative_preference


class ContextInputs(StrictModel):
    pass


class WorkloadInputs(RecommendationInputs):
    request: str = "Update remembered workload"
    gpu_type: str | None = Field(default=None, pattern=r"^gpu-")
    input_tokens: int | None = Field(default=None, gt=0)
    output_tokens: int | None = Field(default=None, gt=0)
    concurrency: int | None = Field(default=None, gt=0)
    num_prompts: int | None = Field(default=None, gt=0)
    dataset: HFDataset | None = None
    seed: int | None = None
    frameworks: list[str] | None = None


class ActiveAnswersInputs(StrictModel):
    answers: dict[str, str | int | float]


class EvidenceInputs(StrictModel):
    model_id: str
    run_id: str | None = None


WORKLOAD_FIELDS = set(RecommendationInputs.model_fields) - {"request"}


def process_identity(pid: int) -> str:
    """Container PID reuse must not revive an interrupted job after restart."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        raise ProcessLookupError(pid)
    return fields[19]  # Linux process start time (field 22).


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".conversation-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, default=str)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ConversationAdapter(OpenClawToolAdapter):
    """One active recommendation and workload profile per group state root."""

    @staticmethod
    def _safe_run_view(run: Mapping[str, Any]) -> dict[str, Any]:
        result = OpenClawToolAdapter._safe_run_view(run)
        if result.get("action_status") == "placeholder":
            result["action"] = "Optional optimization after model selection"
            result["action_status"] = "not_requested"
            result.pop("optimization_available", None)
        if result.get("optimization_handoff"):
            from hf_model_watch.optimization.workflow import connection_status
            connection = connection_status()
            result["optimization_handoff"]["controller_status"] = connection["status"]
            result["optimization_handoff"]["execution_mode"] = connection.get("execution_mode")
        if run.get("optimization_preparation"):
            result["optimization_preparation"] = run["optimization_preparation"]
            result.setdefault("optimization_handoff", {})["next_question"] = "Present prepared configuration for approval, or ask only for missing preparation choices."
        # Status must fit in the provider's tool window. Full evidence has its own tool.
        rows = []
        for row in result.get("recommendations", []):
            compact = {key: row[key] for key in (
                "model_id", "revision", "hf_score", "final_score", "eligible", "why_it_fits",
                "limitations", "task_fit") if key in row}
            runtime = row.get("runtime_assessment") or {}
            compact["runtime_assessment"] = {key: runtime[key] for key in (
                "revision", "estimated_params_billion", "eligible", "blocking_reasons",
                "estimated_bf16_memory_gb", "estimated_serving_memory_gb", "available_memory_gb",
                "parameters_billion", "memory_estimate", "memory_estimate_basis", "memory_estimate_components",
                "estimated_hourly_usd", "runtime", "compatible_frameworks", "context_length", "license") if key in runtime}
            pair = runtime.get("speculative_decoding") or {}
            compact["runtime_assessment"]["speculative_decoding"] = {
                key: pair[key] for key in ("method", "draft_model", "runtime", "runtimes", "usable",
                                          "confidence", "source_url", "reason") if key in pair}
            compact["source_urls"] = list(dict.fromkeys(
                source["source_url"] for source in row.get("evidence", []) if source.get("source_url")))[:3]
            if row.get("nemotron"):
                compact["nemotron"] = {key: row["nemotron"].get(key) for key in ("task_fit", "rationale", "uncertainty")}
            rows.append(compact)
        if "recommendations" in result:
            result["recommendations"] = rows
        rejected = result.get("candidate_rejections") or {}
        counts = {}
        for reasons in rejected.values():
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
        result["candidate_rejection_counts"] = counts
        result["candidate_rejection_count"] = len(rejected)
        result["candidate_rejections"] = dict(list(rejected.items())[:8])
        research = result.get("research") or {}
        coverage = research.get("method_coverage") or {}
        result["research"] = {key: value for key, value in research.items() if key != "method_coverage"}
        result["research"]["method_coverage"] = {
            model: {method: {"status": info.get("status"), "reason": info.get("reason")}
                    for method, info in methods.items()} for model, methods in coverage.items()
            if model in {row.get("model_id") for row in rows}}
        if result.get("selection_prompt"):
            result = {"conversation_next_step": result["selection_prompt"], **result}
            result["response_guidance"] = (
                "Give a concise shortlist with a reason and limitation per model. "
                "End with conversation_next_step verbatim. Use get_active_model_evidence "
                "with run_id for detail; do not inspect tool output files with shell commands."
            )
        return result

    def _activate_run(self, run_id: str) -> None:
        _, run = load_run(self.state_root, run_id)
        with self._locked():
            context = self._context()
            job = self._job(context)
            if job and job.get("status") in {"starting", "running"}:
                raise ValueError("A recommendation is still running")
            context.pop("active_job", None)
            context["active_run_id"] = run_id
            context.setdefault("workload", {}).update({
                k: v for k, v in {**run.get("request_spec", {}), **run.get("optimization_choices", {})}.items()
                if k in WORKLOAD_FIELDS and v is not None})
            write_json(self.context_path, context)

    def select_model(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = SelectionInputs.model_validate(raw)
        _, run = load_run(self.state_root, inputs.run_id)
        # Resolve only the displayed shortlist for this run, never another run's order.
        rows = run.get("recommendations", [])[:3]
        choice = inputs.model_id.strip()
        ids = [row["model_id"] for row in rows]
        if choice.isdecimal():
            index = int(choice) - 1
            matches = ids[index:index + 1] if 0 <= index < len(ids) else []
        else:
            matches = [model for model in ids if model.casefold() == choice.casefold()]
            if not matches:
                matches = [model for model in ids if model.rsplit("/", 1)[-1].casefold() == choice.casefold()]
        if len(matches) != 1:
            raise ValueError("Choice is ambiguous or absent from this run's shortlist; ask for the full model ID or a single listed number")
        result = super().select_model({"run_id": inputs.run_id, "model_id": matches[0], "confirmed": True})
        self._activate_run(inputs.run_id)
        result["optimization_preparation"] = self.prepare_optimization({"run_id": inputs.run_id})
        return result

    def prepare_optimization(self, raw: Mapping[str, Any], *, remember_workload: bool = True) -> dict[str, Any]:
        from hf_model_watch.optimization.preparation import PreparationInputs, prepare
        inputs = PreparationInputs.model_validate(raw)
        preferences = {}
        if remember_workload:
            with self._locked():
                preferences = self._context().get("workload", {})
        with locked_run(self.state_root, inputs.run_id) as (directory, run):
            saved = {**preferences, **{k: v for k, v in run.get("request_spec", {}).items() if v is not None}}
            for key, value in saved.items():
                if key in PreparationInputs.model_fields and key != "run_id":
                    run.setdefault("optimization_choices", {}).setdefault(key, value)
            run.setdefault("optimization_choices", {}).setdefault("gpu_count", 1)
            from hf_model_watch.optimization.intake import DEFAULT_GPU_HOURLY_COST_USD
            run["optimization_choices"].setdefault("gpu_hourly_cost_usd", DEFAULT_GPU_HOURLY_COST_USD)
            choices = inputs.model_dump(exclude_none=True, exclude={"run_id", "execution", "dry_run", "trials"})
            if inputs.execution is not None:
                run.setdefault("optimization_execution", {}).update(inputs.execution)
            overrides = run.setdefault("optimization_execution", {})
            if inputs.dry_run is not None:
                overrides["dry_run"] = inputs.dry_run
            run.setdefault("optimization_choices", {}).update(choices)
            try:
                # Older terminal runs persisted numeric answers as strings.
                normalized = PreparationInputs.model_validate({**run["optimization_choices"], "run_id": inputs.run_id})
                run["optimization_choices"] = normalized.model_dump(exclude_none=True, exclude={"run_id", "execution"})
                execution = json.loads(os.environ.get("HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON", "{}"))
                execution.update(run.get("optimization_execution", {}))
                if inputs.trials is not None:
                    execution["optuna"] = {**execution.get("optuna", {}), "trials": inputs.trials}
                    overrides["optuna"] = execution["optuna"]
                    execution["optuna_by_framework"] = {
                        f: {**config, "trials": inputs.trials}
                        for f, config in execution.get("optuna_by_framework", {}).items()
                    }
                    overrides["optuna_by_framework"] = execution["optuna_by_framework"]
                if not run.get("selected_model_id"):
                    result = {"status": "needs_input", "needs_input": ["selected_model_id"],
                              "requires_explicit_approval": True}
                else:
                    result = prepare(run, preferences, execution, directory / "optimization-prepared",
                                     self.project_root / "config/recommender.yaml")
            except Exception as exc:
                # Preserve selection and choices; no submission happens on research/config failure.
                result = {"status": "preparation_failed", "error_type": type(exc).__name__, "reason": str(exc),
                          "requires_explicit_approval": True, "retry_tool": "prepare_optimization"}
            run["optimization_preparation"] = result
            save_run(directory, run)
        if remember_workload:
            self._activate_run(inputs.run_id)
        return result

    def validate_optimization_approval(self, approved: Any) -> None:
        super().validate_optimization_approval(approved)
        path = self.state_root / approved.run_id / "optimization-prepared/request.json"
        if not path.exists():
            raise ValueError("Prepare optimization before approval")
        payload = json.loads(path.read_text())
        configured = json.loads(os.environ.get("HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON", "{}"))
        _, run = load_run(self.state_root, approved.run_id)
        configured.update(run.get("optimization_execution", {}))
        if payload.get("timeout_minutes", 180) != configured.get("timeout_minutes", 180):
            raise ValueError("Execution duration changed; prepare and approve again")

    def start_optimization(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        from hf_model_watch.openclaw_adapter import OptimizationStartInputs
        from hf_model_watch.optimization.contracts import NewStudyIntent
        from hf_model_watch.optimization.workflow import OptimizationWorkflow
        from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter
        from llm_optimizer.remote.contracts import OptimizationRequest
        inputs = OptimizationStartInputs.model_validate(raw)
        _, run = load_run(self.state_root, inputs.run_id)
        prepared = run.get("optimization_preparation") or {}
        if prepared.get("status") != "prepared":
            raise ValueError("Prepare and review the optimization request first")
        directory = self.state_root / inputs.run_id / "optimization-prepared"
        intent = NewStudyIntent.model_validate_json((directory / "intent.json").read_text())
        if inputs.intent != intent:
            raise ValueError("Approval intent differs from prepared intent")
        payload = json.loads((directory / "request.json").read_text())
        OptimizationRequest.parse(payload)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if digest != prepared.get("request_sha256"):
            raise ValueError("Prepared request changed; prepare and approve again")
        adapter = OptimizationAdapter(self._optimizer()._adapter().controller, payload)
        if adapter.build_request(intent) != payload:
            raise ValueError("Prepared request differs from its intent")
        return OptimizationWorkflow(self.state_root, adapter).start(inputs.run_id, intent)

    @property
    def context_path(self) -> Path:
        return self.state_root / "conversation.json"

    @contextmanager
    def _locked(self):
        self.state_root.mkdir(parents=True, exist_ok=True)
        with (self.state_root / ".conversation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _context(self) -> dict[str, Any]:
        return json.loads(self.context_path.read_text()) if self.context_path.exists() else {"workload": {}}

    def _job(self, context: dict[str, Any]) -> dict[str, Any] | None:
        ident = context.get("active_job")
        if not ident:
            return None
        if not isinstance(ident, str) or not ident.startswith("job_") or not ident[4:].isalnum():
            raise ValueError("Invalid active job ID")
        path = self.state_root / "jobs" / (ident + ".json")
        job = json.loads(path.read_text())
        if job["status"] in {"starting", "running"} and job.get("pid"):
            try:
                os.kill(job["pid"], 0)
                if job.get("process_identity") and process_identity(job["pid"]) != job["process_identity"]:
                    raise ProcessLookupError(job["pid"])
            except (ProcessLookupError, FileNotFoundError):
                job.update(status="interrupted", error_type="WorkerStopped")
                write_json(path, job)
        return job

    def get_run_status(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        result = super().get_run_status(raw)
        # Audit records stay on disk; model context gets only authoritative active status.
        reference = result.pop("optimization_reference", None)
        for key in ("optimization_result", "parent_run_id", "revision_reason", "trigger_feedback_id"):
            result.pop(key, None)
        if reference:
            status = self._optimizer()._adapter().status(reference["study_reference"])
            terminal = status.get("state") in {"cancelled", "completed", "failed", "dry_run"}
            result["optimization_status"] = ({"state": "not_active"} if terminal else {
                key: status.get(key) for key in ("run_id", "state", "jobs")})
        return result

    def get_conversation_context(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        ContextInputs.model_validate(raw)
        with self._locked():
            context = self._context()
            job = self._job(context)
        active_run_id = context.get("active_run_id")
        if job and job.get("operation") == "recommend_models":
            if job.get("status") in {"starting", "running"}:
                active_run_id = None
            elif job.get("result", {}).get("run_id"):
                active_run_id = job["result"]["run_id"]
        result = {"workload": context.get("workload", {}), "active_run_id": active_run_id}
        if job:
            result["job"] = {k: job[k] for k in ("job_id", "status", "error_type", "stage") if k in job}
            if job.get("started_at"):
                result["job"]["elapsed_seconds"] = round((job.get("finished_at") or time.time()) - job["started_at"], 1)
        if result["active_run_id"]:
            view = self.get_run_status({"run_id": result["active_run_id"]})
            # Return current workflow state, not the full recommendation/evidence history.
            result["run"] = {k: view[k] for k in (
                "run_id", "status", "selected_model_id", "selection", "needs_input", "questions",
                "optimization_preparation", "optimization_status", "selection_prompt") if k in view}
            _, run = load_run(self.state_root, result["active_run_id"])
            result["optimization_choices"] = run.get("optimization_choices", {})
            execution = run.get("optimization_execution", {})
            if run.get("selected_model_id"):
                result["optimization_controls"] = {k: execution[k] for k in
                    ("dry_run", "optuna", "timeout_minutes") if k in execution}
            if run.get("selected_model_id"):
                prepared = run.get("optimization_preparation") or {}
                result["next_tool"] = ("start_optimization" if prepared.get("status") == "prepared"
                                       else "prepare_optimization")
                result["requires_explicit_approval"] = True
                result["next_tool_arguments"] = {"run_id": result["active_run_id"]}
                result["input_stage"] = "optimization"
                result["run"].pop("needs_input", None)
                result["run"].pop("questions", None)
            else:
                result["next_tool"] = "provide_active_input" if run.get("status") == "needs_input" else "select_model"
                result["input_stage"] = "recommendation"
                result["run"]["recommendations"] = [{k: row[k] for k in
                    ("model_id", "revision", "eligible", "why_it_fits", "limitations", "task_fit", "source_urls") if k in row}
                    for row in view.get("recommendations", [])[:3]]
        return result

    def update_workload(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        values = WorkloadInputs.model_validate(raw).model_dump(exclude_unset=True)
        with self._locked():
            context = self._context()
            for key, value in values.items():
                if key in WorkloadInputs.model_fields and key != "request":
                    if value is None:
                        context.setdefault("workload", {}).pop(key, None)
                    else:
                        context.setdefault("workload", {})[key] = value
            write_json(self.context_path, context)
        return {"workload": context["workload"]}

    def start_recommendation(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        explicit = RecommendationInputs.model_validate(raw).model_dump(exclude_unset=True)
        with self._locked():
            context = self._context()
            job = self._job(context)
            if job and job["status"] in {"starting", "running"}:
                return {"status": "running", "job_id": job["job_id"], "message": "A recommendation is already running; use get_conversation_context."}
            if (job and job.get("result", {}).get("status") == "needs_input"
                    and "request" in job.get("result", {}).get("needs_input", [])
                    and explicit["request"] == job.get("arguments", {}).get("request")):
                return {"status": "needs_input", "run": job["result"], "message": "Answer the active run with provide_active_input before starting another."}
            parsed = extract_constraints(explicit["request"])
            values = {k: v for k, v in context.get("workload", {}).items() if k in WORKLOAD_FIELDS}
            values.update({k: v for k, v in parsed.items() if k in WORKLOAD_FIELDS and v is not None})
            preference = speculative_preference(explicit["request"])
            if preference:
                values["speculative_decoding"] = preference
            values.update(explicit)
            inputs = RecommendationInputs.model_validate(values)
            return self._launch_locked(context, "recommend_models", inputs.model_dump(exclude_none=True))

    def provide_active_input(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        answers = ActiveAnswersInputs.model_validate(raw).answers
        optimization_run = None
        with self._locked():
            context = self._context()
            job = self._job(context)
            if job and job["status"] in {"starting", "running"}:
                return {"status": "running", "job_id": job["job_id"]}
            run_id = context.get("active_run_id") or (job or {}).get("result", {}).get("run_id")
            if not run_id:
                raise ValueError("No active run awaiting answers")
            _, run = load_run(self.state_root, run_id)
            if run.get("selected_model_id"):
                from hf_model_watch.optimization.preparation import PreparationInputs
                unexpected = set(answers) - (set(PreparationInputs.model_fields) - {"run_id", "execution"})
                if unexpected:
                    raise ValueError(f"Unsupported optimization choices: {sorted(unexpected)}")
                optimization_run = run_id
            else:
                if run.get("status") != "needs_input":
                    raise ValueError("The active run is not awaiting recommendation input; select a model or use prepare_optimization")
                unexpected = set(answers) - set(run.get("needs_input", []))
                if unexpected:
                    raise ValueError(f"Answers were not requested: {sorted(unexpected)}")
                args = ClarificationInputs(run_id=run_id, answers=answers).model_dump()
                return self._launch_locked(context, "provide_recommendation_input", args)
        # Preparation takes the conversation lock itself; never nest it here.
        return self.prepare_optimization({"run_id": optimization_run, **answers})

    def _launch_locked(self, context: dict[str, Any], operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ident = "job_" + uuid4().hex
        path = self.state_root / "jobs" / (ident + ".json")
        write_json(path, {"job_id": ident, "status": "starting", "stage": "queued", "started_at": time.time(), "operation": operation,
                          "arguments": arguments, "project_root": str(self.project_root), "timeout": self.timeout_seconds})
        process = subprocess.Popen([sys.executable, "-m", "hf_model_watch.conversation", str(path)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        job = json.loads(path.read_text())
        # The worker waits for this lock, so it cannot overwrite this publication.
        job["pid"] = process.pid
        job["process_identity"] = process_identity(process.pid)
        write_json(path, job)
        context["active_job"] = ident
        if operation == "recommend_models":
            context.pop("active_run_id", None)
        write_json(self.context_path, context)
        return {"status": "running", "job_id": ident, "message": "Recommendation started. Ask for status; no need to resend the request."}

    def revise_recommendation(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        result = super().revise_recommendation(raw)
        with self._locked():
            context = self._context()
            context.pop("active_job", None)
            context["active_run_id"] = result["run_id"]
            _, run = load_run(self.state_root, result["run_id"])
            context.setdefault("workload", {}).update({k: v for k, v in run.get("request_spec", {}).items()
                                                        if k in WORKLOAD_FIELDS and v is not None})
            write_json(self.context_path, context)
        return result

    def get_active_model_evidence(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        inputs = EvidenceInputs.model_validate(raw)
        model_id = inputs.model_id
        if inputs.run_id:
            run = self.get_run_status({"run_id": inputs.run_id})
        else:
            context = self.get_conversation_context({})
            run = context.get("run", {})
        # Read evidence from the canonical file; status responses intentionally omit it.
        if run.get("run_id"):
            _, run = load_run(self.state_root, run["run_id"])
        rows = [row for row in run.get("recommendations", []) if row.get("model_id") == model_id]
        if not rows:
            raise ValueError("Model is not in the active shortlist")
        row = rows[0]
        return {"run_id": run.get("run_id"), "model_id": model_id,
                "task_fit": row.get("task_fit"), "runtime_assessment": row.get("runtime_assessment"),
                "nemotron": row.get("nemotron"), "limitations": row.get("limitations"),
                "evidence": [{k: (str(v)[:1500] if k == "retrieved_text" else v) for k, v in e.items()}
                             for e in row.get("evidence", [])][:20]}


def run_worker(path: Path) -> None:
    job = json.loads(path.read_text())
    root = path.parent.parent
    adapter = ConversationAdapter(state_root=root, project_root=Path(job["project_root"]), timeout_seconds=job["timeout"])
    def progress(stage: str) -> None:
        with adapter._locked():
            current = json.loads(path.read_text())
            current["stage"] = stage
            write_json(path, current)
    adapter._run_cli = _default_cli_runner(adapter.project_root, progress=progress)
    with adapter._locked():
        job = json.loads(path.read_text())
        job.update(status="running", stage="recommendation")
        write_json(path, job)
    try:
        # Base operations perform all authoritative validation and CLI work.
        result = getattr(adapter, job["operation"])(job["arguments"])
        expected_request = job["arguments"].get("request")
        actual_request = result.get("task_request") or result.get("request_spec", {}).get("request")
        if job["operation"] == "recommend_models" and expected_request and actual_request and actual_request != expected_request:
            raise ValueError("Recommendation returned a stale workload")
        job.update(status="completed", stage="finished", result=result, finished_at=time.time())
        with adapter._locked():
            context = adapter._context()
            if context.get("active_job") == job["job_id"]:
                context["active_run_id"] = result.get("run_id")
                workload = context.setdefault("workload", {})
                _, run = load_run(root, result["run_id"])
                for k, v in run.get("request_spec", {}).items():
                    if k in WORKLOAD_FIELDS and v is not None:
                        workload[k] = v
                write_json(adapter.context_path, context)
            write_json(path, job)
    except Exception as exc:
        # Do not persist exception messages that could contain credentials.
        job.update(status="failed", stage="finished", error_type=type(exc).__name__, finished_at=time.time())
        with adapter._locked():
            write_json(path, job)


if __name__ == "__main__":
    run_worker(Path(sys.argv[1]))
