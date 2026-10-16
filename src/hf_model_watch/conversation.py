"""Group-local conversation state and detached recommendation jobs.

The agent translates natural language; Python validates and persists its inputs.
No model selection, benchmark, or spending is triggered by remembering a workload.
"""
from __future__ import annotations

import fcntl
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

from hf_model_watch.openclaw_adapter import (
    OpenClawToolAdapter, RecommendationInputs, ClarificationInputs, StrictModel, _default_cli_runner,
)
from hf_model_watch.recommender.run_state import load_run
from hf_model_watch.recommender.utilities import extract_constraints, speculative_preference


class ContextInputs(StrictModel):
    pass


class WorkloadInputs(RecommendationInputs):
    request: str = "Update remembered workload"


class ActiveAnswersInputs(StrictModel):
    answers: dict[str, str | int | float]


class EvidenceInputs(StrictModel):
    model_id: str


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

    def get_conversation_context(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        ContextInputs.model_validate(raw)
        with self._locked():
            context = self._context()
            job = self._job(context)
        result = {"workload": context.get("workload", {}), "active_run_id": context.get("active_run_id")}
        if job:
            result["job"] = {k: job[k] for k in ("job_id", "status", "error_type", "stage") if k in job}
            if job.get("started_at"):
                result["job"]["elapsed_seconds"] = round((job.get("finished_at") or time.time()) - job["started_at"], 1)
            if job.get("result"):
                result["active_run_id"] = job["result"].get("run_id")
                result["run"] = self.get_run_status({"run_id": result["active_run_id"]})
        elif result["active_run_id"]:
            result["run"] = self.get_run_status({"run_id": result["active_run_id"]})
        return result

    def update_workload(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        values = WorkloadInputs.model_validate(raw).model_dump(exclude_unset=True)
        with self._locked():
            context = self._context()
            for key, value in values.items():
                if key in WORKLOAD_FIELDS:
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
            if job and job.get("result", {}).get("status") == "needs_input" and explicit["request"] == job.get("arguments", {}).get("request"):
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
        with self._locked():
            context = self._context()
            job = self._job(context)
            if job and job["status"] in {"starting", "running"}:
                return {"status": "running", "job_id": job["job_id"]}
            run_id = (job or {}).get("result", {}).get("run_id") or context.get("active_run_id")
            if not run_id:
                raise ValueError("No active run awaiting answers")
            _, run = load_run(self.state_root, run_id)
            if run.get("status") != "needs_input":
                raise ValueError("The active run is not awaiting input; use start_recommendation for a new request")
            unexpected = set(answers) - set(run.get("needs_input", []))
            if unexpected:
                raise ValueError(f"Answers were not requested: {sorted(unexpected)}")
            args = ClarificationInputs(run_id=run_id, answers=answers).model_dump()
            return self._launch_locked(context, "provide_recommendation_input", args)

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
        model_id = EvidenceInputs.model_validate(raw).model_id
        context = self.get_conversation_context({})
        run = context.get("run", {})
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
