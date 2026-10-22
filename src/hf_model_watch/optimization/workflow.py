"""Optional recommendation transitions; model selection is sufficient on its own."""

from __future__ import annotations

import json
import fcntl
import os
import sqlite3
from pathlib import Path

from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter
from hf_model_watch.recommender.run_state import load_run, locked_run, save_run

from .analysis import analyze
from .catalog import FixtureStudyCatalog, StudyCatalog, incompatibilities
from .contracts import NewStudyIntent, OptimizationReference, ReuseIntent, StudyContext


class OptimizationWorkflow:
    def __init__(
        self,
        state_root: Path,
        adapter: OptimizationAdapter | None = None,
        catalog: StudyCatalog | None = None,
    ):
        self.state_root, self.adapter = state_root, adapter
        self.catalog = catalog or FixtureStudyCatalog()

    def _adapter(self) -> OptimizationAdapter:
        if self.adapter is None:
            raise ValueError("Optimizer controller is not configured")
        return self.adapter

    @staticmethod
    def _check_selection(run: dict, context: StudyContext) -> None:
        if run.get("selected_model_id") != context.model_id:
            raise ValueError("Optimization requires the selected recommendation model")
        revision = (run.get("selection") or {}).get("revision")
        if revision and revision != context.revision:
            raise ValueError("Optimization revision differs from selected revision")

    def start(self, run_id: str, intent: NewStudyIntent) -> dict:
        # Serialize submissions across recommendation runs, independently of reconciliation.
        database = getattr(getattr(self._adapter(), "controller", None), "database", None)
        lock_path = (Path(str(Path(database).resolve()) + ".submission.lock")
                     if isinstance(database, (str, Path)) else self.state_root / ".optimization-submit.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._start_locked(run_id, intent)

    def _start_locked(self, run_id: str, intent: NewStudyIntent) -> dict:
        with locked_run(self.state_root, run_id) as (directory, run):
            self._check_selection(run, intent.context)
            adapter = self._adapter()
            existing = run.get("optimization_reference")
            if existing:
                ref = OptimizationReference.model_validate(existing)
                current = adapter.status(ref.study_reference)
                terminal = current.get("state") in {"cancelled", "completed", "failed", "dry_run"}
                if not terminal:
                    if ref.intent != intent:
                        raise ValueError("Recommendation already references a different optimization intent; study is still active")
                    return {**current, "submission_created": False}
                if ref.intent.idempotency_key == intent.idempotency_key:
                    return {**current, "submission_created": False,
                            "message": "Study is terminal. Prepare a new request and approve it before retrying."}
            database = getattr(getattr(adapter, 'controller', None), 'database', None)
            health = None
            if (isinstance(database, (str, Path))
                    and Path(str(Path(database).resolve()) + '.controller-managed.json').is_file()):
                from llm_optimizer.remote.service import ensure_running
                health = ensure_running(database)
            if isinstance(database, (str, Path)):
                with sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True) as db:
                    active = db.execute("SELECT run_id FROM jobs WHERE state NOT IN "
                                        "('completed','failed','cancelled','dry_run') LIMIT 1").fetchone()
                if active:
                    raise ValueError(f"Optimizer study {active[0]} is still active; no new study submitted")
            response = {**adapter.submit(intent), "submission_created": True}
            if health and health.get('managed'):
                response['controller'] = {k: health.get(k) for k in
                                          ('managed', 'healthy', 'concurrency')}
            ref = OptimizationReference(
                study_reference=response["run_id"], mode="new", intent=intent
            )
            run["optimization_reference"] = ref.model_dump(mode="json")
            save_run(directory, run)
            return response

    def status(self, run_id: str, *, include_results: bool = True) -> dict:
        _, run = load_run(self.state_root, run_id)
        raw = run.get("optimization_reference")
        if not raw:
            return {"status": "not_requested", "optimization_reference": None}
        ref = OptimizationReference.model_validate(raw)
        if ref.mode == "reuse":
            return {
                "status": "analyzed",
                "optimization_reference": raw,
                "result": run.get("optimization_result"),
            }
        adapter = self._adapter()
        status = adapter.status(ref.study_reference)
        if not include_results:
            return {
                'run_id': run_id,
                'study_reference': ref.study_reference,
                'state': status.get('state'),
                'jobs': [{key: job.get(key) for key in
                          ('framework', 'state', 'nebius_id', 'progress', 'error')}
                         for job in status.get('jobs', [])],
            }
        summary = adapter.results(ref.study_reference, ref.intent.context)
        return {
            **status,
            "optimization_reference": raw,
            "summary": summary.model_dump(mode="json"),
            "result": analyze(summary, ref.intent.selection_policy).model_dump(mode="json"),
        }

    def list_prior_studies(self, run_id: str, context: StudyContext) -> dict:
        _, run = load_run(self.state_root, run_id)
        self._check_selection(run, context)
        studies = self.catalog.list_studies()
        return {
            "studies": [
                {
                    "study_reference": s.study_reference,
                    "state": s.state,
                    "comparable": not incompatibilities(context, s.context),
                    "incompatibilities": incompatibilities(context, s.context),
                }
                for s in studies
            ],
            "message": None if studies else "no prior studies available",
        }

    def reuse(self, run_id: str, intent: ReuseIntent) -> dict:
        with locked_run(self.state_root, run_id) as (directory, run):
            self._check_selection(run, intent.context)
            try:
                summary = self.catalog.get_study(intent.study_reference)
            except ValueError:
                # A known controller run can be reused even before catalog indexing exists.
                if self.adapter is None:
                    raise
                summary = self.adapter.results(intent.study_reference, intent.context)
                if not summary.candidates and any(
                    "Incomparable measurements" in error
                    for error in summary.framework_errors.values()
                ):
                    raise ValueError(
                        "Prior study is incomparable with the requested context"
                    ) from None
            mismatch = incompatibilities(intent.context, summary.context)
            if mismatch:
                raise ValueError(f"Prior study is incomparable: {', '.join(mismatch)}")
            result = analyze(summary, intent.selection_policy).model_dump(mode="json")
            run["optimization_reference"] = OptimizationReference(
                study_reference=intent.study_reference, mode="reuse", intent=intent
            ).model_dump(mode="json")
            run["optimization_result"] = result
            save_run(directory, run)
            return result

    def cancel(self, run_id: str) -> dict:
        _, run = load_run(self.state_root, run_id)
        ref = OptimizationReference.model_validate(run.get("optimization_reference"))
        if ref.mode != "new":
            raise ValueError("Reuse is read-only and has no job to cancel")
        return self._adapter().cancel(ref.study_reference)


def configured_workflow(state_root: Path) -> OptimizationWorkflow:
    """Lazy optional dependency; no controller ticks or jobs are launched here."""
    database = os.getenv("HF_MODEL_WATCH_OPTIMIZER_DATABASE")
    if not database:
        return OptimizationWorkflow(state_root)
    try:
        from llm_optimizer.remote.controller import Controller
    except ImportError as exc:
        raise RuntimeError(
            "Install llm-optimizer with its remote extras to configure the controller"
        ) from exc

    execution = json.loads(os.environ.get("HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON", "{}"))
    return OptimizationWorkflow(state_root, OptimizationAdapter(Controller(database), execution))


def connection_status() -> dict:
    """Probe imports, existing controller tables and defaults without creating/ticking jobs."""
    database = os.getenv("HF_MODEL_WATCH_OPTIMIZER_DATABASE")
    result = {"connected": False, "execution_ready": False, "gpu_jobs_started": 0}
    if not database:
        return {**result, "status": "not_connected", "reason": "Optimizer database is not configured"}
    try:
        from llm_optimizer.remote.controller import Controller  # noqa: F401
        path = Path(database).resolve(strict=True)
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"runs", "jobs"} <= tables:
                raise ValueError("Controller schema is missing")
            # Schema verification is read-only, unlike constructing Controller.
            db.execute("SELECT id, request FROM runs LIMIT 0")
            db.execute("SELECT run_id, framework, state FROM jobs LIMIT 0")
        execution = json.loads(os.environ.get("HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON", "{}"))
        if not isinstance(execution, dict):
            raise ValueError("Invalid execution defaults")
        managed = Path(str(path) + '.controller-managed.json').is_file()
        health = {}
        if managed:
            from llm_optimizer.remote.service import controller_health
            health = controller_health(path)
        return {**result, "connected": True, "status": "connected",
                "execution_mode": "dry_run" if execution.get("dry_run", True) else "live",
                "frameworks": sorted(execution.get("images", {})),
                "timeout_minutes": execution.get("timeout_minutes"),
                "trials": execution.get("optuna", {}).get("trials"),
                "controller_managed": managed,
                "controller_healthy": health.get('healthy', False),
                "controller_reconciliation": ("auto-started before approved submission; not started by this probe"
                                              if managed else "external; not started by this probe"),
                "limitation": "Connection verified; selected context, prices, study budget and approval are still required."}
    except (ImportError, OSError, ValueError, sqlite3.Error, TypeError) as exc:
        return {**result, "status": "not_connected", "reason": type(exc).__name__}
