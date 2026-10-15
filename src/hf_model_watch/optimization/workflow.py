"""Optional recommendation transitions; model selection is sufficient on its own."""

from __future__ import annotations

import json
import os
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
        with locked_run(self.state_root, run_id) as (directory, run):
            self._check_selection(run, intent.context)
            existing = run.get("optimization_reference")
            if existing:
                ref = OptimizationReference.model_validate(existing)
                if ref.intent != intent:
                    raise ValueError(
                        "Recommendation already references a different optimization intent"
                    )
                return self._adapter().status(ref.study_reference)
            response = self._adapter().submit(intent)
            ref = OptimizationReference(
                study_reference=response["run_id"], mode="new", intent=intent
            )
            run["optimization_reference"] = ref.model_dump(mode="json")
            save_run(directory, run)
            return response

    def status(self, run_id: str) -> dict:
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
