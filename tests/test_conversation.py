import json
import threading
from pathlib import Path

import pytest

from hf_model_watch.conversation import ConversationAdapter, run_worker, write_json
from hf_model_watch.recommender.run_state import save_run


def test_workload_persists_and_explicit_request_overrides_defaults(tmp_path, monkeypatch):
    adapter = ConversationAdapter(state_root=tmp_path)
    adapter.update_workload({"gpu_memory_gb": 80, "budget_usd_per_hour": 5, "gpu_count": 2})
    next_adapter = ConversationAdapter(state_root=tmp_path)
    assert next_adapter.get_conversation_context({})["workload"]["gpu_memory_gb"] == 80
    captured = {}
    def launch(context, operation, arguments):
        captured.update(arguments)
        return {"status": "running"}
    monkeypatch.setattr(next_adapter, "_launch_locked", launch)
    next_adapter.start_recommendation({"request": "Find models with 40 GB memory, prefer speculative decuding", "budget_usd_per_hour": 3})
    assert captured["gpu_memory_gb"] == 40
    assert captured["budget_usd_per_hour"] == 3
    assert captured["gpu_count"] == 2
    assert captured["speculative_decoding"] == "prefer"
    adapter.update_workload({"budget_usd_per_hour": None})
    assert "budget_usd_per_hour" not in adapter.get_conversation_context({})["workload"]


def test_active_clarification_survives_adapter_restart_and_rejects_unasked_fields(tmp_path, monkeypatch):
    save_run(tmp_path / "run_test", {"run_id": "run_test", "status": "needs_input", "needs_input": ["license"]})
    write_json(tmp_path / "conversation.json", {"active_run_id": "run_test", "workload": {}})
    adapter = ConversationAdapter(state_root=tmp_path)
    captured = {}
    monkeypatch.setattr(adapter, "_launch_locked", lambda c, op, args: captured.update(args) or {"status": "running"})
    adapter.provide_active_input({"answers": {"license": "any"}})
    assert captured == {"run_id": "run_test", "answers": {"license": "any"}}
    with pytest.raises(ValueError, match="not requested"):
        adapter.provide_active_input({"answers": {"gpu_memory_gb": 80}})


def test_running_job_does_not_start_duplicate(tmp_path, monkeypatch):
    adapter = ConversationAdapter(state_root=tmp_path)
    import os
    write_json(tmp_path / "conversation.json", {"workload": {}, "active_job": "job_test"})
    write_json(tmp_path / "jobs/job_test.json", {"job_id": "job_test", "status": "running", "pid": os.getpid()})
    monkeypatch.setattr(adapter, "_launch_locked", lambda *a: pytest.fail("duplicate started"))
    assert adapter.start_recommendation({"request": "Find models"})["job_id"] == "job_test"


def test_background_worker_persists_run_and_workload(tmp_path, monkeypatch):
    from hf_model_watch.openclaw_adapter import OpenClawToolAdapter
    write_json(tmp_path / "conversation.json", {"workload": {}, "active_job": "job_test"})
    path = tmp_path / "jobs/job_test.json"
    write_json(path, {"job_id": "job_test", "status": "starting", "project_root": str(tmp_path), "timeout": 10,
                     "operation": "recommend_models", "arguments": {"request": "Find models"}})
    def fake_recommend(self, arguments):
        run = {"run_id": "run_worker", "status": "needs_input", "request_spec": {"gpu_memory_gb": 80}, "needs_input": ["license"]}
        save_run(tmp_path / "run_worker", run)
        return run
    monkeypatch.setattr(OpenClawToolAdapter, "recommend_models", fake_recommend)
    run_worker(path)
    result = ConversationAdapter(state_root=tmp_path).get_conversation_context({})
    assert result["active_run_id"] == "run_worker"
    assert result["run"]["needs_input"] == ["license"]
    assert result["workload"]["gpu_memory_gb"] == 80


def test_evidence_read_never_starts_research(tmp_path):
    run = {"run_id": "run_evidence", "status": "recommended", "recommendations": [{"model_id": "org/model",
           "evidence": [{"source_url": "https://example.org", "retrieved_text": "x" * 3000}]}]}
    save_run(tmp_path / "run_evidence", run)
    write_json(tmp_path / "conversation.json", {"workload": {}, "active_run_id": "run_evidence"})
    adapter = ConversationAdapter(state_root=tmp_path, cli_runner=lambda *a: pytest.fail("research started"))
    assert len(adapter.get_active_model_evidence({"model_id": "org/model"})["evidence"][0]["retrieved_text"]) == 1500


def test_reused_pid_marks_job_interrupted(tmp_path):
    import os
    write_json(tmp_path / "conversation.json", {"workload": {}, "active_job": "job_test"})
    write_json(tmp_path / "jobs/job_test.json", {"job_id": "job_test", "status": "running", "pid": os.getpid(),
                                               "process_identity": "not-the-current-process"})
    context = ConversationAdapter(state_root=tmp_path).get_conversation_context({})
    assert context["job"]["status"] == "interrupted"


def test_status_is_compact_and_evidence_is_read_separately_for_historical_run(tmp_path):
    run = {"run_id": "run_large", "status": "recommended", "selected_model_id": None,
           "action_status": "placeholder", "recommendations": [{"model_id": "org/model",
               "evidence": [{"source_url": f"https://example.org/{i}", "retrieved_text": "x"*10000}
                            for i in range(30)], "runtime_assessment": {"hf_evidence": "x"*50000}}]}
    save_run(tmp_path / "run_large", run)
    adapter = ConversationAdapter(state_root=tmp_path)
    view = adapter.get_run_status({"run_id": "run_large"})
    assert next(iter(view)) == "conversation_next_step"
    assert view["conversation_next_step"] == "Which model would you like to select?"
    assert len(json.dumps(view)) < 10000
    assert view["action_status"] == "not_requested"
    assert len(adapter.get_active_model_evidence({"model_id": "org/model", "run_id": "run_large"})["evidence"]) == 20


def test_selected_handoff_uses_verified_connection_status(tmp_path, monkeypatch):
    from hf_model_watch.optimization import workflow
    monkeypatch.setattr(workflow, "connection_status", lambda: {
        "status": "connected", "connected": True, "execution_mode": "dry_run"})
    run = {"run_id":"run_selected", "status":"recommended", "selected_model_id":"org/model",
           "selection":{"revision":"a"*40}, "recommendations":[{"model_id":"org/model"}]}
    save_run(tmp_path / "run_selected", run)
    view = ConversationAdapter(state_root=tmp_path).get_run_status({"run_id":"run_selected"})
    assert view["optimization_handoff"]["controller_status"] == "connected"
    assert view["optimization_handoff"]["execution_mode"] == "dry_run"
    assert view["optimization_handoff"]["execution_ready"] is False


@pytest.mark.parametrize("state", ["cancelled", "completed", "failed", "dry_run", "running"])
def test_context_excludes_old_study_history_and_refreshes_status(tmp_path, state):
    from unittest.mock import Mock
    run = {"run_id": "run_current", "status": "completed",
           "optimization_reference": {"study_reference": "old", "intent": {"history": "stale"}},
           "optimization_result": {"history": "old results"}, "parent_run_id": "run_old"}
    save_run(tmp_path / "run_current", run)
    write_json(tmp_path / "conversation.json", {"active_run_id": "run_current"})
    workflow = Mock()
    workflow._adapter.return_value.status.return_value = {"run_id": "old", "state": state, "jobs": []}
    adapter = ConversationAdapter(state_root=tmp_path, optimization_workflow=workflow)
    view = adapter.get_conversation_context({})["run"]
    assert "optimization_reference" not in view
    assert "optimization_result" not in view
    assert "parent_run_id" not in view
    assert view["optimization_status"]["state"] == (state if state == "running" else "not_active")
    workflow._adapter.return_value.status.assert_called_once_with("old")


def test_explicit_selection_remains_active_after_restart(tmp_path):
    from unittest.mock import Mock
    adapter = ConversationAdapter(state_root=tmp_path)
    for run_id in ('run_stale', 'run_selected'):
        save_run(tmp_path / run_id, {'run_id': run_id, 'status': 'recommended',
                 'selected_model_id': 'org/model' if run_id == 'run_selected' else None})
    write_json(tmp_path / 'conversation.json', {'active_run_id': 'run_stale', 'active_job': 'job_old'})
    write_json(tmp_path / 'jobs/job_old.json', {'status': 'completed', 'result': {'run_id': 'run_stale'}})
    adapter._activate_run('run_selected')
    view = ConversationAdapter(state_root=tmp_path).get_conversation_context({})
    assert view['active_run_id'] == 'run_selected'
    assert view['run']['selected_model_id'] == 'org/model'
    assert view['next_tool'] == 'prepare_optimization'
    assert 'job' not in view


def test_optimization_answers_do_not_reopen_recommendation(tmp_path, monkeypatch):
    save_run(tmp_path / 'run_selected', {'run_id': 'run_selected', 'status': 'recommended',
             'selected_model_id': 'org/model', 'optimization_preparation': {'status': 'needs_input'}})
    write_json(tmp_path / 'conversation.json', {'active_run_id': 'run_selected'})
    adapter = ConversationAdapter(state_root=tmp_path)
    called = []
    monkeypatch.setattr(adapter, 'prepare_optimization', lambda values: called.append(values) or {'status': 'prepared'})
    monkeypatch.setattr(adapter, '_launch_locked', lambda *a: pytest.fail('recommendation restarted'))
    assert adapter.provide_active_input({'answers': {'budget_usd_per_hour': 7}})['status'] == 'prepared'
    assert called == [{'run_id': 'run_selected', 'budget_usd_per_hour': 7}]


def test_context_does_not_replay_large_recommendation_history(tmp_path):
    save_run(tmp_path / 'run_selected', {'run_id': 'run_selected', 'status': 'recommended',
             'selected_model_id': 'org/model', 'recommendations': [{'model_id': 'org/model', 'why_it_fits': 'x'*100000}],
             'optimization_choices': {'budget_usd_per_hour': 7},
             'optimization_execution': {'dry_run': True, 'optuna': {'trials': 2}}})
    write_json(tmp_path / 'conversation.json', {'active_run_id': 'run_selected'})
    view = ConversationAdapter(state_root=tmp_path).get_conversation_context({})
    assert len(json.dumps(view)) < 2000
    assert view['optimization_controls']['optuna']['trials'] == 2
    assert view['optimization_choices']['budget_usd_per_hour'] == 7


def test_retry_legacy_deployment_questions_runs_discovery_again(tmp_path, monkeypatch):
    adapter = ConversationAdapter(state_root=tmp_path)
    write_json(tmp_path / 'conversation.json', {'active_job': 'job_legacy', 'workload': {}})
    write_json(tmp_path / 'jobs/job_legacy.json', {'status': 'completed',
        'arguments': {'request': 'Recommend a 4B model'},
        'result': {'status': 'needs_input', 'needs_input': ['gpu_type', 'input_tokens', 'budget_usd_per_hour']}})
    captured = []
    monkeypatch.setattr(adapter, '_launch_locked', lambda c, op, args: captured.append(args) or {'status': 'running'})
    assert adapter.start_recommendation({'request': 'Recommend a 4B model'})['status'] == 'running'
    assert captured[0]['request'] == 'Recommend a 4B model'


def test_running_new_discovery_hides_previous_needs_input_run(tmp_path):
    save_run(tmp_path / 'run_old', {'run_id': 'run_old', 'status': 'needs_input',
             'needs_input': ['gpu_type', 'input_tokens', 'budget_usd_per_hour']})
    write_json(tmp_path / 'conversation.json', {'active_run_id': 'run_old',
        'active_job': 'job_new', 'workload': {}})
    write_json(tmp_path / 'jobs/job_new.json', {'job_id': 'job_new', 'operation': 'recommend_models',
        'status': 'running', 'stage': 'researching', 'pid': None})
    view = ConversationAdapter(state_root=tmp_path).get_conversation_context({})
    assert view['job']['status'] == 'running'
    assert view['job']['stage'] == 'researching'
    assert view['active_run_id'] is None
    assert 'run' not in view


def test_completed_discovery_supersedes_previous_active_run(tmp_path):
    save_run(tmp_path / 'run_old', {'run_id': 'run_old', 'status': 'needs_input',
             'needs_input': ['gpu_type']})
    save_run(tmp_path / 'run_new', {'run_id': 'run_new', 'status': 'recommended',
             'selected_model_id': None, 'recommendations': [{'model_id': 'org/model'}]})
    write_json(tmp_path / 'conversation.json', {'active_run_id': 'run_old',
        'active_job': 'job_new', 'workload': {}})
    write_json(tmp_path / 'jobs/job_new.json', {'job_id': 'job_new', 'operation': 'recommend_models',
        'status': 'completed', 'result': {'run_id': 'run_new'}, 'pid': None})
    view = ConversationAdapter(state_root=tmp_path).get_conversation_context({})
    assert view['active_run_id'] == 'run_new'
    assert view['run']['run_id'] == 'run_new'
    assert 'needs_input' not in view['run']
