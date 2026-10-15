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
