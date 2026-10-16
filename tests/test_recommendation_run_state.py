from __future__ import annotations

import json

from typer.testing import CliRunner

from hf_model_watch.cli import app
from hf_model_watch.recommender import discovery


def _complete_args(root, request="Find a model"):
    return [
        "recommend",
        request,
        "--gpu-memory-gb", "80",
        "--budget-usd-per-hour", "5",
        "--gpu-hourly-cost-usd", "3",
        "--license", "any",
        "--framework-mode", "sweep", "--speculative-decoding", "prefer",
        "--state-root", str(root),
        "--json",
    ]


def _fake_discovery(spec, _settings, _destination, **_kwargs):
    constraints = discovery.extract_constraints(spec.request)
    return {
        "status": "recommended",
        "task_request": spec.request,
        "constraints": constraints,
        "recommendations": [{"model_id": "org/model", "revision": "abc123"}],
        "selected_model_id": None,
        "action": "Optimize Inference + Benchmark",
        "action_status": "placeholder",
    }


def test_needs_input_has_questions_and_can_resume(tmp_path, monkeypatch):
    first = CliRunner().invoke(
        app, ["recommend", "--state-root", str(tmp_path), "--json"]
    )
    assert first.exit_code == 0
    initial = json.loads(first.stdout)
    assert initial["status"] == "needs_input"
    assert initial["action_status"] == "placeholder"
    assert {question["field"] for question in initial["questions"]} == set(
        initial["needs_input"]
    )

    monkeypatch.setattr(discovery, "discover_models", _fake_discovery)
    resumed = CliRunner().invoke(
        app,
        [
            "recommend", "Find a coding model", "--run-id", initial["run_id"],
            "--gpu-memory-gb", "80", "--budget-usd-per-hour", "5",
            "--gpu-hourly-cost-usd", "3", "--license", "any",
            "--framework-mode", "sweep", "--speculative-decoding", "prefer", "--state-root", str(tmp_path), "--json",
        ],
    )
    assert resumed.exit_code == 0
    completed = json.loads(resumed.stdout)
    assert completed["run_id"] == initial["run_id"]
    assert completed["status"] == "recommended"
    state = json.loads(
        (tmp_path / initial["run_id"] / "recommendation_run.json").read_text()
    )
    assert state["task_request"] == "Find a coding model"


def test_generated_token_limit_survives_cli_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "discover_models", _fake_discovery)
    request = (
        "Find a new model like Qwen3.5-4B for one 80GB A100 GPU, a budget of "
        "$5/hour, and a maximum of 2,000 generated tokens; any license, sweep mode."
    )
    result = CliRunner().invoke(app, _complete_args(tmp_path, request))
    assert result.exit_code == 0
    assert json.loads(result.stdout)["constraints"]["max_generation_tokens"] == 2000


def test_selection_feedback_and_traceable_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "discover_models", _fake_discovery)
    created = CliRunner().invoke(app, _complete_args(tmp_path))
    run = json.loads(created.stdout)

    selected = CliRunner().invoke(
        app,
        ["select", "--run-id", run["run_id"], "--model-id", "org/model",
         "--state-root", str(tmp_path), "--json"],
    )
    assert selected.exit_code == 0
    selection = json.loads(selected.stdout)
    assert selection["revision"] == "abc123"
    assert selection["evidence_reference"].endswith("discovery.json")
    duplicate = CliRunner().invoke(
        app,
        ["select", "--run-id", run["run_id"], "--model-id", "org/model",
         "--state-root", str(tmp_path), "--json"],
    )
    assert duplicate.exit_code == 0

    feedback_result = CliRunner().invoke(
        app,
        ["feedback", "--run-id", run["run_id"], "--model-id", "org/model",
         "--category", "out_of_memory", "--details", "OOM at 32k context",
         "--context-json", '{"gpu":"A100 80GB"}',
         "--state-root", str(tmp_path), "--json"],
    )
    assert feedback_result.exit_code == 0
    feedback = json.loads(feedback_result.stdout)
    assert feedback["feedback_status"] == "reported"
    assert feedback["context"] == {"gpu": "A100 80GB"}

    revised = CliRunner().invoke(
        app,
        [*_complete_args(tmp_path, "Find a smaller coding model"),
         "--parent-run-id", run["run_id"],
         "--trigger-feedback-id", feedback["feedback_id"]],
    )
    assert revised.exit_code == 0
    revision = json.loads(revised.stdout)
    assert revision["parent_run_id"] == run["run_id"]
    assert revision["trigger_feedback_id"] == feedback["feedback_id"]


def test_invalid_selection_and_unchanged_revision_fail_predictably(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "discover_models", _fake_discovery)
    created = json.loads(CliRunner().invoke(app, _complete_args(tmp_path)).stdout)
    invalid = CliRunner().invoke(
        app,
        ["select", "--run-id", created["run_id"], "--model-id", "org/other",
         "--state-root", str(tmp_path), "--json"],
    )
    assert invalid.exit_code == 1
    assert json.loads(invalid.stdout)["status"] == "error"

    unchanged = CliRunner().invoke(
        app,
        [*_complete_args(tmp_path), "--parent-run-id", created["run_id"],
         "--revision-reason", "retry"],
    )
    assert unchanged.exit_code == 1
    assert "must change" in json.loads(unchanged.stdout)["error"]


def test_missing_run_conflicting_selection_and_malformed_feedback_fail(tmp_path, monkeypatch):
    missing = CliRunner().invoke(
        app,
        ["select", "--run-id", "run_missing", "--model-id", "org/model",
         "--state-root", str(tmp_path), "--json"],
    )
    assert missing.exit_code == 1
    assert "not found" in json.loads(missing.stdout)["error"]

    monkeypatch.setattr(discovery, "discover_models", _fake_discovery)
    created = json.loads(CliRunner().invoke(app, _complete_args(tmp_path)).stdout)
    state_path = tmp_path / created["run_id"] / "recommendation_run.json"
    state = json.loads(state_path.read_text())
    state["recommendations"].append({"model_id": "org/second"})
    state_path.write_text(json.dumps(state))
    for model_id, expected_code in (("org/model", 0), ("org/second", 1)):
        result = CliRunner().invoke(
            app,
            ["select", "--run-id", created["run_id"], "--model-id", model_id,
             "--state-root", str(tmp_path), "--json"],
        )
        assert result.exit_code == expected_code
    assert "different model" in json.loads(result.stdout)["error"]

    malformed = CliRunner().invoke(
        app,
        ["feedback", "--run-id", created["run_id"], "--model-id", "org/model",
         "--category", "quality", "--details", "bad output",
         "--context-json", "[]", "--state-root", str(tmp_path), "--json"],
    )
    assert malformed.exit_code == 1
    assert "JSON object" in json.loads(malformed.stdout)["error"]
