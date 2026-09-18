from __future__ import annotations

import json

from hf_model_watch.finetune.eval_adaptation import compute_adaptation_metrics
from hf_model_watch.finetune.train_peft import run_peft_adaptation_suite
from hf_model_watch.storage.sqlite_store import SQLiteStore


def _trace_events(output: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("TRACE_EVENT "))
        for line in output.splitlines()
        if line.startswith("TRACE_EVENT ")
    ]


def test_adaptation_lift_calculation() -> None:
    metrics = compute_adaptation_metrics(
        model_id="org/model",
        run_id="run_1",
        task_type="json_extraction",
        peft_config={"method": "qlora", "max_steps": 10},
        base_quality_score=0.4,
        adapted_quality_score=0.7,
        token_counts={
            "training_examples": 2,
            "training_prompt_tokens": 50,
            "training_completion_tokens": 10,
            "training_total_tokens": 60,
        },
        trainable_params=100,
        total_params=1000,
    )

    assert round(metrics["adaptation_lift"], 6) == 0.3
    assert round(metrics["normalized_adaptation_lift"], 6) == 0.5
    assert round(metrics["adaptation_lift_per_1M_training_tokens"], 6) == 5000
    assert metrics["trainable_param_ratio"] == 0.1


def test_finetune_unavailable_path_records_typed_result(tmp_path) -> None:
    class Serverless:
        gpu_memory_gb = 80

    class Config:
        serverless = Serverless()
        finetune = {"method": "qlora", "max_train_steps": 5}

    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    store.start_run("run_1", "org/model", "abc123", artifact_uri=str(tmp_path))

    result = run_peft_adaptation_suite(
        run_id="run_1",
        model_id="org/model",
        revision="abc123",
        run_dir=tmp_path,
        config=Config(),
        store=store,
        smoke_only=True,
        skip=True,
        metadata={"estimated_params_billion": 0.1, "context_length": 1024},
    )

    assert result["status"] == "skipped"
    assert (tmp_path / "finetune_config.json").exists()
    assert (tmp_path / "finetune_plan.json").exists()
    assert (tmp_path / "model_profiles.json").exists()
    assert (tmp_path / "adaptation_results.json").exists()
    assert (tmp_path / "adapter_eval.json").exists()
    adapter_eval = json.loads((tmp_path / "adapter_eval.json").read_text(encoding="utf-8"))
    assert adapter_eval["failure_category"] is None


def test_finetune_unavailable_dependencies_are_visible_in_trace(
    tmp_path, monkeypatch, capsys
) -> None:
    class Serverless:
        gpu_memory_gb = 80

    class Config:
        serverless = Serverless()
        finetune = {"method": "qlora", "max_train_steps": 5}

    monkeypatch.setattr(
        "hf_model_watch.finetune.train_peft._module_available",
        lambda name: False,
    )
    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    store.start_run("run_2", "org/model", "abc123", artifact_uri=str(tmp_path))

    result = run_peft_adaptation_suite(
        run_id="run_2",
        model_id="org/model",
        revision="abc123",
        run_dir=tmp_path,
        config=Config(),
        store=store,
        smoke_only=True,
        metadata={"estimated_params_billion": 1, "context_length": 1024},
    )

    events = _trace_events(capsys.readouterr().out)
    dependencies = next(event for event in events if event["event_type"] == "finetune_dependencies")
    finished = next(event for event in events if event["event_type"] == "finetune_finished")
    assert result["status"] == "smoke_unavailable"
    assert dependencies["status"] == "failed"
    assert dependencies["metadata"]["missing_dependencies"] == [
        "peft",
        "trl",
        "bitsandbytes",
        "wandb",
    ]
    assert finished["status"] == "failed"
    assert finished["metadata"]["training_completed"] is False


def test_successful_peft_training_records_measured_lift(tmp_path, monkeypatch) -> None:
    class Serverless:
        gpu_memory_gb = 80

    class Config:
        serverless = Serverless()
        finetune = {"method": "qlora", "max_train_steps": 2}

    monkeypatch.setattr(
        "hf_model_watch.finetune.train_peft._module_available",
        lambda name: True,
    )
    monkeypatch.setattr(
        "hf_model_watch.finetune.train_peft._load_profile_tokenizer",
        lambda model_id, revision: None,
    )
    calls = []

    def fake_training_attempt(**kwargs):
        task_type = kwargs["task_type"]
        calls.append(kwargs)
        return {
            "success": True,
            "retry_recommended": False,
            "message": "PEFT training completed",
            "adapter_dir": str(tmp_path / "finetune" / task_type / "adapter"),
            "trainable_params": 10,
            "total_params": 100,
            "base_scores": {task_type: 0.25},
            "adapted_scores": {task_type: 0.75},
        }

    monkeypatch.setattr(
        "hf_model_watch.finetune.train_peft.run_training_attempt_subprocess",
        fake_training_attempt,
    )
    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    store.start_run("run_3", "org/model", "abc123", artifact_uri=str(tmp_path))

    result = run_peft_adaptation_suite(
        run_id="run_3",
        model_id="org/model",
        revision="abc123",
        run_dir=tmp_path,
        config=Config(),
        store=store,
        metadata={"estimated_params_billion": 1, "context_length": 1024},
    )

    assert result["status"] == "training_completed"
    assert result["training_completed"] is True
    assert result["overall"]["adaptation_lift"] > 0
    assert all(item["trainable_params"] == 10 for item in result["task_results"])
    assert {call["task_type"] for call in calls} == {
        "json_extraction",
        "text_to_sql",
        "financial_analyst_style",
    }
    assert all(list(call["train_datasets"]) == [call["task_type"]] for call in calls)
    assert all(call["peft_config"]["target_modules"] == "all-linear" for call in calls)


def test_auto_method_retries_once_with_safer_configuration(tmp_path, monkeypatch) -> None:
    class Serverless:
        gpu_memory_gb = 80
        gpu_type = "H100"

    class Config:
        serverless = Serverless()
        finetune = {"method": "auto", "max_train_steps": 2}

    monkeypatch.setattr(
        "hf_model_watch.finetune.train_peft._module_available",
        lambda name: True,
    )
    monkeypatch.setattr(
        "hf_model_watch.finetune.train_peft._load_profile_tokenizer",
        lambda model_id, revision: None,
    )
    calls = []

    def fake_training_attempt(**kwargs):
        calls.append(kwargs)
        task = kwargs["task_type"]
        if kwargs["attempt_number"] == 1:
            return {
                "success": False,
                "retry_recommended": True,
                "retry_reason": "cuda_oom",
                "message": "CUDA out of memory",
            }
        return {
            "success": True,
            "retry_recommended": False,
            "message": "QLORA training completed",
            "adapter_dir": str(tmp_path / "finetune" / task / "adapter"),
            "trainable_params": 10,
            "total_params": 100,
            "base_scores": {task: 0.25},
            "adapted_scores": {task: 0.75},
        }

    monkeypatch.setattr(
        "hf_model_watch.finetune.train_peft.run_training_attempt_subprocess",
        fake_training_attempt,
    )
    store = SQLiteStore(tmp_path / "results.sqlite")
    store.initialize()
    store.start_run("run_4", "org/model", "abc123", artifact_uri=str(tmp_path))

    result = run_peft_adaptation_suite(
        run_id="run_4",
        model_id="org/model",
        revision="abc123",
        run_dir=tmp_path,
        config=Config(),
        store=store,
        metadata={"estimated_params_billion": 1, "context_length": 1024},
    )

    assert result["status"] == "training_completed"
    assert len(calls) == 6
    assert all(calls[index]["peft_config"]["method"] == "lora" for index in (0, 2, 4))
    assert all(calls[index]["peft_config"]["method"] == "qlora" for index in (1, 3, 5))
    assert all(calls[index]["peft_config"]["microbatch_size"] <= 2 for index in (1, 3, 5))
