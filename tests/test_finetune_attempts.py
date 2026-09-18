from __future__ import annotations

import json
import sys

import pytest

from hf_model_watch.finetune.attempt_subprocess import run_training_attempt_subprocess
from hf_model_watch.finetune.datasets import AdaptationExample
from hf_model_watch.finetune.trl_training import _select_optimizer, _wandb_run_name


def test_attempt_runs_as_python_process_inside_current_job(tmp_path, monkeypatch) -> None:
    observed = {}

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs
            request = json.loads(open(command[-1], encoding="utf-8").read())
            observed["request"] = request
            with open(request["result_path"], "w", encoding="utf-8") as handle:
                json.dump({"success": True, "message": "done"}, handle)
            self.stdout = iter(["trace\n"])

        def wait(self):
            return self.returncode

    monkeypatch.setattr("subprocess.Popen", FakeProcess)
    example = AdaptationExample("json_extraction", "prompt", "completion")

    result = run_training_attempt_subprocess(
        attempt_number=1,
        run_id="run_1",
        model_id="org/model",
        revision="abc",
        run_dir=tmp_path,
        train_datasets={"json_extraction": [example]},
        eval_datasets={"json_extraction": [example]},
        peft_config={"method": "lora"},
        smoke_only=True,
        task_type="json_extraction",
    )

    assert observed["command"][:3] == [
        sys.executable,
        "-m",
        "hf_model_watch.finetune.attempt_worker",
    ]
    assert observed["kwargs"]["stderr"] is not None
    assert observed["request"]["peft_config"]["attempt_number"] == 1
    assert result["success"] is True
    assert result["attempt_number"] == 1


def test_optimizer_uses_configured_fallback_when_automatic_choice_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "hf_model_watch.finetune.trl_training._optimizer_available",
        lambda name: name == "paged_adamw_32bit",
    )

    optimizer, reason = _select_optimizer(
        {
            "optimizer": "paged_adamw_8bit",
            "optimizer_fallback": "paged_adamw_32bit",
        }
    )

    assert optimizer == "paged_adamw_32bit"
    assert "unavailable" in reason


def test_wandb_run_name_distinguishes_task_method_and_attempt() -> None:
    assert _wandb_run_name("run_1", "text_to_sql", "qlora", 2) == (
        "run_1-text_to_sql-qlora-attempt-2"
    )


def test_explicit_unavailable_optimizer_is_not_silently_changed(monkeypatch) -> None:
    monkeypatch.setattr(
        "hf_model_watch.finetune.trl_training._optimizer_available",
        lambda name: name == "adamw_torch",
    )

    with pytest.raises(RuntimeError, match="explicit optimizer is unavailable"):
        _select_optimizer(
            {
                "optimizer": "custom_optimizer",
                "optimizer_fallback": "adamw_torch",
                "explicit_overrides": ["optimizer"],
            }
        )
