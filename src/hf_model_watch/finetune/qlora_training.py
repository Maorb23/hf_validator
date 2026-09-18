from __future__ import annotations

from pathlib import Path
from typing import Any

from hf_model_watch.finetune.datasets import AdaptationExample
from hf_model_watch.finetune.trl_training import run_peft_training


def run_qlora_training(
    run_id: str,
    model_id: str,
    revision: str | None,
    run_dir: Path,
    train_datasets: dict[str, list[AdaptationExample]],
    eval_datasets: dict[str, list[AdaptationExample]],
    peft_config: dict[str, Any],
    smoke_only: bool = False,
    task_type: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible entry point for the method-neutral TRL trainer."""
    return run_peft_training(
        run_id=run_id,
        model_id=model_id,
        revision=revision,
        run_dir=run_dir,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        peft_config=peft_config,
        smoke_only=smoke_only,
        task_type=task_type,
    )
