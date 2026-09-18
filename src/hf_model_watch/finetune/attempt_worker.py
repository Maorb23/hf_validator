from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from hf_model_watch.finetune.datasets import AdaptationExample
from hf_model_watch.finetune.trl_training import _finish_wandb, run_peft_training


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m hf_model_watch.finetune.attempt_worker REQUEST.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result_path = Path(request["result_path"])
    try:
        result = run_peft_training(
            run_id=request["run_id"],
            model_id=request["model_id"],
            revision=request.get("revision"),
            run_dir=Path(request["run_dir"]),
            train_datasets=_datasets(request["train_datasets"]),
            eval_datasets=_datasets(request["eval_datasets"]),
            peft_config=request["peft_config"],
            smoke_only=bool(request.get("smoke_only")),
            task_type=request["task_type"],
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        is_oom = "out of memory" in message.lower()
        result = {
            "success": False,
            "retry_recommended": is_oom,
            "retry_reason": "cuda_oom" if is_oom else "training_error",
            "message": message,
            "traceback": traceback.format_exc(),
        }
    finally:
        _finish_wandb(bool(request.get("peft_config", {}).get("wandb_enabled", True)))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.get("success") else 2


def _datasets(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[AdaptationExample]]:
    return {
        task: [AdaptationExample(**example) for example in examples]
        for task, examples in data.items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
