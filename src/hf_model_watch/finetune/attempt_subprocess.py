from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hf_model_watch.finetune.datasets import AdaptationExample


def run_training_attempt_subprocess(
    *,
    attempt_number: int,
    run_id: str,
    model_id: str,
    revision: str | None,
    run_dir: Path,
    train_datasets: dict[str, list[AdaptationExample]],
    eval_datasets: dict[str, list[AdaptationExample]],
    peft_config: dict[str, Any],
    smoke_only: bool,
    task_type: str,
) -> dict[str, Any]:
    attempt_dir = run_dir / "attempts" / f"attempt_{attempt_number}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    request_path = attempt_dir / "request.json"
    result_path = attempt_dir / "result.json"
    log_path = attempt_dir / "process.log"
    attempt_config = dict(peft_config)
    attempt_config["attempt_number"] = attempt_number
    request = {
        "run_id": run_id,
        "model_id": model_id,
        "revision": revision,
        "run_dir": str(run_dir),
        "train_datasets": {
            task: [asdict(example) for example in examples]
            for task, examples in train_datasets.items()
        },
        "eval_datasets": {
            task: [asdict(example) for example in examples]
            for task, examples in eval_datasets.items()
        },
        "peft_config": attempt_config,
        "smoke_only": smoke_only,
        "task_type": task_type,
        "result_path": str(result_path),
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "hf_model_watch.finetune.attempt_worker", str(request_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
    return_code = process.wait()
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["attempt_number"] = attempt_number
        result["attempt_log"] = str(log_path)
        return result
    combined = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    is_oom = "out of memory" in combined.lower()
    return {
        "success": False,
        "retry_recommended": is_oom,
        "retry_reason": "cuda_oom" if is_oom else "attempt_process_failed",
        "message": f"training attempt process exited with code {return_code}",
        "attempt_number": attempt_number,
        "attempt_log": str(log_path),
    }
