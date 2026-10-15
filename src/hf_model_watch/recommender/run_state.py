from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hf_model_watch.storage.schemas import utc_now_iso

STATE_FILENAME = "recommendation_run.json"
FEEDBACK_DIRNAME = "feedback"
MAX_REQUEST_CHARS = 20_000
MAX_FEEDBACK_CHARS = 10_000
_RUN_ID = re.compile(r"run_[A-Za-z0-9_-]+")


def questions_for(missing: list[str]) -> list[dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {
        "request": {
            "prompt": "Describe the inference workload and model requirements.",
            "type": "string",
        },
        "gpu_memory_gb": {
            "prompt": "How much memory is available on each GPU, in GB?",
            "type": "number",
            "minimum": 0,
            "exclusive_minimum": True,
        },
        "budget_usd_per_hour": {
            "prompt": "What is the maximum hourly inference budget in USD?",
            "type": "number",
            "minimum": 0,
            "exclusive_minimum": True,
        },
        "gpu_hourly_cost_usd": {
            "prompt": "What hourly GPU cost should be used in the budget check, in USD?",
            "type": "number",
            "minimum": 0,
            "exclusive_minimum": True,
        },
        "license": {
            "prompt": "Which model license is required?",
            "type": "string",
            "choices": ["any", "apache-2.0", "mit", "llama", "gemma"],
        },
        "speculative_decoding": {
            "prompt": "Prefer speculative decoding when supported? [Y/n]",
            "type": "string",
            "choices": ["prefer", "off", "require"],
            "default": "prefer",
        },
        "framework_mode": {
            "prompt": "Use fast mode or sweep mode?",
            "type": "string",
            "choices": ["fast", "sweep"],
        },
    }
    return [{"field": field, **definitions[field]} for field in missing]


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid recommendation run ID")
    return run_id


def run_dir(root: Path, run_id: str) -> Path:
    return root / validate_run_id(run_id)


def load_run(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    directory = run_dir(root, run_id)
    path = directory / STATE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Recommendation run not found: {run_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("run_id") != run_id:
        raise ValueError("Recommendation run identity does not match its path")
    return directory, data


def save_run(directory: Path, data: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    now = utc_now_iso()
    persisted = {
        **data,
        "created_at": data.get("created_at") or now,
        "updated_at": now,
    }
    payload = json.dumps(persisted, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".recommendation_run.", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory / STATE_FILENAME)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return persisted


@contextmanager
def locked_run(root: Path, run_id: str) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Lock one run while applying a read/validate/write transition."""
    import fcntl

    directory = run_dir(root, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".state.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield load_run(root, run_id)


def add_feedback(
    directory: Path,
    run: dict[str, Any],
    *,
    model_id: str,
    category: str,
    details: str,
    source: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not details.strip():
        raise ValueError("Feedback details must not be empty")
    if len(details) > MAX_FEEDBACK_CHARS:
        raise ValueError(f"Feedback details exceed {MAX_FEEDBACK_CHARS} characters")
    candidates = {row.get("model_id") for row in run.get("recommendations", [])}
    if model_id not in candidates:
        raise ValueError("Feedback model must belong to the recommendation run")
    feedback_id = f"fb_{uuid.uuid4().hex}"
    record = {
        "feedback_id": feedback_id,
        "run_id": run["run_id"],
        "model_id": model_id,
        "category": category,
        "details": details,
        "source": source,
        "context": context or {},
        "feedback_status": "reported",
        "created_at": utc_now_iso(),
    }
    folder = directory / FEEDBACK_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{feedback_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def load_feedback(directory: Path, feedback_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"fb_[0-9a-f]{32}", feedback_id):
        raise ValueError("Invalid feedback ID")
    path = directory / FEEDBACK_DIRNAME / f"{feedback_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Feedback not found: {feedback_id}")
    return json.loads(path.read_text(encoding="utf-8"))
