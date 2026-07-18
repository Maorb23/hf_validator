from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class WorkerJobPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    revision: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    artifact_prefix: str = ""
    skip_finetune: bool = False
    finetune_smoke_only: bool = False


def load_worker_payload(path: str | Path) -> WorkerJobPayload:
    payload_path = Path(path)
    try:
        data = json.loads(payload_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid worker payload JSON: {payload_path}") from exc
    try:
        return WorkerJobPayload.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid worker payload: {payload_path}") from exc


def write_worker_payload(payload: WorkerJobPayload, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
