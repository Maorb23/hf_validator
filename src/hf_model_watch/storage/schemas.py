from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

RUN_MANIFEST_FILENAME = "run_manifest.json"
TRACE_EVENTS_FILENAME = "trace_events.jsonl"
CLAIMS_VS_OBSERVED_PATH = Path("metadata") / "claims_vs_observed.json"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id() -> str:
    return f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class FailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_category: str
    failure_message: str


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    model_id: str
    revision: str | None = None
    started_at: str = Field(default_factory=utc_now_iso)
    completed_at: str | None = None
    status: str = "started"
    artifacts: dict[str, str] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    failure: FailureRecord | None = None


def run_artifact_dir(base_dir: str | Path, run_id: str) -> Path:
    return Path(base_dir) / run_id


def initialize_run_artifacts(base_dir: str | Path, manifest: RunManifest) -> Path:
    root = run_artifact_dir(base_dir, manifest.run_id)
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(root, manifest)
    (root / TRACE_EVENTS_FILENAME).touch(exist_ok=True)
    claims_path = root / CLAIMS_VS_OBSERVED_PATH
    if not claims_path.exists():
        claims_path.write_text(json.dumps({}, sort_keys=True), encoding="utf-8")
    return root


def write_run_manifest(run_dir: str | Path, manifest: RunManifest) -> Path:
    path = Path(run_dir) / RUN_MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_run_manifest(path: str | Path) -> RunManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunManifest.model_validate(data)


def update_manifest_artifact(
    run_dir: str | Path,
    manifest: RunManifest,
    artifact_name: str,
    artifact_uri: str,
) -> RunManifest:
    updated = manifest.model_copy(
        update={"artifacts": {**manifest.artifacts, artifact_name: artifact_uri}}
    )
    write_run_manifest(run_dir, updated)
    return updated


def build_reproducibility_snapshot(
    model_id: str,
    revision: str | None,
    config_snapshot: dict[str, Any] | None = None,
    runtime_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cuda = _cuda_snapshot()
    finetune = (config_snapshot or {}).get("finetune") if config_snapshot else {}
    if not isinstance(finetune, dict):
        finetune = {}
    return {
        "model_id": model_id,
        "revision": revision,
        "package_versions": _package_versions(),
        "cuda_available": cuda["available"],
        "gpu_name": cuda["gpu_name"],
        "selected_backend": (runtime_decision or {}).get("backend"),
        "selected_dtype": (runtime_decision or {}).get("dtype"),
        "selected_quantization": (runtime_decision or {}).get("quantization"),
        "random_seed": finetune.get("seed"),
        "captured_at": utc_now_iso(),
    }


def _package_versions() -> dict[str, str | None]:
    packages = [
        "hf-model-watch",
        "huggingface-hub",
        "transformers",
        "torch",
        "accelerate",
        "safetensors",
        "tokenizers",
        "pydantic",
        "typer",
        "pyyaml",
        "peft",
        "datasets",
        "boto3",
        "prefect",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _cuda_snapshot() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "gpu_name": None}
    available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if available else None
    return {"available": available, "gpu_name": gpu_name}
