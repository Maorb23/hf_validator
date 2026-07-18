from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_FILES = (
    "scan.yaml",
    "eval_tasks.yaml",
    "finetune.yaml",
    "serverless.yaml",
    "storage.yaml",
    "scoring.yaml",
)


class ScanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback_days: int = Field(gt=0)
    max_models_to_scan: int = Field(gt=0)
    max_candidates_to_run: int = Field(gt=0)
    pipeline_tags: list[str] = Field(default_factory=lambda: ["text-generation"])
    max_params_billion: float = Field(gt=0)
    allow_gated: bool
    allow_trust_remote_code: bool
    require_safetensors: bool
    allowed_licenses: list[str]
    blocked_licenses: list[str]
    max_download_gb: float = Field(gt=0)


class ServerlessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str | None = None
    image: str | None = None
    preset: str | None = None
    gpu_type: str
    gpu_memory_gb: float = Field(gt=0)
    memory_safety_margin: float = Field(gt=0, le=1)
    gpu_count: int = Field(gt=0)
    timeout_minutes: int = Field(gt=0)
    max_parallel_jobs: int = Field(gt=0)
    max_params_billion: float = Field(gt=0)
    max_download_gb: float = Field(gt=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    scan: ScanConfig
    eval_tasks: dict[str, Any]
    finetune: dict[str, Any]
    serverless: ServerlessConfig
    storage: dict[str, Any]
    scoring: dict[str, Any]


def load_config(config_dir: str | Path = "config") -> AppConfig:
    """Load, merge, and validate all Phase 1 YAML config files."""
    config_path = Path(config_dir)
    merged: dict[str, Any] = {}

    for file_name in CONFIG_FILES:
        path = config_path / file_name
        if not path.exists():
            raise FileNotFoundError(f"Required config file is missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config file must contain a mapping: {path}")
        merged[Path(file_name).stem] = data

    _apply_env_overrides(merged)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {config_path}") from exc


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load one YAML file for CLI commands that accept a specific config path."""
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {yaml_path}")
    return data


def _apply_env_overrides(config: dict[str, Any]) -> None:
    scan = config["scan"]
    storage = config["storage"]

    _set_bool(scan, "allow_gated", "HF_MODEL_WATCH_ALLOW_GATED")
    _set_bool(scan, "allow_trust_remote_code", "HF_MODEL_WATCH_ALLOW_TRUST_REMOTE_CODE")
    _set_int(scan, "max_models_to_scan", "HF_MODEL_WATCH_MAX_MODELS_TO_SCAN")
    _set_int(scan, "max_candidates_to_run", "HF_MODEL_WATCH_MAX_CANDIDATES_TO_RUN")
    _set_float(scan, "max_params_billion", "HF_MODEL_WATCH_MAX_PARAMS_BILLION")

    project_id = os.getenv("NB_PROJECT_ID") or os.getenv("NEBIUS_PROJECT_ID")
    if project_id:
        config["serverless"]["project_id"] = project_id

    image = os.getenv("HF_MODEL_WATCH_NEBIUS_IMAGE") or os.getenv("NEBIUS_JOB_IMAGE")
    if image:
        config["serverless"]["image"] = image

    platform = os.getenv("HF_MODEL_WATCH_NEBIUS_PLATFORM") or os.getenv("NEBIUS_JOB_PLATFORM")
    if platform:
        config["serverless"]["gpu_type"] = platform

    preset = os.getenv("HF_MODEL_WATCH_NEBIUS_PRESET") or os.getenv("NEBIUS_JOB_PRESET")
    if preset:
        config["serverless"]["preset"] = preset

    storage_uri = os.getenv("HF_MODEL_WATCH_STORAGE_URI")
    if storage_uri:
        storage["local_artifact_dir"] = storage_uri


def _set_bool(target: dict[str, Any], key: str, env_name: str) -> None:
    raw = os.getenv(env_name)
    if raw is None:
        return
    target[key] = raw.strip().lower() in {"1", "true", "yes", "on"}


def _set_int(target: dict[str, Any], key: str, env_name: str) -> None:
    raw = os.getenv(env_name)
    if raw is not None:
        target[key] = int(raw)


def _set_float(target: dict[str, Any], key: str, env_name: str) -> None:
    raw = os.getenv(env_name)
    if raw is not None:
        target[key] = float(raw)
