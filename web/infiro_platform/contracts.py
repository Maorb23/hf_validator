"""Bounded web inputs, separate from the backward-compatible CLI contracts."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Requirements(Input):
    request: str = Field(min_length=3, max_length=12000)
    gpu_memory_gb: float | None = Field(default=None, gt=0, le=2048)
    budget_usd_per_hour: float | None = Field(default=None, gt=0, le=10000)
    gpu_hourly_cost_usd: float | None = Field(default=None, gt=0, le=1000)
    gpu_count: int = Field(default=1, ge=1, le=8)
    license: str = Field(default="any", max_length=200)
    framework_mode: Literal["fast", "sweep"] = "sweep"
    speculative_decoding: Literal["off", "prefer", "require"] = "prefer"


class DatasetConfig(Input):
    repo_id: str = Field(min_length=1, max_length=180)
    revision: str = Field(default="main", max_length=100)
    config: str | None = Field(default=None, max_length=100)
    split: str = Field(default="train", min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=180)
    model_revision: str = Field(default="main", max_length=100)
    num_prompts: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=1, ge=0, le=2**31 - 1)
    input_tokens: int = Field(default=512, ge=1, le=16384)
    output_tokens: int = Field(default=128, ge=1, le=8192)
    concurrency: int = Field(default=1, ge=1, le=256)

    @field_validator("repo_id")
    @classmethod
    def repository(cls, value):
        from llm_optimizer.hf_workload import dataset_id

        return dataset_id(value)

    @field_validator("model")
    @classmethod
    def model_id(cls, value):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", value) or ".." in value:
            raise ValueError("Use an HF model repository ID.")
        return value

    @model_validator(mode="after")
    def bounded_tokens(self):
        if self.num_prompts * (self.input_tokens + self.output_tokens) > 2_000_000:
            raise ValueError("Sample exceeds the 2,000,000 requested token limit.")
        for value in (self.revision, self.model_revision, self.config, self.split):
            if value and (not re.fullmatch(r"[\w./:\[\]%+-]+", value) or ".." in value):
                raise ValueError("Invalid revision/config/split.")
        return self


class Choices(Input):
    workload_mode: Literal["random", "sharegpt"] = "random"
    gpu_type: Literal["gpu-h100-sxm", "gpu-h200-sxm", "gpu-l40s-d"] = "gpu-h100-sxm"
    gpu_count: Literal[1, 8] = 1
    frameworks: list[Literal["vllm", "sglang", "trtllm"]] = Field(
        default_factory=lambda: ["vllm"], min_length=1, max_length=3
    )
    gpu_hourly_cost_usd: float = Field(default=1, gt=0, le=1000)
    budget_usd_per_hour: float = Field(default=5, gt=0, le=10000)
    spending_limit_usd: float = Field(default=20, gt=0, le=50000)
    timeout_minutes: int = Field(default=60, ge=60, le=240)
    trials: int = Field(default=10, ge=1, le=100)
    dry_run: bool = True
    num_prompts: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=1, ge=0, le=2**31 - 1)
    input_tokens: int = Field(default=512, ge=1, le=16384)
    output_tokens: int = Field(default=128, ge=1, le=8192)
    concurrency: int = Field(default=1, ge=1, le=256)
    speculative_decoding: Literal["off", "prefer", "require"] = "prefer"

    @model_validator(mode="after")
    def limits(self):
        if len(set(self.frameworks)) != len(self.frameworks):
            raise ValueError("Frameworks must be unique.")
        estimate = self.gpu_count * self.gpu_hourly_cost_usd * len(self.frameworks)
        if (
            estimate > self.budget_usd_per_hour
            or estimate * self.timeout_minutes / 60 > self.spending_limit_usd
        ):
            raise ValueError(
                "Estimated GPU allocation exceeds your hourly or total estimate limit."
            )
        if self.num_prompts * (self.input_tokens + self.output_tokens) > 2_000_000:
            raise ValueError("Sample exceeds the 2,000,000 requested token limit.")
        if self.gpu_type == "gpu-l40s-d" and self.gpu_count != 1:
            raise ValueError("The configured L40S preset supports one GPU.")
        return self
