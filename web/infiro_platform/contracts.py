"""Bounded web inputs, separate from the backward-compatible CLI contracts."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DirectModel(Input):
    model_id: str = Field(min_length=1, max_length=180)
    revision: str = Field(default="main", min_length=1, max_length=100)

    @field_validator("model_id")
    @classmethod
    def repository(cls, value):
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(value)
        return value

    @field_validator("revision")
    @classmethod
    def safe_revision(cls, value):
        if not re.fullmatch(r"[\w./-]+", value) or ".." in value:
            raise ValueError("Use a Hugging Face branch, tag or commit revision.")
        return value


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
    format: Literal["sharegpt", "alpaca", "messages"] = "sharegpt"
    repo_id: str = Field(min_length=1, max_length=180)
    revision: str = Field(default="main", max_length=100)
    config: str | None = Field(default=None, max_length=100)
    data_file: str | None = Field(default=None, max_length=300)
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
        from .dataset_source import dataset_id

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
        if self.data_file:
            if (
                not re.fullmatch(r"[\w./-]+", self.data_file)
                or self.data_file.startswith("/")
                or ".." in self.data_file.split("/")
            ):
                raise ValueError("Use a relative data file path from the HF dataset repository.")
            if self.config:
                raise ValueError("Choose a dataset configuration or one data file, not both.")
            if not self.data_file.lower().endswith(".json"):
                raise ValueError("A selected data file must be JSON.")
        if not self.data_file:
            raise ValueError("Select the relative JSON file inside the dataset repository.")
        if self.config or self.split != "train":
            raise ValueError(
                "Split/configuration filtering is unavailable; select the JSON file "
                "containing the desired records instead."
            )
        return self


class Choices(Input):
    workload_mode: Literal["sharegpt"] = "sharegpt"
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
    objectives: list[Literal["output_throughput", "mean_e2e_latency_ms", "mean_ttft_ms",
                             "p95_e2e_latency_ms", "p95_ttft_ms"]] = Field(
        default_factory=list, max_length=3
    )
    sampler: Literal["TPESampler", "RandomSampler", "GridSampler", "GPSampler", "NSGAIISampler"] = (
        "TPESampler"
    )
    tpe_multivariate: bool = False
    pruner: Literal["NopPruner", "MedianPruner", "SuccessiveHalvingPruner", "HyperbandPruner"] = (
        "NopPruner"
    )
    dry_run: bool = True
    num_prompts: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=1, ge=0, le=2**31 - 1)
    input_tokens: int = Field(default=512, ge=1, le=16384)
    output_tokens: int = Field(default=128, ge=1, le=8192)
    concurrency: int = Field(default=1, ge=1, le=256)
    speculative_decoding: Literal["off", "prefer", "require"] = "prefer"

    @model_validator(mode="after")
    def limits(self):
        from hf_model_watch.optimization.contracts import StudyContext, Workload

        # Keep saved mean-based requests valid; new web requests use p95.
        if not self.objectives:
            self.objectives = (
                ["p95_e2e_latency_ms", "p95_ttft_ms"] if self.concurrency == 1
                else ["output_throughput", "p95_e2e_latency_ms"]
            )
        context = StudyContext(
            model_id="web/validation",
            revision="validation",
            gpu_type=self.gpu_type,
            concurrency=self.concurrency,
            workload=Workload(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
            objectives=self.objectives,
        )
        self.objectives = context.objectives
        if self.tpe_multivariate and self.sampler != "TPESampler":
            raise ValueError("Multivariate TPE requires TPESampler.")
        if self.pruner != "NopPruner" and len(self.objectives) != 1:
            raise ValueError("Pruning requires an unconstrained scalar objective.")
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
