"""Versioned semantic contracts; execution settings belong to the controller."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Metric = Literal["mean_e2e_latency_ms", "mean_ttft_ms", "output_throughput",
                 "p95_e2e_latency_ms", "p95_ttft_ms"]
Policy = Literal[
    "latency_first", "ttft_first", "throughput_first", "balanced", "knee_point", "constrained"
]
DIRECTIONS = {
    "mean_e2e_latency_ms": "minimize",
    "mean_ttft_ms": "minimize",
    "output_throughput": "maximize",
    "p95_e2e_latency_ms": "minimize",
    "p95_ttft_ms": "minimize",
}


def objectives_for(concurrency: int) -> list[Metric]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    return (
        ["mean_e2e_latency_ms", "mean_ttft_ms"]
        if concurrency == 1
        else ["output_throughput", "mean_e2e_latency_ms"]
    )


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1


class HFDataset(Contract):
    repo_id: str = Field(min_length=1)
    revision: str | None = None
    config: str | None = None
    split: str = Field(default="train", min_length=1)
    data_file: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator("data_file")
    @classmethod
    def relative_data_file(cls, value):
        import re

        if value is not None and (
            not re.fullmatch(r"[\w./-]+", value)
            or value.startswith("/")
            or ".." in value.split("/")
            or not value.lower().endswith(".json")
        ):
            raise ValueError("Use a relative JSON file path from the HF dataset repository")
        return value

    @model_validator(mode="after")
    def normalize(self):
        from llm_optimizer.hf_workload import dataset_id
        self.repo_id = dataset_id(self.repo_id)
        if self.data_file and self.config:
            raise ValueError("Choose a dataset configuration or one data file, not both")
        return self


class Workload(Contract):
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    num_prompts: int = Field(default=100, gt=0)
    seed: int = 1
    dataset: HFDataset | None = None
    workload_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class Bound(Contract):
    metric: Metric
    operator: Literal["<=", ">="]
    value: float = Field(ge=0)


class StudyContext(Contract):
    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    gpu_type: str = Field(pattern=r"^gpu-")
    gpu_count: int = Field(default=1, gt=0)
    workload: Workload
    concurrency: int = Field(gt=0)
    objectives: list[Metric] = Field(default_factory=list)
    constraints: list[Bound] = Field(default_factory=list)

    @model_validator(mode="after")
    def fixed_objectives(self) -> StudyContext:
        expected = objectives_for(self.concurrency)
        if not self.objectives:
            self.objectives = expected
        else:
            from llm_optimizer.optimization import objective_names

            objective_names(",".join(self.objectives))
        return self


class SelectionPolicy(Contract):
    name: Policy = "balanced"
    weights: dict[Metric, float] = Field(default_factory=dict)
    constraints: list[Bound] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_weights(self) -> SelectionPolicy:
        if any(value <= 0 for value in self.weights.values()):
            raise ValueError("Weights must be positive")
        if self.name == "constrained" and not self.constraints:
            raise ValueError("Constrained selection requires bounds")
        return self


class NewStudyIntent(Contract):
    mode: Literal["new"] = "new"
    context: StudyContext
    frameworks: list[str] = Field(min_length=1)
    selection_policy: SelectionPolicy = Field(default_factory=SelectionPolicy)
    idempotency_key: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique_frameworks(self) -> NewStudyIntent:
        if len(set(self.frameworks)) != len(self.frameworks) or any(not x for x in self.frameworks):
            raise ValueError("Frameworks must be nonempty and unique")
        return self


class ReuseIntent(Contract):
    mode: Literal["reuse"] = "reuse"
    study_reference: str = Field(min_length=1)
    context: StudyContext
    selection_policy: SelectionPolicy = Field(default_factory=SelectionPolicy)


class Candidate(Contract):
    configuration_id: str
    framework: str
    configuration: dict[str, Any]
    metrics: dict[str, float]
    artifacts: dict[str, str] = Field(default_factory=dict)
    validation_status: str = "not_independently_validated"
    trial_id: str | None = None
    timings: dict[str, float | None] = Field(default_factory=dict)


class TrialRecord(Contract):
    framework: str
    trial_id: str
    status: str
    configuration: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float | None] = Field(default_factory=dict)
    reason: str | None = None
    configuration_id: str | None = None


class StudySummary(Contract):
    study_reference: str
    context: StudyContext
    state: str
    trial_state_counts: dict[str, dict[str, int]] = Field(default_factory=dict)
    gpu_seconds: dict[str, float] = Field(default_factory=dict)
    failed_trials: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list)
    trials: list[TrialRecord] = Field(default_factory=list)
    framework_timings: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    framework_errors: dict[str, str] = Field(default_factory=dict)
    importance_per_objective: dict[str, dict[str, Any]] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class OptimizationReference(Contract):
    study_reference: str
    mode: Literal["new", "reuse"]
    intent: NewStudyIntent | ReuseIntent


class OptimizationResult(Contract):
    study_reference: str
    policy: SelectionPolicy
    selected: Candidate | None = None
    reason: str | None = None
    pareto_front: list[Candidate] = Field(default_factory=list)
    best_per_objective: dict[str, Candidate] = Field(default_factory=dict)
    importance_per_objective: dict[str, dict[str, Any]] = Field(default_factory=dict)
    framework_errors: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
