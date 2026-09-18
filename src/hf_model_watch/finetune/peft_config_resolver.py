from __future__ import annotations

from typing import Any

from hf_model_watch.finetune.policy import (
    DatasetProfile,
    HardwareProfile,
    ResolvedFinetunePolicy,
    resolve_finetune_policy,
)

PeftResolvedConfig = ResolvedFinetunePolicy


def resolve_peft_config(
    model_size_billion: float,
    architecture: str | list[str] | None,
    available_vram_gb: float,
    task: str,
    desired_budget: str = "standard",
    context_length: int = 2048,
    quantization_mode: str | None = None,
    finetune_config: dict[str, Any] | None = None,
    dataset_profile: DatasetProfile | None = None,
) -> ResolvedFinetunePolicy:
    """Compatibility wrapper around the dataset- and memory-aware policy resolver."""
    cfg = dict(finetune_config or {})
    if quantization_mode == "4bit" and "method" not in cfg:
        cfg["method"] = "qlora"
    profile = dataset_profile or DatasetProfile(
        task=task,
        task_breadth=(
            "domain" if task in {"finance_analysis", "financial_analyst_style"} else "narrow"
        ),
        examples=0,
        total_tokens=0,
        mean_tokens=128,
        p95_tokens=min(context_length, 2048),
        p99_tokens=min(context_length, 2048),
    )
    policy = resolve_finetune_policy(
        model_size_billion=model_size_billion,
        architecture=architecture,
        context_length=context_length,
        dataset=profile,
        hardware=HardwareProfile(
            gpu_name="configured GPU",
            total_vram_bytes=int(available_vram_gb * 2**30),
            bf16_supported=bool(cfg.get("bf16", True)),
        ),
        config=cfg,
    )
    if desired_budget != "standard":
        object.__setattr__(
            policy,
            "reasons",
            (*policy.reasons, f"requested budget={desired_budget}"),
        )
    return policy


def resolve_target_modules(
    model: Any | None,
    architecture: str | list[str] | None,
    requested: str | list[str] = "all-linear",
) -> str | list[str]:
    """Return PEFT-native targets without fragile suffix-based discovery."""
    del model, architecture
    return requested


def discover_linear_module_names(model: Any) -> list[str]:
    """Retained for callers that explicitly need full module paths for diagnostics."""
    linear_class_names = {"Linear", "Linear4bit", "Linear8bitLt", "Conv1D"}
    return sorted(
        name
        for name, module in model.named_modules()
        if name and module.__class__.__name__ in linear_class_names
    )
