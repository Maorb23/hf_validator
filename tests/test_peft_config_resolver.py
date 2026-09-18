from __future__ import annotations

import pytest

from hf_model_watch.finetune.datasets import load_all_adaptation_datasets
from hf_model_watch.finetune.peft_config_resolver import (
    discover_linear_module_names,
    resolve_peft_config,
    resolve_target_modules,
)
from hf_model_watch.finetune.policy import (
    DatasetProfile,
    HardwareProfile,
    estimate_peak_vram_bytes,
    resolve_finetune_policy,
)


def test_auto_policy_prefers_lora_when_bf16_has_headroom() -> None:
    cfg = resolve_peft_config(3, "LlamaForCausalLM", 80, "json_extraction").as_dict()

    assert cfg["method"] == "lora"
    assert cfg["r"] == 16
    assert cfg["alpha"] == 16
    assert cfg["use_rslora"] is True
    assert cfg["target_modules"] == "all-linear"


def test_auto_policy_uses_qlora_when_lora_estimate_is_too_large() -> None:
    cfg = resolve_peft_config(40, "MistralForCausalLM", 80, "finance_analysis").as_dict()

    assert cfg["method"] == "qlora"
    assert cfg["bnb_4bit_quant_type"] == "nf4"
    assert cfg["bnb_4bit_use_double_quant"] is True


def test_moe_active_parameters_reduce_activation_estimate_not_weight_estimate() -> None:
    dense_peak = estimate_peak_vram_bytes(
        model_size_billion=40,
        method="qlora",
        sequence_length=2048,
        microbatch_size=4,
        rank=16,
        gradient_checkpointing=True,
    )
    moe_peak = estimate_peak_vram_bytes(
        model_size_billion=40,
        activation_model_size_billion=10,
        method="qlora",
        sequence_length=2048,
        microbatch_size=4,
        rank=16,
        gradient_checkpointing=True,
    )

    assert moe_peak < dense_peak
    assert moe_peak > 40_000_000_000 * 0.65


def test_explicit_overrides_are_preserved() -> None:
    profile = DatasetProfile("task", "narrow", 500, 5000, 10, 64, 128)
    policy = resolve_finetune_policy(
        model_size_billion=30,
        architecture="LlamaForCausalLM",
        context_length=8192,
        dataset=profile,
        hardware=HardwareProfile("H100", 80 * 2**30, True),
        config={
            "method": "lora",
            "lora_rank": 32,
            "lora_alpha": 64,
            "learning_rate": 2e-4,
            "microbatch_size": 3,
            "optimizer": "adamw_torch",
        },
    )

    assert policy.method == "lora"
    assert policy.rank == 32
    assert policy.alpha == 64
    assert policy.learning_rate == 2e-4
    assert policy.microbatch_size == 3
    assert policy.optimizer == "adamw_torch"
    assert policy.allow_method_fallback is False
    assert policy.allow_batch_fallback is False


def test_token_target_drives_gradient_accumulation() -> None:
    profile = DatasetProfile("task", "general", 10_000, 2_560_000, 256, 512, 768)
    policy = resolve_finetune_policy(
        model_size_billion=8,
        architecture="Qwen2ForCausalLM",
        context_length=4096,
        dataset=profile,
        hardware=HardwareProfile("H100", 80 * 2**30, True),
        config={"method": "qlora", "microbatch_size": 4, "target_tokens_per_update": 32768},
    )

    assert policy.gradient_accumulation == 32
    assert policy.effective_batch_size == 128


def test_monitoring_defaults_are_resolved_and_eval_frequency_is_bounded() -> None:
    profile = DatasetProfile("task", "narrow", 500, 5000, 10, 64, 128)
    policy = resolve_finetune_policy(
        model_size_billion=1,
        architecture="LlamaForCausalLM",
        context_length=2048,
        dataset=profile,
        hardware=HardwareProfile("H100", 80 * 2**30, True),
        config={"max_train_steps": 3, "eval_steps": 25},
    )

    assert policy.eval_steps == 3
    assert policy.eval_microbatch_size == 1
    assert policy.logging_steps == 1
    assert policy.wandb_enabled is True
    assert policy.wandb_project == "hf-model-watch"


def test_method_fallback_preserves_explicit_batch_override() -> None:
    profile = DatasetProfile("task", "narrow", 500, 5000, 10, 64, 128)
    hardware = HardwareProfile("H100", 80 * 2**30, True)
    config = {"method": "auto", "microbatch_size": 4}
    first = resolve_finetune_policy(
        model_size_billion=1,
        architecture="LlamaForCausalLM",
        context_length=2048,
        dataset=profile,
        hardware=hardware,
        config=config,
    )
    retry = resolve_finetune_policy(
        model_size_billion=1,
        architecture="LlamaForCausalLM",
        context_length=2048,
        dataset=profile,
        hardware=hardware,
        config=config,
        retry_of=first,
    )

    assert first.method == "lora"
    assert retry.method == "qlora"
    assert retry.microbatch_size == 4


def test_forced_lora_never_silently_changes_to_qlora() -> None:
    profile = DatasetProfile("task", "narrow", 500, 5000, 10, 64, 128)
    hardware = HardwareProfile("H100", 80 * 2**30, True)
    first = resolve_finetune_policy(
        model_size_billion=1,
        architecture="LlamaForCausalLM",
        context_length=2048,
        dataset=profile,
        hardware=hardware,
        config={"method": "lora"},
    )
    retry = resolve_finetune_policy(
        model_size_billion=1,
        architecture="LlamaForCausalLM",
        context_length=2048,
        dataset=profile,
        hardware=hardware,
        config={"method": "lora"},
        retry_of=first,
    )

    assert retry.method == "lora"
    assert retry.microbatch_size == 1


def test_forced_lora_and_batch_block_automatic_fallback() -> None:
    profile = DatasetProfile("task", "narrow", 500, 5000, 10, 64, 128)
    hardware = HardwareProfile("H100", 80 * 2**30, True)
    config = {"method": "lora", "microbatch_size": 4}
    first = resolve_finetune_policy(
        model_size_billion=1,
        architecture="LlamaForCausalLM",
        context_length=2048,
        dataset=profile,
        hardware=hardware,
        config=config,
    )

    with pytest.raises(ValueError, match="explicit batch override"):
        resolve_finetune_policy(
            model_size_billion=1,
            architecture="LlamaForCausalLM",
            context_length=2048,
            dataset=profile,
            hardware=hardware,
            config=config,
            retry_of=first,
        )


def test_all_linear_is_passed_directly_to_peft() -> None:
    assert resolve_target_modules(None, "GPT2LMHeadModel", "all-linear") == "all-linear"


def test_module_discovery_uses_full_paths_for_diagnostics() -> None:
    class Linear:
        pass

    class MockModel:
        def named_modules(self):
            return [("layers.0.q_proj", Linear()), ("layers.0.v_proj", Linear())]

    assert discover_linear_module_names(MockModel()) == [
        "layers.0.q_proj",
        "layers.0.v_proj",
    ]


def test_local_synthetic_datasets_load() -> None:
    datasets = load_all_adaptation_datasets()

    assert set(datasets) == {"json_extraction", "text_to_sql", "financial_analyst_style"}
    assert all(datasets.values())
