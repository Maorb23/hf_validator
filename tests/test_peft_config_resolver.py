from __future__ import annotations

from hf_model_watch.finetune.datasets import load_all_adaptation_datasets
from hf_model_watch.finetune.peft_config_resolver import (
    discover_linear_module_names,
    resolve_peft_config,
    resolve_target_modules,
)


def test_peft_resolver_chooses_sane_config_for_3b() -> None:
    cfg = resolve_peft_config(3, "LlamaForCausalLM", 80, "json_extraction").as_dict()

    assert cfg["r"] == 16
    assert cfg["alpha"] == 32
    assert cfg["effective_batch_size"] == 32
    assert cfg["optimizer"] in {"adamw_torch_fused", "paged_adamw_8bit"}


def test_peft_resolver_chooses_sane_config_for_8b() -> None:
    cfg = resolve_peft_config(8, "Qwen2ForCausalLM", 80, "text_to_sql").as_dict()

    assert cfg["method"] == "qlora"
    assert cfg["optimizer"] == "paged_adamw_8bit"
    assert cfg["effective_batch_size"] == 16


def test_peft_resolver_chooses_sane_config_for_20b() -> None:
    cfg = resolve_peft_config(20, "MistralForCausalLM", 80, "finance_analysis").as_dict()

    assert cfg["method"] == "qlora"
    assert cfg["r"] in {8, 16}
    assert cfg["effective_batch_size"] in {8, 16}


def test_target_module_fallback_works_on_mock_modules() -> None:
    class Linear:
        pass

    class MockModel:
        def named_modules(self):
            return [("layers.0.q_proj", Linear()), ("layers.0.v_proj", Linear())]

    modules = discover_linear_module_names(MockModel())

    assert modules == ["q_proj", "v_proj"]
    assert resolve_target_modules(MockModel(), "Unknown", "all-linear") == ["q_proj", "v_proj"]


def test_architecture_target_module_fallback() -> None:
    modules = resolve_target_modules(None, "GPT2LMHeadModel", "all-linear")

    assert "c_attn" in modules


def test_local_synthetic_datasets_load() -> None:
    datasets = load_all_adaptation_datasets()

    assert set(datasets) == {"json_extraction", "text_to_sql", "financial_analyst_style"}
    assert all(datasets.values())
