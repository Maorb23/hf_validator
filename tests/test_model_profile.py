from __future__ import annotations

from hf_model_watch.finetune.model_profile import count_unique_parameters, profile_model_config


def test_dense_model_profile_uses_total_as_active() -> None:
    profile = profile_model_config(
        {"model_type": "llama", "hidden_size": 4096},
        estimated_params_billion=8,
    )

    assert profile.is_moe is False
    assert profile.total_parameters == 8_000_000_000
    assert profile.active_parameters_estimate == profile.total_parameters


def test_moe_profile_reports_total_and_active_parameters() -> None:
    profile = profile_model_config(
        {
            "architectures": ["MixtralForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        },
        estimated_params_billion=47,
    )

    assert profile.is_moe is True
    assert profile.total_experts == 8
    assert profile.active_experts_per_token == 2
    assert profile.active_parameters_estimate is not None
    assert profile.active_parameters_estimate < profile.total_parameters


def test_parameter_counter_does_not_double_count_shared_objects() -> None:
    class Parameter:
        def numel(self):
            return 10

    shared = Parameter()

    class Model:
        def parameters(self):
            return [shared, shared, Parameter()]

    assert count_unique_parameters(Model()) == 20


def test_moe_is_detected_even_when_active_expert_count_is_missing() -> None:
    profile = profile_model_config(
        {"model_type": "custom_sparse", "num_experts": 16, "moe_intermediate_size": 4096},
        estimated_params_billion=30,
    )

    assert profile.is_moe is True
    assert profile.confidence == "medium"
    assert profile.active_parameters_estimate is None
