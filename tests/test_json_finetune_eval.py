from __future__ import annotations

from hf_model_watch.finetune.policy import (
    DatasetProfile,
    HardwareProfile,
    resolve_finetune_policy,
)
from hf_model_watch.finetune.trl_training import (
    _evaluation_generation_budget,
    _parse_first_json_value,
    _score_prediction,
)


class CharacterTokenizer:
    def __call__(self, text: str, **kwargs):
        return {"input_ids": list(range(len(text)))}


def test_compact_json_uses_tokenizer_based_generation_budget() -> None:
    expected = '{"name":"Ada","city":"London"}'

    budget = _evaluation_generation_budget(
        "json_extraction",
        CharacterTokenizer(),
        expected,
        {},
    )

    assert budget == len(expected) * 2
    assert budget > len(expected.split()) * 3


def test_json_generation_budget_is_configurable_and_capped() -> None:
    budget = _evaluation_generation_budget(
        "json_extraction",
        CharacterTokenizer(),
        '{"long":"' + "x" * 200 + '"}',
        {
            "json_eval_min_new_tokens": 32,
            "json_eval_max_new_tokens": 80,
            "json_eval_generation_token_multiplier": 3,
        },
    )

    assert budget == 80


def test_non_json_generation_budget_keeps_existing_behavior() -> None:
    budget = _evaluation_generation_budget(
        "text_to_sql",
        CharacterTokenizer(),
        "SELECT * FROM users",
        {},
    )

    assert budget == 12


def test_resolved_policy_preserves_generation_budget_settings() -> None:
    policy = resolve_finetune_policy(
        model_size_billion=1,
        architecture="LlamaForCausalLM",
        context_length=2048,
        dataset=DatasetProfile(
            task="json_extraction",
            task_breadth="narrow",
            examples=100,
            total_tokens=10_000,
            mean_tokens=100,
            p95_tokens=256,
            p99_tokens=300,
        ),
        hardware=HardwareProfile(
            gpu_name="test",
            total_vram_bytes=80 * 2**30,
            bf16_supported=True,
        ),
        config={
            "json_eval_min_new_tokens": 96,
            "json_eval_max_new_tokens": 320,
            "json_eval_generation_token_multiplier": 2.5,
        },
    )

    resolved = policy.as_dict()
    assert resolved["json_eval_min_new_tokens"] == 96
    assert resolved["json_eval_max_new_tokens"] == 320
    assert resolved["json_eval_generation_token_multiplier"] == 2.5


def test_json_parser_accepts_fenced_output_and_ignores_trailing_text() -> None:
    prediction = 'Result:\n```json\n{"city":"London","name":"Ada"}\n```\nDone {later}'

    assert _parse_first_json_value(prediction) == {"name": "Ada", "city": "London"}
    assert (
        _score_prediction(
            "json_extraction",
            prediction,
            '{"name":"Ada","city":"London"}',
        )
        == 1.0
    )


def test_json_score_rejects_truncated_or_incorrect_output() -> None:
    assert _score_prediction("json_extraction", '{"name":"Ada"', '{"name":"Ada"}') == 0.0
    assert _score_prediction("json_extraction", '{"name":"Grace"}', '{"name":"Ada"}') == 0.0
