from __future__ import annotations

from types import SimpleNamespace

from hf_model_watch.hub.scan import (
    _apply_context_fallback,
    _candidate_from_model_info,
    _has_chat_template,
    _pipeline_allowed,
)


def test_explicit_multimodal_pipeline_is_not_overridden_by_text_generation_tag() -> None:
    candidate = {
        "pipeline_tag": "image-text-to-text",
        "tags": ["image-text-to-text", "text-generation"],
    }

    assert _pipeline_allowed(candidate, {"pipeline_tags": ["text-generation"]}) is False


def test_missing_pipeline_can_fall_back_to_repository_tags() -> None:
    candidate = {"pipeline_tag": None, "tags": ["text-generation"]}

    assert _pipeline_allowed(candidate, {"pipeline_tags": ["text-generation"]}) is True


def test_candidate_extracts_safetensors_parameter_count_and_embedded_chat_template() -> None:
    info = SimpleNamespace(
        id="org/model",
        author="org",
        tags=["text-generation", "license:apache-2.0"],
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=4 * 1024**3)],
        config={
            "architectures": ["Qwen2ForCausalLM"],
            "max_position_embeddings": 32768,
            "tokenizer_config": {"chat_template": "{{ messages }}"},
        },
        safetensors=SimpleNamespace(total=7_000_000_000),
        pipeline_tag="text-generation",
        downloads=100,
        likes=1,
        gated=False,
        private=False,
        sha="abc",
        card_data={"license": "apache-2.0"},
    )

    candidate = _candidate_from_model_info(info)

    assert candidate["estimated_params_billion"] == 7.0
    assert candidate["estimated_download_size_gb"] == 4.0
    assert candidate["chat_template_present"] is True
    assert candidate["context_length"] == 32768
    assert candidate["context_length_source"] == "metadata"


def test_standalone_chat_template_file_is_detected() -> None:
    assert _has_chat_template({}, {}, ["chat_template.jinja"]) is True


def test_missing_context_length_uses_configured_fallback() -> None:
    records = [{"model_id": "org/model", "context_length": None}]

    _apply_context_fallback(records, {"fallback_context_length": 2048})

    assert records[0]["context_length"] == 2048
    assert records[0]["context_length_source"] == "fallback"
