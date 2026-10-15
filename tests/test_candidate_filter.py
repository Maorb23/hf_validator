from __future__ import annotations

from hf_model_watch.errors import FailureCategory
from hf_model_watch.hub.candidate_filter import filter_candidate
import pytest


@pytest.mark.parametrize("architecture,pipeline", [
    ("Qwen3_5ForConditionalGeneration", "image-text-to-text"),
    ("Gemma3ForConditionalGeneration", "image-text-to-text"),
    ("Gemma4ForConditionalGeneration", "any-to-any"),
    ("Gemma4UnifiedForConditionalGeneration", "any-to-any"),
])
def test_text_generating_multimodal_wrapper(architecture, pipeline):
    candidate = _valid_candidate()
    candidate.update(pipeline_tag=pipeline, architecture=[architecture],
                     config={"architectures": [architecture]})
    assert filter_candidate(candidate, _scan_config()).eligible
    candidate["has_tokenizer"] = False
    assert FailureCategory.MISSING_TOKENIZER.value in filter_candidate(candidate, _scan_config()).blocking_reasons


def test_unrecognized_multimodal_wrapper_is_rejected():
    candidate = _valid_candidate()
    candidate.update(pipeline_tag="image-text-to-text", architecture=["VisionEncoderDecoderModel"],
                     config={"architectures": ["VisionEncoderDecoderModel"]})
    assert FailureCategory.NOT_CAUSAL_LM.value in filter_candidate(candidate, _scan_config()).blocking_reasons


def test_candidate_filter_blocks_missing_tokenizer() -> None:
    candidate = _valid_candidate()
    candidate["has_tokenizer"] = False

    result = filter_candidate(candidate, _scan_config())

    assert result.eligible is False
    assert FailureCategory.MISSING_TOKENIZER.value in result.blocking_reasons


def test_candidate_filter_blocks_too_large_model() -> None:
    candidate = _valid_candidate()
    candidate["estimated_params_billion"] = 70

    result = filter_candidate(candidate, _scan_config())

    assert result.eligible is False
    assert FailureCategory.TOO_LARGE_FOR_H100.value in result.blocking_reasons


def test_candidate_filter_blocks_gated_model_when_not_allowed() -> None:
    candidate = _valid_candidate()
    candidate["gated"] = True

    result = filter_candidate(candidate, _scan_config())

    assert result.eligible is False
    assert FailureCategory.GATED_NO_ACCESS.value in result.blocking_reasons


def test_candidate_filter_allows_valid_mock_causal_lm() -> None:
    result = filter_candidate(_valid_candidate(), _scan_config())

    assert result.eligible is True
    assert result.blocking_reasons == []


def test_candidate_filter_blocks_explicit_multimodal_pipeline() -> None:
    candidate = _valid_candidate()
    candidate["model_id"] = "google/gemma-4-E2B-it"
    candidate["pipeline_tag"] = "image-text-to-text"
    candidate["tags"] = ["image-text-to-text", "text-generation", "license:gemma"]

    result = filter_candidate(candidate, _scan_config())

    assert result.eligible is False
    assert FailureCategory.NOT_CAUSAL_LM.value in result.blocking_reasons


def test_candidate_filter_blocks_inference_only_checkpoint_formats() -> None:
    mlx_candidate = _valid_candidate()
    mlx_candidate["model_id"] = "org/model-MLX-6bit"
    compressed_candidate = _valid_candidate()
    compressed_candidate["tags"] = ["text-generation", "compressed-tensors"]

    mlx_result = filter_candidate(mlx_candidate, _scan_config())
    compressed_result = filter_candidate(compressed_candidate, _scan_config())

    assert FailureCategory.UNSUPPORTED_FINETUNE_FORMAT.value in mlx_result.blocking_reasons
    assert (
        FailureCategory.UNSUPPORTED_FINETUNE_FORMAT.value
        in compressed_result.blocking_reasons
    )


def test_candidate_filter_requires_complete_selection_metadata() -> None:
    cases = (
        ("estimated_params_billion", None, FailureCategory.MISSING_PARAMETER_COUNT.value),
        ("estimated_download_size_gb", None, FailureCategory.MISSING_DOWNLOAD_SIZE.value),
        ("license", None, FailureCategory.MISSING_LICENSE.value),
        ("downloads", 49, FailureCategory.INSUFFICIENT_DOWNLOADS.value),
        ("chat_template_present", False, FailureCategory.MISSING_CHAT_TEMPLATE.value),
        ("context_length", None, FailureCategory.MISSING_CONTEXT_LENGTH.value),
    )

    for field, value, expected_reason in cases:
        candidate = _valid_candidate()
        candidate[field] = value

        result = filter_candidate(candidate, _scan_config())

        assert result.eligible is False
        assert expected_reason in result.blocking_reasons


def _valid_candidate() -> dict[str, object]:
    return {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "abc123",
        "pipeline_tag": "text-generation",
        "tags": ["text-generation", "license:apache-2.0"],
        "license": "apache-2.0",
        "gated": False,
        "private": False,
        "has_tokenizer": True,
        "has_config": True,
        "has_weights": True,
        "has_safetensors": True,
        "estimated_params_billion": 7,
        "estimated_download_size_gb": 16,
        "downloads": 100,
        "architecture": ["Qwen2ForCausalLM"],
        "config": {"architectures": ["Qwen2ForCausalLM"]},
        "chat_template_present": True,
        "context_length": 32768,
    }


def _scan_config() -> dict[str, object]:
    return {
        "max_params_billion": 30,
        "min_downloads": 50,
        "allow_gated": False,
        "allow_trust_remote_code": False,
        "require_safetensors": True,
        "allowed_licenses": ["apache-2.0", "mit", "llama", "other"],
        "blocked_licenses": ["unknown-restrictive"],
        "max_download_gb": 150,
    }
