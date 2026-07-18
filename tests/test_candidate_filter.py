from __future__ import annotations

from hf_model_watch.errors import FailureCategory
from hf_model_watch.hub.candidate_filter import filter_candidate


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
        "architecture": ["Qwen2ForCausalLM"],
        "config": {"architectures": ["Qwen2ForCausalLM"]},
    }


def _scan_config() -> dict[str, object]:
    return {
        "max_params_billion": 30,
        "allow_gated": False,
        "allow_trust_remote_code": False,
        "require_safetensors": True,
        "allowed_licenses": ["apache-2.0", "mit", "llama", "other"],
        "blocked_licenses": ["unknown-restrictive"],
        "max_download_gb": 150,
    }
