from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hf_model_watch.hub.scoring import score_candidate


def test_scoring_ranks_newer_popular_complete_candidate_above_weak_one() -> None:
    strong = _candidate(
        "Qwen/Qwen2.5-7B-Instruct",
        downloads=200_000,
        likes=900,
        last_modified=datetime.now(UTC).isoformat(),
        complete=True,
    )
    weak = _candidate(
        "unknown/tiny-upload",
        downloads=2,
        likes=0,
        last_modified=(datetime.now(UTC) - timedelta(days=120)).isoformat(),
        complete=False,
    )

    strong_score = score_candidate(strong, _scan_config(), _scoring_config())
    weak_score = score_candidate(weak, _scan_config(), _scoring_config())

    assert strong_score["eligible"] is True
    assert weak_score["eligible"] is True
    assert strong_score["score"] > weak_score["score"]


def _candidate(
    model_id: str,
    downloads: int,
    likes: int,
    last_modified: str,
    complete: bool,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "revision": "abc123",
        "author": model_id.split("/", 1)[0],
        "last_modified": last_modified,
        "downloads": downloads,
        "likes": likes,
        "pipeline_tag": "text-generation",
        "tags": ["text-generation", "license:apache-2.0"],
        "license": "apache-2.0",
        "gated": False,
        "private": False,
        "has_tokenizer": True,
        "has_config": True,
        "has_weights": True,
        "has_safetensors": True,
        "estimated_params_billion": 7 if "Qwen" in model_id else 0.5,
        "estimated_download_size_gb": 16,
        "architecture": ["Qwen2ForCausalLM"] if complete else ["GPT2LMHeadModel"],
        "config": {"architectures": ["Qwen2ForCausalLM"] if complete else ["GPT2LMHeadModel"]},
        "chat_template_present": complete,
        "context_length": 32768 if complete else None,
        "model_card_present": complete,
    }


def _scan_config() -> dict[str, object]:
    return {
        "max_candidates_to_run": 5,
        "max_params_billion": 30,
        "allow_gated": False,
        "allow_trust_remote_code": False,
        "require_safetensors": True,
        "allowed_licenses": ["apache-2.0", "mit", "llama", "other"],
        "blocked_licenses": ["unknown-restrictive"],
        "max_download_gb": 150,
    }


def _scoring_config() -> dict[str, object]:
    return {
        "weights": {
            "recency": 0.18,
            "downloads": 0.14,
            "likes": 0.1,
            "credible_author": 0.1,
            "complete_metadata": 0.12,
            "chat_template": 0.08,
            "context_length": 0.08,
            "lineage": 0.08,
            "known_family": 0.06,
            "useful_size_band": 0.04,
            "uniqueness": 0.02,
        },
        "credible_authors": ["Qwen"],
        "known_families": ["Qwen", "Llama", "Mistral", "Gemma", "Phi", "DeepSeek", "Yi"],
    }
