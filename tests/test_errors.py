from __future__ import annotations

from hf_model_watch.errors import FAILURE_CATEGORIES

REQUIRED_FAILURE_CATEGORIES = {
    "not_causal_lm",
    "too_large_for_h100",
    "missing_tokenizer",
    "missing_config",
    "missing_weights",
    "missing_safetensors",
    "gated_no_access",
    "license_blocked",
    "already_tested_revision",
    "requires_trust_remote_code",
    "unsupported_transformers_architecture",
    "unsupported_vllm_architecture",
    "download_failed",
    "download_too_large",
    "load_oom",
    "load_failed",
    "generation_failed",
    "latency_benchmark_failed",
    "quality_eval_failed",
    "context_test_failed",
    "finetune_smoke_failed",
    "finetune_oom",
    "finetune_failed",
    "adapter_save_reload_failed",
    "artifact_upload_failed",
    "controller_auth_failed",
    "nebius_job_failed",
    "unknown_failure",
}


def test_required_failure_categories_exist() -> None:
    assert REQUIRED_FAILURE_CATEGORIES <= FAILURE_CATEGORIES
