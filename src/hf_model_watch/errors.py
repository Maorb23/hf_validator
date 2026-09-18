from __future__ import annotations

from enum import StrEnum


class FailureCategory(StrEnum):
    NOT_CAUSAL_LM = "not_causal_lm"
    TOO_LARGE_FOR_H100 = "too_large_for_h100"
    MISSING_PARAMETER_COUNT = "missing_parameter_count"
    MISSING_DOWNLOAD_SIZE = "missing_download_size"
    MISSING_LICENSE = "missing_license"
    INSUFFICIENT_DOWNLOADS = "insufficient_downloads"
    MISSING_CHAT_TEMPLATE = "missing_chat_template"
    MISSING_CONTEXT_LENGTH = "missing_context_length"
    MISSING_TOKENIZER = "missing_tokenizer"
    MISSING_CONFIG = "missing_config"
    MISSING_WEIGHTS = "missing_weights"
    MISSING_SAFETENSORS = "missing_safetensors"
    GATED_NO_ACCESS = "gated_no_access"
    LICENSE_BLOCKED = "license_blocked"
    ALREADY_TESTED_REVISION = "already_tested_revision"
    REQUIRES_TRUST_REMOTE_CODE = "requires_trust_remote_code"
    UNSUPPORTED_FINETUNE_FORMAT = "unsupported_finetune_format"
    UNSUPPORTED_TRANSFORMERS_ARCHITECTURE = "unsupported_transformers_architecture"
    UNSUPPORTED_VLLM_ARCHITECTURE = "unsupported_vllm_architecture"
    DOWNLOAD_FAILED = "download_failed"
    DOWNLOAD_TOO_LARGE = "download_too_large"
    LOAD_OOM = "load_oom"
    LOAD_FAILED = "load_failed"
    GENERATION_FAILED = "generation_failed"
    LATENCY_BENCHMARK_FAILED = "latency_benchmark_failed"
    QUALITY_EVAL_FAILED = "quality_eval_failed"
    CONTEXT_TEST_FAILED = "context_test_failed"
    FINETUNE_SMOKE_FAILED = "finetune_smoke_failed"
    FINETUNE_OOM = "finetune_oom"
    FINETUNE_FAILED = "finetune_failed"
    ADAPTER_SAVE_RELOAD_FAILED = "adapter_save_reload_failed"
    ARTIFACT_UPLOAD_FAILED = "artifact_upload_failed"
    CONTROLLER_AUTH_FAILED = "controller_auth_failed"
    NEBIUS_JOB_FAILED = "nebius_job_failed"
    UNKNOWN_FAILURE = "unknown_failure"


FAILURE_CATEGORIES: frozenset[str] = frozenset(category.value for category in FailureCategory)
