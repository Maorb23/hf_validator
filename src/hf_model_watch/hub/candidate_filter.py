from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.hub.metadata import looks_like_causal_lm, requires_trust_remote_code

SUPPORTED_TRANSFORMERS_ARCH_HINTS = (
    "CausalLM",
    "GPT",
    "Llama",
    "Mistral",
    "Qwen",
    "Gemma",
    "Phi",
    "Yi",
    "Deepseek",
    "DeepSeek",
)


@dataclass(frozen=True)
class FilterResult:
    model_id: str
    revision: str | None
    eligible: bool
    blocking_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "eligible": self.eligible,
            "blocking_reasons": self.blocking_reasons,
        }


def filter_candidate(
    candidate: dict[str, Any],
    scan_config: dict[str, Any],
    already_tested_revisions: set[tuple[str, str]] | None = None,
) -> FilterResult:
    reasons: list[str] = []
    model_id = str(candidate.get("model_id", ""))
    revision = candidate.get("revision")
    config = candidate.get("config") if isinstance(candidate.get("config"), dict) else {}
    tags = candidate.get("tags") if isinstance(candidate.get("tags"), list) else []

    if not looks_like_causal_lm(config, candidate.get("pipeline_tag"), tags):
        reasons.append(FailureCategory.NOT_CAUSAL_LM.value)
    if not candidate.get("has_tokenizer"):
        reasons.append(FailureCategory.MISSING_TOKENIZER.value)
    if not candidate.get("has_config"):
        reasons.append(FailureCategory.MISSING_CONFIG.value)
    if not candidate.get("has_weights"):
        reasons.append(FailureCategory.MISSING_WEIGHTS.value)
    if scan_config.get("require_safetensors", True) and not candidate.get("has_safetensors"):
        reasons.append(FailureCategory.MISSING_SAFETENSORS.value)

    if candidate.get("gated") and not scan_config.get("allow_gated", False):
        reasons.append(FailureCategory.GATED_NO_ACCESS.value)
    if candidate.get("private") and not scan_config.get("allow_gated", False):
        reasons.append(FailureCategory.GATED_NO_ACCESS.value)

    license_value = _normalize_license(candidate.get("license"))
    allowed = {str(item).lower() for item in scan_config.get("allowed_licenses", [])}
    blocked = {str(item).lower() for item in scan_config.get("blocked_licenses", [])}
    if license_value in blocked or (license_value and allowed and license_value not in allowed):
        reasons.append(FailureCategory.LICENSE_BLOCKED.value)

    params = _coerce_float(candidate.get("estimated_params_billion"))
    max_params = _coerce_float(scan_config.get("max_params_billion"))
    if params is not None and max_params is not None and params > max_params:
        reasons.append(FailureCategory.TOO_LARGE_FOR_H100.value)

    download_size = _coerce_float(candidate.get("estimated_download_size_gb"))
    max_download = _coerce_float(scan_config.get("max_download_gb"))
    if download_size is not None and max_download is not None and download_size > max_download:
        reasons.append(FailureCategory.DOWNLOAD_TOO_LARGE.value)

    if revision and (model_id, str(revision)) in (already_tested_revisions or set()):
        reasons.append(FailureCategory.ALREADY_TESTED_REVISION.value)

    trust_remote_code_disabled = not scan_config.get("allow_trust_remote_code", False)
    if requires_trust_remote_code(config, tags) and trust_remote_code_disabled:
        reasons.append(FailureCategory.REQUIRES_TRUST_REMOTE_CODE.value)

    architectures = candidate.get("architecture") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    if architectures and not any(
        any(hint in str(architecture) for hint in SUPPORTED_TRANSFORMERS_ARCH_HINTS)
        for architecture in architectures
    ):
        reasons.append(FailureCategory.UNSUPPORTED_TRANSFORMERS_ARCHITECTURE.value)

    deduped = list(dict.fromkeys(reasons))
    return FilterResult(
        model_id=model_id,
        revision=str(revision) if revision else None,
        eligible=not deduped,
        blocking_reasons=deduped,
    )


def _normalize_license(value: Any) -> str | None:
    if not value:
        return None
    return str(value).strip().lower()


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
