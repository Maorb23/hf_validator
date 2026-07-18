from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hf_model_watch.errors import FailureCategory
from hf_model_watch.hub.metadata import requires_trust_remote_code
from hf_model_watch.runtime.backend_vllm import vllm_probe
from hf_model_watch.runtime.memory_estimator import estimate_memory


@dataclass(frozen=True)
class RuntimeDecision:
    backend: str | None
    dtype: str | None
    attention_implementation: str | None
    quantization: str | None
    max_context: int
    fits_h100: bool
    fail_fast: bool
    failure_category: str | None
    failure_message: str | None
    trust_remote_code_required: bool
    selected_load_args: dict[str, Any]
    fallback_ladder: list[dict[str, Any]]
    memory_estimate: dict[str, Any]
    vllm: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "dtype": self.dtype,
            "attention_implementation": self.attention_implementation,
            "quantization": self.quantization,
            "max_context": self.max_context,
            "fits_h100": self.fits_h100,
            "fail_fast": self.fail_fast,
            "failure_category": self.failure_category,
            "failure_message": self.failure_message,
            "trust_remote_code_required": self.trust_remote_code_required,
            "selected_load_args": self.selected_load_args,
            "fallback_ladder": self.fallback_ladder,
            "memory_estimate": self.memory_estimate,
            "vllm": self.vllm,
        }


def resolve_runtime(
    metadata: dict[str, Any],
    scan_config: dict[str, Any],
    serverless_config: dict[str, Any],
) -> RuntimeDecision:
    tags = metadata.get("tags") if isinstance(metadata.get("tags"), list) else []
    config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
    trust_required = requires_trust_remote_code(config, tags)
    trust_allowed = bool(scan_config.get("allow_trust_remote_code", False))
    max_context = int(
        metadata.get("context_length") or config.get("max_position_embeddings") or 2048
    )
    params = float(metadata.get("estimated_params_billion") or 0.1)
    memory = estimate_memory(
        params_billion=params,
        context_tokens=min(max_context, 4096),
        batch_size=1,
        config=serverless_config,
    )
    vllm = vllm_probe(metadata)
    fallback_ladder = [
        {"attempt": 1, "backend": "vllm", "available": vllm["usable"]},
        {"attempt": 2, "backend": "transformers", "dtype": "bf16", "attention": "sdpa"},
        {"attempt": 3, "backend": "transformers", "dtype": "bf16", "attention": "eager"},
        {"attempt": 4, "backend": "transformers", "quantization": "4bit"},
        {"attempt": 5, "backend": None, "failure": FailureCategory.LOAD_FAILED.value},
    ]

    if trust_required and not trust_allowed:
        return _failed_decision(
            max_context=max_context,
            trust_required=trust_required,
            memory=memory.as_dict(),
            vllm=vllm,
            fallback_ladder=fallback_ladder,
            category=FailureCategory.REQUIRES_TRUST_REMOTE_CODE.value,
            message="Model appears to require trust_remote_code but config disallows it",
        )

    max_params = float(
        serverless_config.get("max_params_billion", scan_config.get("max_params_billion", 30))
    )
    if params > max_params or not memory.fits_qlora_4bit:
        return _failed_decision(
            max_context=max_context,
            trust_required=trust_required,
            memory=memory.as_dict(),
            vllm=vllm,
            fallback_ladder=fallback_ladder,
            category=FailureCategory.TOO_LARGE_FOR_H100.value,
            message="Estimated memory exceeds configured H100 budget",
        )

    if vllm["usable"] and memory.fits_bf16:
        backend = "vllm"
        attention = None
    else:
        backend = "transformers"
        attention = "sdpa" if memory.fits_bf16 else "eager"

    quantization = None if memory.fits_bf16 else "4bit"
    dtype = "bf16" if memory.fits_bf16 else "fp16"
    return RuntimeDecision(
        backend=backend,
        dtype=dtype,
        attention_implementation=attention,
        quantization=quantization,
        max_context=max_context,
        fits_h100=True,
        fail_fast=False,
        failure_category=None,
        failure_message=None,
        trust_remote_code_required=trust_required,
        selected_load_args={
            "trust_remote_code": trust_allowed and trust_required,
            "torch_dtype": dtype,
            "attn_implementation": attention,
            "load_in_4bit": quantization == "4bit",
        },
        fallback_ladder=fallback_ladder,
        memory_estimate=memory.as_dict(),
        vllm=vllm,
    )


def _failed_decision(
    max_context: int,
    trust_required: bool,
    memory: dict[str, Any],
    vllm: dict[str, Any],
    fallback_ladder: list[dict[str, Any]],
    category: str,
    message: str,
) -> RuntimeDecision:
    return RuntimeDecision(
        backend=None,
        dtype=None,
        attention_implementation=None,
        quantization=None,
        max_context=max_context,
        fits_h100=False,
        fail_fast=True,
        failure_category=category,
        failure_message=message,
        trust_remote_code_required=trust_required,
        selected_load_args={},
        fallback_ladder=fallback_ladder,
        memory_estimate=memory,
        vllm=vllm,
    )
