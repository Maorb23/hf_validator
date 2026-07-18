from __future__ import annotations

import importlib.util
from typing import Any

SUPPORTED_ARCHITECTURE_HINTS = (
    "Llama",
    "Mistral",
    "Qwen",
    "Gemma",
    "Phi",
    "Deepseek",
    "DeepSeek",
)


def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def appears_supported_by_vllm(metadata: dict[str, Any]) -> bool:
    architectures = metadata.get("architecture") or metadata.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    return any(
        any(hint in str(architecture) for hint in SUPPORTED_ARCHITECTURE_HINTS)
        for architecture in architectures
    )


def vllm_probe(metadata: dict[str, Any]) -> dict[str, Any]:
    available = is_vllm_available()
    supported = appears_supported_by_vllm(metadata)
    return {
        "available": available,
        "architecture_appears_supported": supported,
        "usable": available and supported,
        "message": "vLLM is available" if available else "vLLM is not installed",
    }
