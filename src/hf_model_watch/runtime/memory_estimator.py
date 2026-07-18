from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BYTES_PER_GB = 1024**3


@dataclass(frozen=True)
class MemoryEstimate:
    params_billion: float
    bf16_inference_gb: float
    kv_cache_gb: float
    qlora_4bit_training_gb: float
    lora_adapter_gb: float
    total_inference_gb: float
    total_qlora_gb: float
    effective_gpu_memory_gb: float
    fits_bf16: bool
    fits_qlora_4bit: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "params_billion": self.params_billion,
            "bf16_inference_gb": self.bf16_inference_gb,
            "kv_cache_gb": self.kv_cache_gb,
            "qlora_4bit_training_gb": self.qlora_4bit_training_gb,
            "lora_adapter_gb": self.lora_adapter_gb,
            "total_inference_gb": self.total_inference_gb,
            "total_qlora_gb": self.total_qlora_gb,
            "effective_gpu_memory_gb": self.effective_gpu_memory_gb,
            "fits_bf16": self.fits_bf16,
            "fits_qlora_4bit": self.fits_qlora_4bit,
        }


def estimate_memory(
    params_billion: float,
    context_tokens: int,
    batch_size: int,
    config: dict[str, Any],
) -> MemoryEstimate:
    gpu_memory_gb = float(config.get("gpu_memory_gb", 80))
    safety_margin = float(config.get("memory_safety_margin", 0.85))
    effective_gpu_memory_gb = gpu_memory_gb * safety_margin

    bf16_weights_gb = params_billion * 2_000_000_000 / BYTES_PER_GB
    overhead_gb = max(1.0, bf16_weights_gb * 0.12)
    kv_cache_gb = estimate_kv_cache_gb(params_billion, context_tokens, batch_size)
    qlora_weights_gb = params_billion * 0.5
    optimizer_overhead_gb = max(2.0, params_billion * 0.25)
    lora_adapter_gb = max(0.02, params_billion * 0.015)

    total_inference_gb = bf16_weights_gb + kv_cache_gb + overhead_gb
    total_qlora_gb = qlora_weights_gb + kv_cache_gb + optimizer_overhead_gb + lora_adapter_gb

    return MemoryEstimate(
        params_billion=params_billion,
        bf16_inference_gb=round(bf16_weights_gb + overhead_gb, 3),
        kv_cache_gb=round(kv_cache_gb, 3),
        qlora_4bit_training_gb=round(qlora_weights_gb + optimizer_overhead_gb, 3),
        lora_adapter_gb=round(lora_adapter_gb, 3),
        total_inference_gb=round(total_inference_gb, 3),
        total_qlora_gb=round(total_qlora_gb, 3),
        effective_gpu_memory_gb=round(effective_gpu_memory_gb, 3),
        fits_bf16=total_inference_gb <= effective_gpu_memory_gb,
        fits_qlora_4bit=total_qlora_gb <= effective_gpu_memory_gb,
    )


def estimate_kv_cache_gb(params_billion: float, context_tokens: int, batch_size: int) -> float:
    # Heuristic for decoder-only models when layer/head dimensions are not yet known.
    model_scale = max(params_billion, 0.1)
    bytes_per_token = 65_536 * (model_scale / 7) ** 0.5
    return (bytes_per_token * context_tokens * batch_size) / BYTES_PER_GB
