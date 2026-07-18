from __future__ import annotations

from hf_model_watch.runtime.memory_estimator import estimate_memory


def test_memory_estimator_rejects_obviously_too_large_model() -> None:
    estimate = estimate_memory(
        params_billion=180,
        context_tokens=8192,
        batch_size=1,
        config={"gpu_memory_gb": 80, "memory_safety_margin": 0.85},
    )

    assert estimate.fits_bf16 is False
    assert estimate.fits_qlora_4bit is False
    assert estimate.total_inference_gb > estimate.effective_gpu_memory_gb
