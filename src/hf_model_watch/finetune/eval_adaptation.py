from __future__ import annotations

from typing import Any

EPSILON = 1e-9


def compute_adaptation_metrics(
    model_id: str,
    run_id: str,
    task_type: str,
    peft_config: dict[str, Any],
    base_quality_score: float,
    adapted_quality_score: float,
    token_counts: dict[str, int | None],
    trainable_params: int | None = None,
    total_params: int | None = None,
) -> dict[str, Any]:
    lift = adapted_quality_score - base_quality_score
    normalized_lift = lift / max(1 - base_quality_score, EPSILON)
    total_tokens = token_counts.get("training_total_tokens")
    result = {
        "model_id": model_id,
        "run_id": run_id,
        "task_type": task_type,
        "peft_method": peft_config.get("peft_method") or peft_config.get("method"),
        "base_quality_score": base_quality_score,
        "adapted_quality_score": adapted_quality_score,
        "adaptation_lift": lift,
        "normalized_adaptation_lift": normalized_lift,
        "training_examples": token_counts.get("training_examples"),
        "training_prompt_tokens": token_counts.get("training_prompt_tokens"),
        "training_completion_tokens": token_counts.get("training_completion_tokens"),
        "training_total_tokens": total_tokens,
        "train_steps": peft_config.get("max_steps"),
        "effective_batch_size": peft_config.get("effective_batch_size"),
        "max_seq_length": peft_config.get("max_sequence_length"),
        "optimizer": peft_config.get("optimizer"),
        "scheduler": peft_config.get("scheduler"),
        "learning_rate": peft_config.get("learning_rate"),
        "lora_r": peft_config.get("r") or peft_config.get("rank"),
        "lora_alpha": peft_config.get("alpha"),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_param_ratio": _param_ratio(trainable_params, total_params),
        "saturated_baseline": base_quality_score >= 0.90,
    }
    if total_tokens:
        result["adaptation_lift_per_1k_training_tokens"] = 1000 * lift / total_tokens
        result["adaptation_lift_per_1M_training_tokens"] = 1_000_000 * lift / total_tokens
        result["normalized_lift_per_1M_training_tokens"] = (
            1_000_000 * normalized_lift / total_tokens
        )
        result[f"{task_type}_adaptation_lift_per_1M_training_tokens"] = (
            1_000_000 * lift / total_tokens
        )
    else:
        result["adaptation_lift_per_1k_training_tokens"] = None
        result["adaptation_lift_per_1M_training_tokens"] = None
        result["normalized_lift_per_1M_training_tokens"] = None
        result[f"{task_type}_adaptation_lift_per_1M_training_tokens"] = None
    return result


def _param_ratio(trainable_params: int | None, total_params: int | None) -> float | None:
    if trainable_params is None or not total_params:
        return None
    return trainable_params / total_params


def weighted_overall_lift(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {"base_quality_score": 0.0, "adapted_quality_score": 0.0, "adaptation_lift": 0.0}
    base = sum(float(item["base_quality_score"]) for item in results) / len(results)
    adapted = sum(float(item["adapted_quality_score"]) for item in results) / len(results)
    return {
        "base_quality_score": base,
        "adapted_quality_score": adapted,
        "adaptation_lift": adapted - base,
    }
