from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ModelProfile:
    architecture: str
    is_moe: bool
    total_parameters: int
    active_parameters_estimate: int | None
    expert_parameters: int | None
    shared_parameters: int | None
    total_experts: int | None
    active_experts_per_token: int | None
    shared_experts: int | None
    detection_reason: str
    confidence: Literal["high", "medium", "low"]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_model_config(
    config: Any,
    *,
    estimated_params_billion: float | None = None,
) -> ModelProfile:
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    architectures = data.get("architectures") or []
    architecture = str(architectures[0] if architectures else data.get("model_type") or "unknown")
    total_experts = _first_int(
        data,
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
    )
    active_experts = _first_int(
        data,
        "num_experts_per_tok",
        "num_selected_experts",
        "top_k",
    )
    shared_experts = _first_int(data, "num_shared_experts", "n_shared_experts") or 0
    moe_fields = [
        key
        for key in (
            "num_local_experts",
            "num_experts",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "router_aux_loss_coef",
        )
        if data.get(key) is not None
    ]
    is_moe = bool(total_experts and total_experts > 1 and moe_fields)
    if is_moe and active_experts:
        reason = f"MoE config fields detected: {', '.join(moe_fields)}"
        confidence: Literal["high", "medium", "low"] = "high"
    elif is_moe:
        reason = f"partial MoE config fields detected: {', '.join(moe_fields)}"
        confidence = "medium"
    else:
        reason = "no routed-expert configuration fields detected"
        confidence = "high"

    total_parameters = int(max(0.0, estimated_params_billion or 0.0) * 1_000_000_000)
    shared_parameters = None
    expert_parameters = None
    active_parameters = None
    if is_moe and total_parameters and active_experts:
        expert_fraction = _estimate_expert_fraction(data)
        expert_parameters = int(total_parameters * expert_fraction)
        shared_parameters = total_parameters - expert_parameters
        expert_denominator = max(total_experts + shared_experts, 1)
        routed_parameters = int(expert_parameters * total_experts / expert_denominator)
        shared_expert_parameters = expert_parameters - routed_parameters
        active_fraction = min(1.0, active_experts / max(total_experts, 1))
        active_parameters = (
            shared_parameters
            + shared_expert_parameters
            + int(routed_parameters * active_fraction)
        )
    elif total_parameters and not is_moe:
        shared_parameters = total_parameters
        active_parameters = total_parameters

    return ModelProfile(
        architecture=architecture,
        is_moe=is_moe,
        total_parameters=total_parameters,
        active_parameters_estimate=active_parameters,
        expert_parameters=expert_parameters,
        shared_parameters=shared_parameters,
        total_experts=total_experts,
        active_experts_per_token=active_experts,
        shared_experts=shared_experts or None,
        detection_reason=reason,
        confidence=confidence,
    )


def count_unique_parameters(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += int(parameter.numel())
    return total


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _estimate_expert_fraction(data: dict[str, Any]) -> float:
    hidden = _first_int(data, "hidden_size", "d_model")
    layers = _first_int(data, "num_hidden_layers", "num_layers")
    experts = _first_int(data, "num_local_experts", "num_experts", "n_routed_experts")
    expert_intermediate = _first_int(data, "moe_intermediate_size", "intermediate_size")
    if not all((hidden, layers, experts, expert_intermediate)):
        return 0.7
    expert_weights = layers * experts * 3 * hidden * expert_intermediate
    attention_weights = layers * 4 * hidden * hidden
    embedding_weights = (_first_int(data, "vocab_size") or 0) * hidden
    denominator = expert_weights + attention_weights + embedding_weights
    return min(0.95, max(0.1, expert_weights / max(denominator, 1)))
