from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from hf_model_watch.finetune.datasets import AdaptationExample

TaskBreadth = Literal["narrow", "general", "domain"]
PeftMethod = Literal["lora", "qlora"]


@dataclass(frozen=True)
class FinetuneOverrides:
    method: str = "auto"
    rank: int | None = None
    alpha: int | None = None
    dropout: float | None = None
    learning_rate: float | None = None
    microbatch_size: int | None = None
    gradient_accumulation: int | None = None
    max_sequence_length: int | None = None
    optimizer: str | None = None
    gradient_checkpointing: bool | None = None
    packing: bool | None = None
    warmup_ratio: float | None = None
    max_steps: int | None = None
    target_modules: str | list[str] | None = None
    allow_method_fallback: bool | None = None
    allow_batch_fallback: bool | None = None


@dataclass(frozen=True)
class DatasetProfile:
    task: str
    task_breadth: TaskBreadth
    examples: int
    total_tokens: int
    mean_tokens: float
    p95_tokens: int
    p99_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareProfile:
    gpu_name: str
    total_vram_bytes: int
    bf16_supported: bool
    gpu_count: int = 1
    flash_attention_supported: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedFinetunePolicy:
    method: PeftMethod
    rank: int
    alpha: int
    use_rslora: bool
    dropout: float
    learning_rate: float
    microbatch_size: int
    gradient_accumulation: int
    effective_batch_size: int
    target_tokens_per_update: int
    max_sequence_length: int
    optimizer: str
    optimizer_fallback: str
    scheduler: str
    router_aux_loss_coef: float
    gradient_checkpointing: bool
    packing: bool
    warmup_ratio: float
    warmup_steps: int
    max_steps: int
    eval_steps: int
    eval_microbatch_size: int
    json_eval_generation_token_multiplier: float
    json_eval_min_new_tokens: int
    json_eval_max_new_tokens: int
    logging_steps: int
    wandb_enabled: bool
    wandb_project: str
    max_grad_norm: float
    weight_decay: float
    target_modules: str | list[str]
    exclude_modules: tuple[str, ...]
    fan_in_fan_out: bool
    bnb_4bit_quant_type: str | None
    bnb_4bit_compute_dtype: str | None
    bnb_4bit_use_double_quant: bool
    bf16: bool
    seed: int
    planning_memory_fraction: float
    max_memory_fraction: float
    estimated_peak_vram_bytes: int
    allow_method_fallback: bool
    allow_batch_fallback: bool
    explicit_overrides: frozenset[str] = field(default_factory=frozenset)
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "peft_method": self.method,
                "r": self.rank,
                "batch_size": self.microbatch_size,
                "dropout": self.dropout,
                "reason": "; ".join(self.reasons),
                "explicit_overrides": sorted(self.explicit_overrides),
                "exclude_modules": list(self.exclude_modules),
            }
        )
        return data


DEFAULT_EXCLUDED_MODULES = (
    "lm_head",
    "classifier",
    "score",
    "reward_head",
    "multi_modal_projector",
    "multimodal_projector",
)


def build_dataset_profile(
    task: str,
    examples: list[AdaptationExample],
    task_config: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
) -> DatasetProfile:
    config = task_config or {}
    breadth = str(config.get("task_breadth") or _default_task_breadth(task)).lower()
    if breadth not in {"narrow", "general", "domain"}:
        raise ValueError(f"unsupported task_breadth for {task}: {breadth}")
    lengths = sorted(_example_token_length(example, tokenizer) for example in examples)
    total = sum(lengths)
    return DatasetProfile(
        task=task,
        task_breadth=breadth,  # type: ignore[arg-type]
        examples=len(examples),
        total_tokens=total,
        mean_tokens=total / max(len(lengths), 1),
        p95_tokens=_percentile(lengths, 0.95),
        p99_tokens=_percentile(lengths, 0.99),
    )


def resolve_finetune_policy(
    *,
    model_size_billion: float,
    active_model_size_billion: float | None = None,
    architecture: str | list[str] | None,
    context_length: int,
    dataset: DatasetProfile,
    hardware: HardwareProfile,
    config: dict[str, Any] | None = None,
    retry_of: ResolvedFinetunePolicy | None = None,
) -> ResolvedFinetunePolicy:
    cfg = config or {}
    overrides = _overrides_from_config(cfg)
    explicit = _explicit_fields(cfg)
    reasons: list[str] = []
    planning_fraction = float(cfg.get("planning_memory_fraction", 0.75))
    max_memory_fraction = float(cfg.get("max_memory_fraction", 0.85))
    if not 0 < planning_fraction < max_memory_fraction <= 1:
        raise ValueError("memory fractions must satisfy 0 < planning < maximum <= 1")

    max_sequence_length = overrides.max_sequence_length or min(
        max(128, dataset.p95_tokens),
        max(1, context_length),
        int(cfg.get("max_sequence_length_cap", 2048)),
    )
    if overrides.max_sequence_length is None:
        reasons.append(
            f"sequence length {max_sequence_length} selected from dataset p95 and model limits"
        )

    rank = overrides.rank or _automatic_rank(dataset)
    alpha = overrides.alpha or rank
    use_rslora = bool(cfg.get("use_rslora", rank >= 16))
    dropout = overrides.dropout if overrides.dropout is not None else _automatic_dropout(dataset)
    learning_rate = (
        overrides.learning_rate
        if overrides.learning_rate is not None
        else _automatic_learning_rate(dataset, model_size_billion, rank)
    )
    checkpointing = (
        overrides.gradient_checkpointing
        if overrides.gradient_checkpointing is not None
        else True
    )
    packing = (
        overrides.packing
        if overrides.packing is not None
        else hardware.flash_attention_supported
    )
    if overrides.packing is None:
        reasons.append(
            "packing enabled because FlashAttention is available"
            if packing
            else "packing disabled because supported FlashAttention is unavailable"
        )

    method_request = overrides.method.lower()
    if method_request not in {"auto", "lora", "qlora"}:
        raise ValueError(f"unsupported fine-tune method: {method_request}")
    lora_estimate = estimate_peak_vram_bytes(
        model_size_billion=model_size_billion,
        activation_model_size_billion=active_model_size_billion,
        method="lora",
        sequence_length=max_sequence_length,
        microbatch_size=1,
        rank=rank,
        gradient_checkpointing=checkpointing,
    )
    if method_request == "auto":
        method: PeftMethod = (
            "lora"
            if lora_estimate <= hardware.total_vram_bytes * planning_fraction
            else "qlora"
        )
        reasons.append(
            f"{method} selected from conservative memory estimate "
            f"({lora_estimate / 2**30:.1f} GiB LoRA estimate)"
        )
    else:
        method = method_request  # type: ignore[assignment]
        reasons.append(f"{method} preserved from explicit method override")

    if retry_of is not None:
        method, retry_batch, retry_reason = _retry_choice(retry_of, overrides, cfg)
        reasons.append(retry_reason)
    else:
        retry_batch = None

    target_modules = overrides.target_modules or "all-linear"
    if target_modules != "all-linear":
        reasons.append("custom target modules preserved from explicit override")
    else:
        reasons.append("PEFT all-linear targets selected for full linear-layer coverage")

    candidate_batches = [
        int(value) for value in cfg.get("candidate_microbatches", [16, 8, 4, 2, 1])
    ]
    if retry_batch is not None:
        microbatch = retry_batch
    elif overrides.microbatch_size is not None:
        microbatch = overrides.microbatch_size
    else:
        microbatch = _estimated_microbatch(
            candidates=candidate_batches,
            hardware=hardware,
            model_size_billion=model_size_billion,
            active_model_size_billion=active_model_size_billion,
            method=method,
            sequence_length=max_sequence_length,
            rank=rank,
            gradient_checkpointing=checkpointing,
            planning_fraction=planning_fraction,
        )
        reasons.append(f"microbatch {microbatch} selected from conservative memory estimate")

    target_tokens = int(cfg.get("target_tokens_per_update", 32768))
    accumulation = overrides.gradient_accumulation or max(
        1,
        math.ceil(target_tokens / max(1.0, microbatch * dataset.mean_tokens)),
    )
    effective_batch = microbatch * accumulation
    max_steps = overrides.max_steps or int(
        cfg.get("max_train_steps") or cfg.get("max_steps") or 100
    )
    eval_steps = min(max_steps, max(1, int(cfg.get("eval_steps") or 5)))
    eval_microbatch_size = max(1, int(cfg.get("eval_microbatch_size") or 1))
    json_eval_generation_token_multiplier = max(
        1.0, float(cfg.get("json_eval_generation_token_multiplier") or 2.0)
    )
    json_eval_min_new_tokens = max(1, int(cfg.get("json_eval_min_new_tokens") or 32))
    json_eval_max_new_tokens = max(
        json_eval_min_new_tokens, int(cfg.get("json_eval_max_new_tokens") or 256)
    )
    logging_steps = min(max_steps, max(1, int(cfg.get("logging_steps") or 1)))
    warmup_ratio = overrides.warmup_ratio if overrides.warmup_ratio is not None else 0.03
    warmup_steps = min(max_steps, max(0, math.ceil(warmup_ratio * max_steps)))
    estimated_peak = estimate_peak_vram_bytes(
        model_size_billion=model_size_billion,
        activation_model_size_billion=active_model_size_billion,
        method=method,
        sequence_length=max_sequence_length,
        microbatch_size=microbatch,
        rank=rank,
        gradient_checkpointing=checkpointing,
    )

    optimizer = overrides.optimizer or _automatic_optimizer(
        method, estimated_peak, hardware.total_vram_bytes, planning_fraction
    )
    optimizer_fallback = "paged_adamw_32bit" if method == "qlora" else "adamw_torch"
    allow_method_fallback = (
        overrides.allow_method_fallback
        if overrides.allow_method_fallback is not None
        else method_request == "auto"
    )
    allow_batch_fallback = (
        overrides.allow_batch_fallback
        if overrides.allow_batch_fallback is not None
        else overrides.microbatch_size is None
    )
    arch_text = (
        " ".join(architecture) if isinstance(architecture, list) else str(architecture or "")
    )
    return ResolvedFinetunePolicy(
        method=method,
        rank=rank,
        alpha=alpha,
        use_rslora=use_rslora,
        dropout=dropout,
        learning_rate=learning_rate,
        microbatch_size=microbatch,
        gradient_accumulation=accumulation,
        effective_batch_size=effective_batch,
        target_tokens_per_update=target_tokens,
        max_sequence_length=max_sequence_length,
        optimizer=optimizer,
        optimizer_fallback=optimizer_fallback,
        scheduler=str(cfg.get("scheduler", "cosine")),
        router_aux_loss_coef=float(cfg.get("router_aux_loss_coef", 0.001)),
        gradient_checkpointing=checkpointing,
        packing=packing,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        eval_steps=eval_steps,
        eval_microbatch_size=eval_microbatch_size,
        json_eval_generation_token_multiplier=json_eval_generation_token_multiplier,
        json_eval_min_new_tokens=json_eval_min_new_tokens,
        json_eval_max_new_tokens=json_eval_max_new_tokens,
        logging_steps=logging_steps,
        wandb_enabled=bool(cfg.get("wandb_enabled", True)),
        wandb_project=str(cfg.get("wandb_project") or "hf-model-watch"),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        target_modules=target_modules,
        exclude_modules=tuple(cfg.get("exclude_modules") or DEFAULT_EXCLUDED_MODULES),
        fan_in_fan_out="gpt2" in arch_text.lower(),
        bnb_4bit_quant_type=(
            str(cfg.get("bnb_4bit_quant_type", "nf4")) if method == "qlora" else None
        ),
        bnb_4bit_compute_dtype=(
            str(cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
            if method == "qlora"
            else None
        ),
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        bf16=bool(cfg.get("bf16", hardware.bf16_supported)),
        seed=int(cfg.get("seed", 42)),
        planning_memory_fraction=planning_fraction,
        max_memory_fraction=max_memory_fraction,
        estimated_peak_vram_bytes=estimated_peak,
        allow_method_fallback=allow_method_fallback,
        allow_batch_fallback=allow_batch_fallback,
        explicit_overrides=frozenset(explicit),
        reasons=tuple(reasons),
    )


def estimate_peak_vram_bytes(
    *,
    model_size_billion: float,
    activation_model_size_billion: float | None = None,
    method: PeftMethod,
    sequence_length: int,
    microbatch_size: int,
    rank: int,
    gradient_checkpointing: bool,
) -> int:
    parameters = max(model_size_billion, 0.01) * 1_000_000_000
    activation_parameters = max(
        activation_model_size_billion or model_size_billion,
        0.01,
    ) * 1_000_000_000
    weight_bytes = parameters * (2.0 if method == "lora" else 0.65)
    adapter_and_optimizer = parameters * min(0.08, 0.003 * (rank / 16))
    activation_scale = 0.45 if gradient_checkpointing else 1.0
    activation_bytes = (
        activation_parameters**0.5
        * max(sequence_length, 128)
        * max(microbatch_size, 1)
        * 120
        * activation_scale
    )
    allocator_margin = 1.12
    return int((weight_bytes + adapter_and_optimizer + activation_bytes) * allocator_margin)


def _automatic_rank(dataset: DatasetProfile) -> int:
    if dataset.task_breadth == "narrow":
        return 16 if dataset.examples < 20_000 else 32
    if dataset.task_breadth == "general":
        return 16 if dataset.examples < 2_000 else 32
    if dataset.examples < 2_000:
        return 16
    return 32 if dataset.examples < 20_000 else 64


def _automatic_dropout(dataset: DatasetProfile) -> float:
    if dataset.examples < 2_000:
        return 0.1
    if dataset.examples < 20_000:
        return 0.05
    return 0.02


def _automatic_learning_rate(
    dataset: DatasetProfile, model_size_billion: float, rank: int
) -> float:
    learning_rate = 5e-5 if dataset.task_breadth == "domain" else 1e-4
    if model_size_billion >= 30 or rank >= 64:
        learning_rate = min(learning_rate, 5e-5)
    return learning_rate


def _automatic_optimizer(
    method: PeftMethod,
    estimated_peak: int,
    total_vram: int,
    planning_fraction: float,
) -> str:
    if method == "qlora" and estimated_peak > total_vram * planning_fraction:
        return "paged_adamw_8bit"
    return "adamw_torch_fused"


def _estimated_microbatch(
    *,
    candidates: list[int],
    hardware: HardwareProfile,
    model_size_billion: float,
    active_model_size_billion: float | None,
    method: PeftMethod,
    sequence_length: int,
    rank: int,
    gradient_checkpointing: bool,
    planning_fraction: float,
) -> int:
    valid = sorted({value for value in candidates if value > 0}, reverse=True)
    if not valid:
        raise ValueError("candidate_microbatches must contain a positive integer")
    for candidate in valid:
        estimate = estimate_peak_vram_bytes(
            model_size_billion=model_size_billion,
            activation_model_size_billion=active_model_size_billion,
            method=method,
            sequence_length=sequence_length,
            microbatch_size=candidate,
            rank=rank,
            gradient_checkpointing=gradient_checkpointing,
        )
        if estimate <= hardware.total_vram_bytes * planning_fraction:
            return candidate
    return 1


def _retry_choice(
    previous: ResolvedFinetunePolicy,
    overrides: FinetuneOverrides,
    config: dict[str, Any],
) -> tuple[PeftMethod, int, str]:
    if previous.method == "lora" and previous.allow_method_fallback:
        batch = previous.microbatch_size
        if previous.allow_batch_fallback:
            batch = min(batch, int(config.get("qlora_fallback_batch_size", 2)))
        return "qlora", max(1, batch), "single retry changed LoRA to safer QLoRA"
    if not previous.allow_batch_fallback:
        raise ValueError("configuration failed and explicit batch override forbids fallback")
    if overrides.method == "lora" and previous.method == "lora":
        return "lora", 1, "single retry preserved forced LoRA and reduced batch to 1"
    return "qlora", 1, "single retry preserved QLoRA and reduced batch to 1"


def _overrides_from_config(config: dict[str, Any]) -> FinetuneOverrides:
    def optional_int(*keys: str) -> int | None:
        value = _first_present(config, *keys)
        return int(value) if value is not None else None

    def optional_float(*keys: str) -> float | None:
        value = _first_present(config, *keys)
        return float(value) if value is not None else None

    return FinetuneOverrides(
        method=str(config.get("method") or config.get("default_method") or "auto"),
        rank=optional_int("lora_rank", "rank", "r"),
        alpha=optional_int("lora_alpha", "alpha"),
        dropout=optional_float("lora_dropout", "dropout"),
        learning_rate=optional_float("learning_rate"),
        microbatch_size=optional_int("microbatch_size", "batch_size"),
        gradient_accumulation=optional_int("gradient_accumulation"),
        max_sequence_length=optional_int("max_sequence_length"),
        optimizer=_first_present(config, "optimizer"),
        gradient_checkpointing=_optional_bool(config, "gradient_checkpointing"),
        packing=_optional_bool(config, "packing"),
        warmup_ratio=optional_float("warmup_ratio"),
        max_steps=optional_int("max_train_steps", "max_steps"),
        target_modules=_first_present(config, "target_modules"),
        allow_method_fallback=_optional_bool(config, "allow_method_fallback"),
        allow_batch_fallback=_optional_bool(config, "allow_batch_fallback"),
    )


def _explicit_fields(config: dict[str, Any]) -> set[str]:
    aliases = {
        "rank": ("lora_rank", "rank", "r"),
        "alpha": ("lora_alpha", "alpha"),
        "dropout": ("lora_dropout", "dropout"),
        "microbatch_size": ("microbatch_size", "batch_size"),
        "max_steps": ("max_train_steps", "max_steps"),
    }
    explicit = {name for name, keys in aliases.items() if _first_present(config, *keys) is not None}
    for key in (
        "method",
        "learning_rate",
        "gradient_accumulation",
        "max_sequence_length",
        "optimizer",
        "gradient_checkpointing",
        "packing",
        "warmup_ratio",
        "target_modules",
    ):
        if config.get(key) is not None:
            explicit.add(key)
    return explicit


def _first_present(config: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def _optional_bool(config: dict[str, Any], key: str) -> bool | None:
    return bool(config[key]) if key in config and config[key] is not None else None


def _default_task_breadth(task: str) -> TaskBreadth:
    return "domain" if task == "financial_analyst_style" else "narrow"


def _example_token_length(example: AdaptationExample, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return max(1, len(example.prompt.split()) + len(example.completion.split()) + 1)
    prompt = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
    completion = tokenizer(example.completion, add_special_tokens=False)["input_ids"]
    return max(1, len(prompt) + len(completion) + 1)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 1
    index = max(0, math.ceil(percentile * len(values)) - 1)
    return values[index]
