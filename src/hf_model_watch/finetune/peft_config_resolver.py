from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KNOWN_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
}


@dataclass(frozen=True)
class PeftResolvedConfig:
    method: str
    optimizer: str
    optimizer_fallback: str
    scheduler: str
    r: int
    alpha: int
    dropout: float
    target_modules: str | list[str]
    batch_size: int
    gradient_accumulation: int
    effective_batch_size: int
    max_sequence_length: int
    learning_rate: float
    max_steps: int
    warmup_steps: int
    max_grad_norm: float
    weight_decay: float
    bnb_4bit_quant_type: str | None
    bnb_4bit_compute_dtype: str | None
    bf16: bool
    gradient_checkpointing: bool
    seed: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "peft_method": self.method,
            "method": self.method,
            "optimizer": self.optimizer,
            "optimizer_fallback": self.optimizer_fallback,
            "scheduler": self.scheduler,
            "r": self.r,
            "rank": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "max_sequence_length": self.max_sequence_length,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "reason": self.reason,
        }


def resolve_peft_config(
    model_size_billion: float,
    architecture: str | list[str] | None,
    available_vram_gb: float,
    task: str,
    desired_budget: str = "standard",
    context_length: int = 2048,
    quantization_mode: str | None = None,
    finetune_config: dict[str, Any] | None = None,
) -> PeftResolvedConfig:
    cfg = finetune_config or {}
    if model_size_billion <= 7:
        method = (
            "lora"
            if quantization_mode in {None, "none"} and available_vram_gb >= 40
            else "qlora"
        )
        r, alpha, lr, effective_batch = 16, 32, 2e-4, 32
        reason = "model estimated <=7B; using LoRA when memory allows, otherwise QLoRA"
    elif model_size_billion <= 15:
        method = "qlora"
        r, alpha, lr, effective_batch = 16, 32, 2e-4, 16
        reason = "model estimated 7B-15B; using QLoRA to fit H100 memory"
    else:
        method = "qlora"
        r, alpha, lr, effective_batch = (
            (8, 16, 1e-4, 8) if model_size_billion > 22 else (16, 32, 2e-4, 16)
        )
        reason = "model estimated 15B-30B; using lower-rank QLoRA policy when needed"

    method = str(cfg.get("method") or cfg.get("default_method") or method).lower()
    if quantization_mode == "4bit":
        method = "qlora"
    optimizer = "paged_adamw_8bit" if method == "qlora" else "adamw_torch_fused"
    optimizer_fallback = "paged_adamw_32bit" if method == "qlora" else "adamw_torch"
    max_steps = int(cfg.get("max_train_steps") or cfg.get("max_steps") or 100)
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    batch_size = 1
    gradient_accumulation = max(1, effective_batch // batch_size)
    return PeftResolvedConfig(
        method=method,
        optimizer=optimizer,
        optimizer_fallback=optimizer_fallback,
        scheduler=str(cfg.get("scheduler", "cosine")),
        r=int(cfg.get("lora_rank", r)),
        alpha=int(cfg.get("lora_alpha", alpha)),
        dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=cfg.get("target_modules", "all-linear"),
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        effective_batch_size=effective_batch,
        max_sequence_length=min(context_length, 2048),
        learning_rate=float(cfg.get("learning_rate", lr)),
        max_steps=max_steps,
        warmup_steps=max(5, int(warmup_ratio * max_steps)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        bnb_4bit_quant_type="nf4" if method == "qlora" else None,
        bnb_4bit_compute_dtype="bfloat16" if method == "qlora" else None,
        bf16=bool(cfg.get("bf16", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        seed=int(cfg.get("seed", 42)),
        reason=f"{reason}; task={task}; budget={desired_budget}",
    )


def discover_linear_module_names(model: Any) -> list[str]:
    linear_class_names = {"Linear", "Linear4bit", "Linear8bitLt", "Conv1D"}
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if module.__class__.__name__ in linear_class_names and module_name:
            names.add(module_name.split(".")[-1])
    return sorted(names)


def resolve_target_modules(
    model: Any | None,
    architecture: str | list[str] | None,
    requested: str | list[str] = "all-linear",
) -> str | list[str]:
    if requested != "all-linear":
        return requested
    if model is not None:
        discovered = discover_linear_module_names(model)
        if discovered:
            return discovered
    arch_parts = architecture if isinstance(architecture, list) else [str(architecture or "")]
    arch_text = " ".join(arch_parts).lower()
    for key, modules in KNOWN_TARGET_MODULES.items():
        if key in arch_text:
            return modules
    raise ValueError("no trainable adapter modules found")
