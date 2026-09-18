from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Any

from hf_model_watch.finetune.datasets import AdaptationExample
from hf_model_watch.finetune.model_profile import count_unique_parameters, profile_model_config
from hf_model_watch.tracing import emit_trace_event


def run_peft_training(
    run_id: str,
    model_id: str,
    revision: str | None,
    run_dir: Path,
    train_datasets: dict[str, list[AdaptationExample]],
    eval_datasets: dict[str, list[AdaptationExample]],
    peft_config: dict[str, Any],
    smoke_only: bool = False,
    task_type: str | None = None,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("PEFT training requires an available CUDA GPU")
    method = str(peft_config.get("method") or peft_config.get("peft_method") or "qlora")
    if method not in {"lora", "qlora"}:
        raise ValueError(f"unsupported resolved PEFT method: {method}")
    wandb_enabled = bool(peft_config.get("wandb_enabled", True))
    wandb_project = str(
        os.environ.get("WANDB_PROJECT")
        or peft_config.get("wandb_project")
        or "hf-model-watch"
    )
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_GROUP"] = run_id
    os.environ["WANDB_JOB_TYPE"] = task_type or "finetune"
    attempt_number = int(peft_config.get("attempt_number") or 1)
    wandb_run_name = _wandb_run_name(run_id, task_type, method, attempt_number)
    _start_wandb(
        enabled=wandb_enabled,
        project=wandb_project,
        run_name=wandb_run_name,
        run_id=run_id,
        task_type=task_type,
        config={
            "model_id": model_id,
            "revision": revision,
            "method": method,
            "attempt_number": attempt_number,
            "resolved_policy": peft_config,
        },
    )
    set_seed(int(peft_config.get("seed") or 42))
    gc.collect()
    torch.cuda.empty_cache()

    examples = [example for values in train_datasets.values() for example in values]
    if not examples:
        raise ValueError("PEFT training requires at least one training example")
    model_revision = revision if revision and not revision.startswith("local-phase") else None
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires an eos or pad token for PEFT training")
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = _compute_dtype(torch, peft_config)
    model_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "device_map": {"": 0},
        "dtype": compute_dtype,
    }
    if bool(peft_config.get("packing", False)):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    quantization: dict[str, Any] | None = None
    if method == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(peft_config.get("bnb_4bit_quant_type") or "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(
                peft_config.get("bnb_4bit_use_double_quant", True)
            ),
        )
        model_kwargs["quantization_config"] = quantization_config
        quantization = {
            "load_in_4bit": True,
            "quant_type": peft_config.get("bnb_4bit_quant_type") or "nf4",
            "compute_dtype": str(compute_dtype),
            "double_quant": bool(peft_config.get("bnb_4bit_use_double_quant", True)),
        }

    raw_config = AutoConfig.from_pretrained(model_id, revision=model_revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    # Evaluation supplies max_new_tokens per example. Some repositories persist
    # the context window as a generation max_length, which conflicts with that
    # explicit budget and produces one Transformers warning per sample.
    model.generation_config.max_length = None
    model.config.use_cache = False
    exact_total_params = count_unique_parameters(model)
    model_profile = profile_model_config(
        raw_config,
        estimated_params_billion=exact_total_params / 1_000_000_000,
    )
    model_profile_data = model_profile.as_dict()

    emit_trace_event(
        "finetune_baseline_eval_started",
        "finetune",
        "started",
        "evaluating the base model before adapter training",
        metadata={"run_id": run_id, "model_id": model_id, "task_type": task_type},
        metrics={"eval_examples": sum(len(items) for items in eval_datasets.values())},
    )
    _log_wandb(wandb_enabled, {"monitoring/baseline_eval_started": 1})
    base_scores = _evaluate_tasks(model, tokenizer, eval_datasets, peft_config)
    _log_wandb(
        wandb_enabled,
        {f"baseline/{task}_score": score for task, score in base_scores.items()},
    )
    emit_trace_event(
        "finetune_baseline_eval_completed",
        "finetune",
        "completed",
        "base-model evaluation completed",
        metadata={"run_id": run_id, "model_id": model_id, "task_type": task_type},
        metrics={f"{task}_score": score for task, score in base_scores.items()},
    )
    gc.collect()
    torch.cuda.empty_cache()
    dataset = Dataset.from_list(
        [{"prompt": item.prompt, "completion": item.completion} for item in examples]
    )
    eval_examples = [item for values in eval_datasets.values() for item in values]
    eval_dataset = (
        Dataset.from_list(
            [{"prompt": item.prompt, "completion": item.completion} for item in eval_examples]
        )
        if eval_examples
        else None
    )
    rank = int(peft_config.get("rank") or peft_config.get("r") or 16)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=int(peft_config.get("alpha") or rank),
        lora_dropout=float(peft_config.get("dropout") or 0.0),
        target_modules=peft_config.get("target_modules") or "all-linear",
        exclude_modules=peft_config.get("exclude_modules") or None,
        use_rslora=bool(peft_config.get("use_rslora", rank >= 16)),
        fan_in_fan_out=bool(peft_config.get("fan_in_fan_out", False)),
        task_type="CAUSAL_LM",
        bias="none",
    )
    optimizer_steps = 1 if smoke_only else int(peft_config.get("max_steps") or 100)
    eval_steps = min(optimizer_steps, max(1, int(peft_config.get("eval_steps") or 5)))
    logging_steps = min(
        optimizer_steps, max(1, int(peft_config.get("logging_steps") or 1))
    )
    optimizer_name, optimizer_reason = _select_optimizer(peft_config)
    output_dir = run_dir / "trainer"
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(
            peft_config.get("microbatch_size") or peft_config.get("batch_size") or 1
        ),
        per_device_eval_batch_size=max(
            1, int(peft_config.get("eval_microbatch_size") or 1)
        ),
        gradient_accumulation_steps=max(
            1, int(peft_config.get("gradient_accumulation") or 1)
        ),
        max_length=int(peft_config.get("max_sequence_length") or 2048),
        packing=bool(peft_config.get("packing", True)),
        packing_strategy=str(peft_config.get("packing_strategy") or "bfd_split"),
        padding_free=False,
        completion_only_loss=True,
        learning_rate=float(peft_config.get("learning_rate") or 1e-4),
        max_steps=optimizer_steps,
        warmup_ratio=float(peft_config.get("warmup_ratio") or 0.0),
        max_grad_norm=float(peft_config.get("max_grad_norm") or 0.3),
        weight_decay=float(peft_config.get("weight_decay") or 0.0),
        lr_scheduler_type=str(peft_config.get("scheduler") or "cosine"),
        router_aux_loss_coef=float(peft_config.get("router_aux_loss_coef") or 0.001),
        gradient_checkpointing=bool(peft_config.get("gradient_checkpointing", True)),
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim=optimizer_name,
        save_strategy="no",
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps if eval_dataset is not None else None,
        prediction_loss_only=True,
        logging_steps=logging_steps,
        logging_first_step=True,
        report_to="wandb" if wandb_enabled else "none",
        run_name=wandb_run_name,
        seed=int(peft_config.get("seed") or 42),
    )
    memory_callback = _memory_guard_callback(
        torch,
        float(peft_config.get("max_memory_fraction") or 0.85),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[memory_callback],
    )
    trainer.log(
        {f"eval_baseline_{task}_score": score for task, score in base_scores.items()}
    )
    torch.cuda.reset_peak_memory_stats()
    try:
        train_output = trainer.train()
    except Exception:
        _finish_wandb(wandb_enabled)
        raise
    torch.cuda.synchronize()
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_vram = int(torch.cuda.get_device_properties(0).total_memory)
    if memory_callback.threshold_exceeded:
        trainer.log(
            {
                "train_peak_allocated_bytes": peak_allocated,
                "train_peak_reserved_bytes": peak_reserved,
                "train_total_vram_bytes": total_vram,
                "train_memory_threshold_exceeded": 1,
            }
        )
        _finish_wandb(wandb_enabled)
        return {
            "success": False,
            "retry_recommended": True,
            "retry_reason": "memory_threshold_exceeded",
            "message": (
                f"{method} exceeded the configured memory threshold after its first step"
            ),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_vram_bytes": total_vram,
            "model_profile": model_profile_data,
        }

    trained_model = trainer.model
    trainable_params = sum(
        parameter.numel() for parameter in trained_model.parameters() if parameter.requires_grad
    )
    adapted_scores = _evaluate_tasks(trained_model, tokenizer, eval_datasets, peft_config)
    trainer.log(
        {
            **{
                f"eval_adapted_{task}_score": score
                for task, score in adapted_scores.items()
            },
            "train_peak_allocated_bytes": peak_allocated,
            "train_peak_reserved_bytes": peak_reserved,
            "train_total_vram_bytes": total_vram,
        }
    )
    adapter_dir = run_dir / "adapter"
    trained_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError("adapter save completed without adapter_config.json")
    completed_steps = int(trainer.state.global_step)
    losses = [
        float(item["loss"])
        for item in trainer.state.log_history
        if item.get("loss") is not None
    ]
    eval_history = [
        dict(item) for item in trainer.state.log_history if item.get("eval_loss") is not None
    ]
    result = {
        "success": True,
        "retry_recommended": False,
        "message": (
            f"{method.upper()} training completed"
            if not smoke_only
            else f"{method.upper()} smoke step completed"
        ),
        "adapter_dir": str(adapter_dir),
        "trainable_params": trainable_params,
        "total_params": exact_total_params,
        "optimizer": optimizer_name,
        "optimizer_reason": optimizer_reason,
        "optimizer_steps": completed_steps,
        "losses": losses,
        "final_loss": losses[-1] if losses else float(train_output.training_loss),
        "eval_history": eval_history,
        "base_scores": base_scores,
        "adapted_scores": adapted_scores,
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "quantization": quantization,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "model_profile": model_profile_data,
    }
    _finish_wandb(wandb_enabled)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _wandb_run_name(
    run_id: str, task_type: str | None, method: str, attempt_number: int
) -> str:
    task = task_type or "finetune"
    return f"{run_id}-{task}-{method}-attempt-{attempt_number}"


def _start_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    run_id: str,
    task_type: str | None,
    config: dict[str, Any],
) -> None:
    if not enabled:
        return
    import wandb

    wandb.init(
        project=project,
        name=run_name,
        group=run_id,
        job_type=task_type or "finetune",
        config=config,
    )


def _log_wandb(enabled: bool, metrics: dict[str, int | float]) -> None:
    if not enabled:
        return
    import wandb

    if wandb.run is not None:
        wandb.log(metrics)


def _finish_wandb(enabled: bool) -> None:
    if not enabled:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        # W&B finalization must not discard an otherwise successful adapter.
        return


def _memory_guard_callback(torch: Any, max_fraction: float) -> Any:
    from transformers import TrainerCallback

    class MemoryGuardCallback(TrainerCallback):
        threshold_exceeded = False

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if state.global_step != 1:
                return control
            torch.cuda.synchronize()
            reserved = int(torch.cuda.max_memory_reserved())
            total = int(torch.cuda.get_device_properties(0).total_memory)
            if reserved > total * max_fraction:
                self.threshold_exceeded = True
                control.should_training_stop = True
            return control

    return MemoryGuardCallback()


def _compute_dtype(torch: Any, peft_config: dict[str, Any]) -> Any:
    if not bool(peft_config.get("bf16", True)):
        return torch.float16
    requested = str(peft_config.get("bnb_4bit_compute_dtype") or "bfloat16")
    if requested == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _select_optimizer(peft_config: dict[str, Any]) -> tuple[str, str]:
    requested = str(peft_config.get("optimizer") or "adamw_torch_fused")
    explicit = "optimizer" in set(peft_config.get("explicit_overrides") or [])
    candidates = [
        requested,
        str(peft_config.get("optimizer_fallback") or ""),
        "adamw_torch_fused",
        "adamw_torch",
    ]
    for candidate in dict.fromkeys(name for name in candidates if name):
        if _optimizer_available(candidate):
            if candidate == requested:
                return candidate, "resolved optimizer is available"
            if explicit:
                raise RuntimeError(f"explicit optimizer is unavailable: {requested}")
            return candidate, f"{requested} unavailable; using {candidate}"
    raise RuntimeError(f"no supported optimizer is available after trying: {candidates}")


def _optimizer_available(name: str) -> bool:
    if name == "paged_adamw_8bit":
        try:
            from bitsandbytes.optim import PagedAdamW8bit  # noqa: F401

            return True
        except Exception:
            return False
    if name == "paged_adamw_32bit":
        try:
            from bitsandbytes.optim import PagedAdamW32bit  # noqa: F401

            return True
        except Exception:
            return False
    return name in {"adamw_torch_fused", "adamw_torch"}


def _evaluate_tasks(
    model: Any,
    tokenizer: Any,
    datasets: dict[str, list[AdaptationExample]],
    peft_config: dict[str, Any],
) -> dict[str, float]:
    import torch

    scores: dict[str, float] = {}
    input_device = model.get_input_embeddings().weight.device
    model.eval()
    for task_type, examples in datasets.items():
        task_scores = []
        for example in examples:
            max_new_tokens = _evaluation_generation_budget(
                task_type,
                tokenizer,
                example.completion,
                peft_config,
            )
            encoded = tokenizer(
                example.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(peft_config.get("max_sequence_length") or 2048),
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = output[0, encoded["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
            task_scores.append(_score_prediction(task_type, prediction, example.completion))
        scores[task_type] = sum(task_scores) / max(len(task_scores), 1)
    return scores


def _score_prediction(task_type: str, prediction: str, expected: str) -> float:
    if task_type == "json_extraction":
        try:
            return float(_parse_first_json_value(prediction) == json.loads(expected))
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0
    if task_type == "text_to_sql":
        return float(_normalize_sql(prediction) == _normalize_sql(expected))
    predicted_tokens = set(re.findall(r"\w+", prediction.lower()))
    expected_tokens = set(re.findall(r"\w+", expected.lower()))
    if not predicted_tokens or not expected_tokens:
        return 0.0
    precision = len(predicted_tokens & expected_tokens) / len(predicted_tokens)
    recall = len(predicted_tokens & expected_tokens) / len(expected_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _evaluation_generation_budget(
    task_type: str,
    tokenizer: Any,
    expected: str,
    peft_config: dict[str, Any],
) -> int:
    """Size evaluation generations from model tokens, not whitespace-delimited words."""
    if task_type != "json_extraction":
        return min(128, max(8, len(expected.split()) * 3))

    encoded = tokenizer(expected, add_special_tokens=False)
    token_ids = encoded.get("input_ids", []) if hasattr(encoded, "get") else []
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    expected_tokens = max(1, len(token_ids))
    multiplier = max(
        1.0,
        float(peft_config.get("json_eval_generation_token_multiplier") or 2.0),
    )
    minimum = max(1, int(peft_config.get("json_eval_min_new_tokens") or 32))
    maximum = max(
        minimum,
        int(peft_config.get("json_eval_max_new_tokens") or 256),
    )
    return min(maximum, max(minimum, int(expected_tokens * multiplier)))


def _parse_first_json_value(text: str) -> Any:
    """Return the first complete JSON object or array embedded in model output."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    raise ValueError("model output does not contain a complete JSON object or array")


def _normalize_sql(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().rstrip(";").lower())
