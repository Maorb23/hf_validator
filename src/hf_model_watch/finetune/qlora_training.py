from __future__ import annotations

import gc
import json
import os
import re
from itertools import cycle
from pathlib import Path
from typing import Any

from hf_model_watch.finetune.datasets import AdaptationExample
from hf_model_watch.finetune.peft_config_resolver import resolve_target_modules
from hf_model_watch.tracing import emit_trace_event


def run_qlora_training(
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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_scheduler,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires an available CUDA GPU")
    set_seed(int(peft_config.get("seed") or 42))
    gc.collect()
    torch.cuda.empty_cache()
    examples = [example for task_examples in train_datasets.values() for example in task_examples]
    if not examples:
        raise ValueError("QLoRA requires at least one training example")

    model_revision = revision if revision and not revision.startswith("local-phase") else None
    compute_dtype = _compute_dtype(torch, peft_config)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(peft_config.get("bnb_4bit_quant_type") or "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires an eos or pad token for QLoRA training")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        device_map="auto",
        dtype=compute_dtype,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(peft_config.get("gradient_checkpointing", True)),
    )
    target_modules = resolve_target_modules(
        model,
        model.config.__class__.__name__,
        peft_config.get("target_modules", "all-linear"),
    )
    adapter_config = LoraConfig(
        r=int(peft_config["r"]),
        lora_alpha=int(peft_config["alpha"]),
        lora_dropout=float(peft_config["dropout"]),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, adapter_config)
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    if trainable_params <= 0:
        raise RuntimeError("no trainable adapter modules found")

    base_scores = _evaluate_tasks(model, tokenizer, eval_datasets, peft_config)
    encoded_examples = [
        _encode_training_example(tokenizer, example, int(peft_config["max_sequence_length"]))
        for example in examples
    ]
    optimizer, optimizer_name = _build_optimizer(model, peft_config)
    optimizer_steps = 1 if smoke_only else int(peft_config["max_steps"])
    gradient_accumulation = max(1, int(peft_config["gradient_accumulation"]))
    scheduler = get_scheduler(
        name=str(peft_config.get("scheduler") or "cosine"),
        optimizer=optimizer,
        num_warmup_steps=min(int(peft_config.get("warmup_steps") or 0), optimizer_steps),
        num_training_steps=optimizer_steps,
    )
    wandb_run = _start_wandb(run_id, model_id, peft_config, task_type)
    losses: list[float] = []
    micro_steps = 0
    completed_steps = 0
    input_device = model.get_input_embeddings().weight.device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        for batch in cycle(encoded_examples):
            moved = {key: value.unsqueeze(0).to(input_device) for key, value in batch.items()}
            output = model(**moved)
            loss = output.loss / gradient_accumulation
            loss.backward()
            micro_steps += 1
            if micro_steps % gradient_accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(peft_config.get("max_grad_norm") or 1.0),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            step_loss = float(loss.detach().item() * gradient_accumulation)
            losses.append(step_loss)
            emit_trace_event(
                "finetune_step",
                "finetune",
                "completed",
                "QLoRA optimizer step completed",
                metadata={
                    "run_id": run_id,
                    "model_id": model_id,
                    "task_type": task_type,
                    "step": completed_steps,
                },
                metrics={"loss": step_loss, "learning_rate": scheduler.get_last_lr()[0]},
            )
            if wandb_run is not None:
                wandb_run.log(
                    {"train/loss": step_loss, "train/learning_rate": scheduler.get_last_lr()[0]},
                    step=completed_steps,
                )
            if completed_steps >= optimizer_steps:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    model.eval()
    adapted_scores = _evaluate_tasks(model, tokenizer, eval_datasets, peft_config)
    adapter_dir = run_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise RuntimeError("adapter save completed without adapter_config.json")
    result = {
        "success": True,
        "message": "QLoRA training completed" if not smoke_only else "QLoRA smoke step completed",
        "adapter_dir": str(adapter_dir),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "optimizer": optimizer_name,
        "optimizer_steps": completed_steps,
        "micro_steps": micro_steps,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "base_scores": base_scores,
        "adapted_scores": adapted_scores,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": peft_config.get("bnb_4bit_quant_type"),
            "compute_dtype": str(compute_dtype),
            "double_quant": True,
        },
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _encode_training_example(
    tokenizer: Any,
    example: AdaptationExample,
    max_sequence_length: int,
) -> dict[str, Any]:
    import torch

    prompt_ids = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(example.completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = [*completion_ids, tokenizer.eos_token_id]
    completion_ids = completion_ids[: max(1, max_sequence_length - 1)]
    prompt_ids = prompt_ids[: max(0, max_sequence_length - len(completion_ids))]
    input_ids = [*prompt_ids, *completion_ids]
    labels = [-100] * len(prompt_ids) + completion_ids
    if not completion_ids:
        raise ValueError(f"training completion tokenized to zero tokens for {example.task_type}")
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


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
            encoded = tokenizer(
                example.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(peft_config["max_sequence_length"]),
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            max_new_tokens = min(128, max(8, len(example.completion.split()) * 3))
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
            start, end = prediction.index("{"), prediction.rindex("}") + 1
            return float(json.loads(prediction[start:end]) == json.loads(expected))
        except (ValueError, json.JSONDecodeError):
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


def _normalize_sql(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().rstrip(";").lower())


def _build_optimizer(model: Any, peft_config: dict[str, Any]) -> tuple[Any, str]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    kwargs = {
        "lr": float(peft_config["learning_rate"]),
        "weight_decay": float(peft_config.get("weight_decay") or 0.0),
    }
    try:
        from bitsandbytes.optim import PagedAdamW8bit

        return PagedAdamW8bit(parameters, **kwargs), "paged_adamw_8bit"
    except Exception:
        try:
            from bitsandbytes.optim import PagedAdamW32bit

            return PagedAdamW32bit(parameters, **kwargs), "paged_adamw_32bit"
        except Exception:
            import torch

            return torch.optim.AdamW(parameters, **kwargs), "adamw_torch"


def _compute_dtype(torch: Any, peft_config: dict[str, Any]) -> Any:
    requested = str(peft_config.get("bnb_4bit_compute_dtype") or "bfloat16")
    if requested == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _start_wandb(
    run_id: str,
    model_id: str,
    peft_config: dict[str, Any],
    task_type: str | None,
) -> Any | None:
    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb

        return wandb.init(
            project=os.getenv("WANDB_PROJECT", "hf-model-watch"),
            name=f"{run_id}-{task_type}" if task_type else run_id,
            group=run_id,
            config={"model_id": model_id, "task_type": task_type, **peft_config},
            reinit=True,
        )
    except Exception as exc:
        emit_trace_event(
            "wandb_initialization",
            "finetune",
            "failed",
            f"W&B initialization failed; training will continue: {type(exc).__name__}: {exc}",
            metadata={"run_id": run_id, "model_id": model_id},
        )
        return None
