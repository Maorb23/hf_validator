from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }


class TransformersSession:
    def __init__(self, model_id: str, revision: str | None, load_args: dict[str, Any]) -> None:
        trust_remote_code = bool(load_args.get("trust_remote_code", False))
        self.tokenizer = load_tokenizer(model_id, revision, trust_remote_code)
        self.model = load_model(
            model_id=model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            dtype=str(load_args.get("torch_dtype") or load_args.get("dtype") or "bf16"),
            attention_implementation=load_args.get("attn_implementation"),
            load_in_4bit=bool(load_args.get("load_in_4bit", False)),
        )

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        return deterministic_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

    def score_completion(self, prompt: str, completion: str) -> dict[str, Any]:
        return score_reference_completion(self.model, self.tokenizer, prompt, completion)


def load_tokenizer(
    model_id: str,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=None if _is_placeholder_revision(revision) else revision,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str,
    revision: str | None = None,
    trust_remote_code: bool = False,
    dtype: str = "bf16",
    attention_implementation: str | None = None,
    load_in_4bit: bool = False,
) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    load_kwargs: dict[str, Any] = {
        "revision": None if _is_placeholder_revision(revision) else revision,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit bitsandbytes loading requires an available CUDA GPU")
        compute_dtype = _torch_dtype(torch, dtype)
        load_kwargs["dtype"] = compute_dtype
        load_kwargs["device_map"] = "auto"
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif torch.cuda.is_available():
        load_kwargs["dtype"] = _torch_dtype(torch, dtype)
        load_kwargs["device_map"] = "auto"
        if attention_implementation and attention_implementation != "eager":
            load_kwargs["attn_implementation"] = attention_implementation
    else:
        load_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    # This runtime controls output length with max_new_tokens. Clear an inherited
    # total-length default so Transformers has only one length control.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    model.eval()
    return model


def deterministic_generate(
    model: Any,
    tokenizer: Any,
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 8,
) -> GenerationResult:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated_tokens = int(output.shape[-1] - prompt_tokens)
    text = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
    return GenerationResult(
        prompt=prompt,
        text=text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        elapsed_seconds=round(elapsed, 6),
    )


def score_reference_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> dict[str, Any]:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    combined_ids = tokenizer(
        prompt + completion,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]
    completion_tokens = int(combined_ids.numel() - prompt_ids.numel())
    if completion_tokens <= 0:
        raise ValueError("reference completion tokenized to zero tokens")
    max_context = int(
        getattr(model.config, "max_position_embeddings", 0)
        or getattr(model.config, "n_positions", 0)
        or 2048
    )
    available_prompt = max_context - completion_tokens
    if available_prompt <= 0:
        raise ValueError("reference completion exceeds the model context window")
    retained_prompt_tokens = min(int(prompt_ids.numel()), available_prompt)
    input_ids = combined_ids[-(retained_prompt_tokens + completion_tokens) :].unsqueeze(0)
    labels = input_ids.clone()
    labels[:, :retained_prompt_tokens] = -100
    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device)
    labels = labels.to(input_device)
    with torch.no_grad():
        output = model(input_ids=input_ids, labels=labels)
    cross_entropy = float(output.loss.detach().float().cpu())
    nll_nats = cross_entropy * completion_tokens
    byte_count = max(1, len(completion.encode("utf-8")))
    return {
        "reference_cross_entropy": cross_entropy,
        "reference_perplexity": math.exp(min(cross_entropy, 50.0)),
        "reference_negative_log_likelihood": nll_nats,
        "reference_bits_per_byte": nll_nats / (math.log(2) * byte_count),
        "reference_completion_tokens": completion_tokens,
        "reference_utf8_bytes": byte_count,
    }


def smoke_generate(
    model_id: str,
    revision: str | None,
    load_args: dict[str, Any],
    prompt: str = "Hello, my name is",
) -> GenerationResult:
    trust_remote_code = bool(load_args.get("trust_remote_code", False))
    tokenizer = load_tokenizer(model_id, revision, trust_remote_code)
    model = load_model(
        model_id=model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        dtype=str(load_args.get("torch_dtype") or "bf16"),
        attention_implementation=load_args.get("attn_implementation"),
        load_in_4bit=bool(load_args.get("load_in_4bit", False)),
    )
    return deterministic_generate(model, tokenizer, prompt=prompt)


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if dtype in {"bf16", "fp16"}:
        return torch.float16
    return torch.float32


def _is_placeholder_revision(revision: str | None) -> bool:
    return revision is None or revision.startswith("local-phase")
