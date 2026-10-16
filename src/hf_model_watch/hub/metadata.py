from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

CAUSAL_LM_ARCHITECTURE_HINTS = (
    "ForCausalLM",
    "CausalLM",
    "GPT",
    "Llama",
    "Mistral",
    "Qwen",
    "Gemma",
    "Phi",
    "Yi",
    "Deepseek",
    "DeepSeek",
)

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
)


def normalize_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def extract_author(model_id: str) -> str | None:
    if "/" not in model_id:
        return None
    return model_id.split("/", 1)[0]


def sibling_names(siblings: list[Any] | None) -> list[str]:
    names: list[str] = []
    for sibling in siblings or []:
        name = getattr(sibling, "rfilename", None)
        if name is None and isinstance(sibling, dict):
            name = sibling.get("rfilename") or sibling.get("path")
        if name:
            names.append(str(name))
    return names


def has_tokenizer_file(files: list[str]) -> bool:
    return any(Path(file_name).name in TOKENIZER_FILENAMES for file_name in files)


def has_config_file(files: list[str]) -> bool:
    return "config.json" in {Path(file_name).name for file_name in files}


def has_weight_file(files: list[str]) -> bool:
    return any(file_name.endswith(WEIGHT_SUFFIXES) for file_name in files)


def has_safetensors_file(files: list[str]) -> bool:
    return any(file_name.endswith(".safetensors") for file_name in files)


def estimate_download_size_gb(siblings: list[Any] | None) -> float | None:
    total_bytes = 0
    found_size = False
    for sibling in siblings or []:
        name = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if isinstance(sibling, dict):
            name = name or sibling.get("rfilename") or sibling.get("path")
            size = size or sibling.get("size")
        if name and str(name).endswith(WEIGHT_SUFFIXES) and isinstance(size, int | float):
            total_bytes += int(size)
            found_size = True
    if not found_size:
        return None
    return round(total_bytes / 1024**3, 3)


def extract_license(card_data: Any, tags: list[str]) -> str | None:
    if isinstance(card_data, dict):
        license_value = card_data.get("license")
        if isinstance(license_value, str):
            return license_value.lower()
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1].lower()
    return None


def estimate_params_billion(config: dict[str, Any], tags: list[str], model_id: str) -> float | None:
    explicit = _coerce_float(config.get("num_parameters") or config.get("n_params"))
    if explicit is not None:
        return round(explicit / 1_000_000_000 if explicit > 1_000_000 else explicit, 3)

    text = " ".join([model_id, *tags]).lower()
    import re

    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?:[^a-z]|$)", text)
    if match:
        return float(match.group(1))
    return None


def parameter_estimate(config, tags, model_id, stored_elements=None):
    """Quantized tensor elements are not the uncompressed model parameter count."""
    import re
    quantized = bool(config.get("quantization_config")) or bool(re.search(
        r"fp[48]|nvfp4|gptq|awq|gguf|[248]bit|int[48]", model_id, re.I))
    if quantized:
        named = estimate_params_billion({}, [], model_id)
        explicit = estimate_params_billion(config, [], "")
        return explicit or named  # Unknown is preferable to a packed tensor count.
    return stored_elements or estimate_params_billion(config, tags, model_id)


def extract_architectures(config: dict[str, Any]) -> list[str]:
    architectures = config.get("architectures") or []
    if isinstance(architectures, str):
        return [architectures]
    if isinstance(architectures, list):
        return [str(item) for item in architectures]
    return []


def looks_like_causal_lm(config: dict[str, Any], pipeline_tag: str | None, tags: list[str]) -> bool:
    if pipeline_tag == "text-generation":
        return True
    architectures = extract_architectures(config)
    if any(
        any(hint in architecture for hint in CAUSAL_LM_ARCHITECTURE_HINTS)
        for architecture in architectures
    ):
        return True
    tag_text = " ".join(tags).lower()
    return "text-generation" in tag_text or "causal-lm" in tag_text


def requires_trust_remote_code(config: dict[str, Any], tags: list[str]) -> bool:
    auto_map = config.get("auto_map")
    if isinstance(auto_map, dict) and auto_map:
        return True
    return any(tag.lower() == "custom_code" or tag.lower() == "trust_remote_code" for tag in tags)


def extract_context_length(config: dict[str, Any], tags: list[str]) -> int | None:
    for key in (
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "model_max_length",
        "max_sequence_length",
    ):
        value = _coerce_int(config.get(key))
        if value:
            return value
    tag_text = " ".join(tags).lower()
    import re

    match = re.search(r"(\d+)\s*k(?:-)?context", tag_text)
    if match:
        return int(match.group(1)) * 1024
    return None


def tokenizer_has_chat_template(tokenizer_config: dict[str, Any]) -> bool:
    return bool(tokenizer_config.get("chat_template"))


def read_json_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
