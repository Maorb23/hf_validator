from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINANCIAL_ANALYST_STYLE = "financial_analyst_style"
FINANCE_ANALYSIS_ALIAS = "finance_analysis"
TASK_ALIASES = {FINANCE_ANALYSIS_ALIAS: FINANCIAL_ANALYST_STYLE}
FIXTURE_LOADERS: dict[str, str] = {
    "json_extraction": "load_json_extraction_dataset",
    "text_to_sql": "load_text_to_sql_dataset",
    FINANCIAL_ANALYST_STYLE: "load_finance_style_dataset",
}


@dataclass(frozen=True)
class AdaptationExample:
    task_type: str
    prompt: str
    completion: str


def load_json_extraction_dataset() -> list[AdaptationExample]:
    return [
        AdaptationExample(
            "json_extraction",
            'Extract JSON with name and city from: "Ada lives in London."',
            '{"name":"Ada","city":"London"}',
        ),
        AdaptationExample(
            "json_extraction",
            'Extract JSON with name and city from: "Grace lives in Arlington."',
            '{"name":"Grace","city":"Arlington"}',
        ),
    ]


def load_text_to_sql_dataset() -> list[AdaptationExample]:
    return [
        AdaptationExample(
            "text_to_sql",
            "Table users(id, active). Select active users.",
            "SELECT * FROM users WHERE active = 1;",
        ),
        AdaptationExample(
            "text_to_sql",
            "Table orders(id). Count all orders.",
            "SELECT COUNT(*) FROM orders;",
        ),
    ]


def load_finance_style_dataset() -> list[AdaptationExample]:
    return [
        AdaptationExample(
            FINANCIAL_ANALYST_STYLE,
            "Context: Revenue rose 5% and margin fell. Give a concise educational analyst note.",
            "Revenue improved, but lower margin is a risk. This is not financial advice.",
        ),
        AdaptationExample(
            FINANCIAL_ANALYST_STYLE,
            "Context: Debt increased while cash flow declined. Summarize with caveats.",
            "Higher debt and weaker cash flow increase risk; assumptions should be checked.",
        ),
    ]


def load_finance_analysis_dataset() -> list[AdaptationExample]:
    return load_finance_style_dataset()


def load_adaptation_dataset(
    task_type: str,
    config: dict[str, Any] | None = None,
    split: str = "train",
) -> list[AdaptationExample]:
    canonical_task = canonical_task_type(task_type)
    dataset_config = config or {}
    source = str(dataset_config.get("source") or "fixture").lower()
    if source == "disabled":
        return []
    if source == "fixture":
        return _load_fixture_dataset(canonical_task)
    if source == "local_jsonl":
        return _examples_from_rows(
            canonical_task,
            _load_local_jsonl(dataset_config, split),
            dataset_config,
            split,
        )
    if source == "huggingface":
        return _examples_from_rows(
            canonical_task,
            _load_huggingface_rows(dataset_config, split),
            dataset_config,
            split,
        )
    raise ValueError(
        "Unsupported adaptation dataset source "
        f"{source!r}; expected fixture, huggingface, local_jsonl, or disabled"
    )


def load_all_adaptation_datasets(
    datasets_config: dict[str, Any] | None = None,
    split: str = "train",
) -> dict[str, list[AdaptationExample]]:
    if datasets_config:
        return {
            canonical_task_type(task): load_adaptation_dataset(task, task_config, split)
            for task, task_config in datasets_config.items()
            if canonical_task_type(task) in FIXTURE_LOADERS
        }
    return {
        "json_extraction": load_json_extraction_dataset(),
        "text_to_sql": load_text_to_sql_dataset(),
        FINANCIAL_ANALYST_STYLE: load_finance_style_dataset(),
    }


def count_training_tokens(examples: list[AdaptationExample]) -> dict[str, int]:
    prompt_tokens = sum(len(example.prompt.split()) for example in examples)
    completion_tokens = sum(len(example.completion.split()) for example in examples)
    return {
        "training_examples": len(examples),
        "training_prompt_tokens": prompt_tokens,
        "training_completion_tokens": completion_tokens,
        "training_total_tokens": prompt_tokens + completion_tokens,
    }


def canonical_task_type(task_type: str) -> str:
    return TASK_ALIASES.get(task_type, task_type)


def _load_fixture_dataset(task_type: str) -> list[AdaptationExample]:
    loader_name = FIXTURE_LOADERS.get(task_type)
    if loader_name is None:
        raise ValueError(f"Unsupported adaptation task type for fixture dataset: {task_type}")
    return globals()[loader_name]()


def _load_local_jsonl(config: dict[str, Any], split: str = "train") -> list[dict[str, Any]]:
    path = config.get("local_jsonl_path")
    if split != "train":
        path = config.get("local_eval_jsonl_path") or path

    if not path:
        raise ValueError("local_jsonl adaptation dataset requires local_jsonl_path")
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record at line {line_number} must be an object: {path}")
            rows.append(row)
    return rows


def _load_huggingface_rows(config: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    dataset_name = config.get("hf_dataset_name")
    if not dataset_name:
        raise ValueError("huggingface adaptation dataset requires hf_dataset_name")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError(
            "huggingface adaptation dataset source requires the datasets package"
        ) from exc
    dataset_split = _resolved_split(config, split)
    hf_config_name = config.get("hf_config_name")
    if hf_config_name:
        return load_dataset(dataset_name, hf_config_name, split=dataset_split)
    return load_dataset(dataset_name, split=dataset_split)


def _examples_from_rows(
    task_type: str,
    rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
    split: str,
) -> list[AdaptationExample]:
    mapping = config.get("column_mapping") or {}
    prompt_column = mapping.get("prompt_column") or "prompt"
    completion_column = mapping.get("completion_column") or "completion"
    completion_fallback_column = mapping.get("completion_fallback_column")
    context_column = mapping.get("context_column")
    context_columns = mapping.get("context_columns") or []
    schema_column = mapping.get("schema_column")
    max_examples = _max_examples(config, split)
    examples = []
    for row in rows:
        _require_columns(row, [prompt_column, completion_column])
        prompt = _format_value(row[prompt_column])

        prefix_parts = []
        if context_column and row.get(context_column) not in (None, ""):
            prefix_parts.append(f"Context: {_format_value(row[context_column])}")
        for column in context_columns:
            if row.get(column) not in (None, ""):
                label = column.replace("_", " ").title()
                prefix_parts.append(f"{label}: {_format_value(row[column])}")
        if schema_column and row.get(schema_column) not in (None, ""):
            prefix_parts.append(f"Schema: {_format_value(row[schema_column])}")

        if prefix_parts:
            prompt = "\n".join(prefix_parts) + f"\nQuestion: {prompt}"

        completion = row[completion_column]
        if completion in (None, "") and completion_fallback_column:
            _require_columns(row, [completion_fallback_column])
            completion = row[completion_fallback_column]
        examples.append(AdaptationExample(task_type, prompt, _format_value(completion)))
        if max_examples is not None and len(examples) >= max_examples:
            break
    source = str(config.get("source") or "fixture").lower()
    if source in {"huggingface", "local_jsonl"} and not examples:
        raise ValueError(f"Adaptation dataset for {task_type} split {split} is empty")

    return examples


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        if value and all(isinstance(item, list) for item in value):
            return "\n".join(" | ".join(str(cell) for cell in row) for row in value)
        return "\n".join(_format_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _require_columns(row: dict[str, Any], columns: list[str]) -> None:
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Adaptation dataset row is missing required column(s): {missing}")


def _resolved_split(config: dict[str, Any], split: str) -> str:
    if split == "train":
        return str(config.get("train_split") or split)
    return str(config.get("eval_split") or split)


def _max_examples(config: dict[str, Any], split: str) -> int | None:
    key = "max_train_examples" if split == "train" else "max_eval_examples"
    value = config.get(key)
    return int(value) if value is not None else None
