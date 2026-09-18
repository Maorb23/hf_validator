from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalPrompt:
    prompt_id: str
    task_type: str
    prompt_family: str
    difficulty_bucket: str
    prompt: str
    max_new_tokens: int
    expected: dict[str, object]


DEFAULT_GROUND_TRUTH_DATASET = {
    "name": "openai/gsm8k",
    "config_name": "main",
    "split": "test",
}


def load_ground_truth_prompts(
    dataset_config: dict[str, Any] | None,
    benchmark: str,
) -> list[EvalPrompt]:
    config = {**DEFAULT_GROUND_TRUTH_DATASET, **(dataset_config or {})}
    benchmark_config = config.get(benchmark) or {}
    start = int(benchmark_config.get("start", 0))
    count = int(benchmark_config.get("count", 3))
    split = str(config["split"])
    sliced_split = f"{split}[{start}:{start + count}]"
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("ground-truth evaluation requires the datasets package") from exc

    rows = load_dataset(config["name"], config.get("config_name"), split=sliced_split)
    max_new_tokens = int(benchmark_config.get("max_new_tokens", 256))
    prompts = []
    for index, row in enumerate(rows, start=start):
        answer = str(row["answer"])
        prompts.append(
            EvalPrompt(
                prompt_id=f"gsm8k_{benchmark}_{index}",
                task_type="grade_school_math",
                prompt_family="ground_truth_math",
                difficulty_bucket="medium",
                prompt=(
                    "Solve this grade-school math problem. Show concise reasoning, then write "
                    "the final numeric answer after ####.\n\n"
                    f"Question: {row['question']}"
                ),
                max_new_tokens=max_new_tokens,
                expected={
                    "dataset": str(config["name"]),
                    "answer": answer,
                    "final_answer": _gsm8k_final_answer(answer),
                },
            )
        )
    if not prompts:
        raise ValueError(f"ground-truth dataset returned no rows for {sliced_split}")
    return prompts


def load_quality_prompts(config: dict[str, Any] | None) -> list[EvalPrompt]:
    quality = config or {}
    prompts: list[EvalPrompt] = []
    math_config = _task_config(quality, "math")
    sentence_config = _task_config(quality, "sentence_completion")
    if bool(math_config.get("enabled", False)):
        prompts.extend(_load_math_prompts(math_config))
    if bool(sentence_config.get("enabled", False)):
        prompts.extend(_load_sentence_completion_prompts(sentence_config))
    prompt_id_counts = Counter(prompt.prompt_id for prompt in prompts)
    duplicates = sorted(
        prompt_id for prompt_id, count in prompt_id_counts.items() if count > 1
    )
    if duplicates:
        raise ValueError(f"quality benchmark prompt IDs must be unique: {duplicates}")
    return prompts


def _load_math_prompts(config: dict[str, Any]) -> list[EvalPrompt]:
    rows = _load_local_jsonl(config, "math")
    prompts = []
    for row in rows:
        _require_fields(
            row,
            ("id", "question", "expected_answer", "answer_type", "source_url", "license"),
            "math",
        )
        _reject_placeholder(row, "math")
        answer_type = str(row["answer_type"])
        if answer_type not in {"integer", "decimal", "fraction"}:
            raise ValueError(
                f"math row {row['id']!r} has unsupported answer_type {answer_type!r}"
            )
        tolerance = _non_negative_decimal(row.get("absolute_tolerance", "0"), row["id"])
        prompts.append(
            EvalPrompt(
                prompt_id=str(row["id"]),
                task_type="custom_math",
                prompt_family="custom_math",
                difficulty_bucket=str(row.get("difficulty") or "unspecified"),
                prompt=(
                    "Solve the following math problem. Give concise reasoning and end with "
                    "`Final answer: <answer>`.\n\n"
                    f"{row['question']}"
                ),
                max_new_tokens=_positive_token_limit(row.get("max_new_tokens", 128), row["id"]),
                expected={
                    "final_answer": str(row["expected_answer"]),
                    "answer_type": answer_type,
                    "absolute_tolerance": str(tolerance),
                    "source_url": row.get("source_url"),
                    "license": row.get("license"),
                },
            )
        )
    return prompts


def _load_sentence_completion_prompts(config: dict[str, Any]) -> list[EvalPrompt]:
    rows = _load_local_jsonl(config, "sentence_completion")
    prompts = []
    for row in rows:
        _require_fields(
            row,
            ("id", "query", "references", "source_url", "license"),
            "sentence_completion",
        )
        _reject_placeholder(row, "sentence_completion")
        references = row["references"]
        if not isinstance(references, list) or len(references) < 2 or not all(
            isinstance(reference, str) and reference.strip() for reference in references
        ):
            raise ValueError(
                f"sentence_completion row {row['id']!r} requires at least two "
                "non-empty string references"
            )
        if len(set(references)) != len(references):
            raise ValueError(
                f"sentence_completion row {row['id']!r} contains duplicate references"
            )
        prompts.append(
            EvalPrompt(
                prompt_id=str(row["id"]),
                task_type="sentence_completion",
                prompt_family="online_discourse",
                difficulty_bucket=str(row.get("difficulty") or "unspecified"),
                prompt=(
                    "Complete the following query naturally. Return only the completion.\n\n"
                    f"Query: {row['query']}\nCompletion:\n"
                ),
                max_new_tokens=_positive_token_limit(row.get("max_new_tokens", 64), row["id"]),
                expected={
                    "references": references,
                    "language": row.get("language"),
                    "domain": row.get("domain"),
                    "source_url": row.get("source_url"),
                    "license": row.get("license"),
                },
            )
        )
    return prompts


def _task_config(config: dict[str, Any], task: str) -> dict[str, Any]:
    value = config.get(task) or {}
    if not isinstance(value, dict):
        raise ValueError(f"quality benchmark task {task!r} must be a mapping")
    return value


def _load_local_jsonl(config: dict[str, Any], task: str) -> list[dict[str, Any]]:
    raw_path = config.get("dataset_path")
    if not raw_path:
        raise ValueError(f"enabled quality benchmark task {task!r} requires dataset_path")
    path = Path(str(raw_path))
    if not path.is_file():
        raise FileNotFoundError(f"quality benchmark dataset does not exist: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"quality benchmark row must be an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"quality benchmark dataset is empty: {path}")
    return rows


def _require_fields(row: dict[str, Any], fields: tuple[str, ...], task: str) -> None:
    missing = [field for field in fields if row.get(field) in (None, "")]
    if missing:
        raise ValueError(
            f"{task} row {row.get('id', '<unknown>')!r} is missing required fields: {missing}"
        )


def _reject_placeholder(row: dict[str, Any], task: str) -> None:
    serialized = json.dumps(row, sort_keys=True)
    if re.search(r"<[A-Z][A-Z0-9 _,.-]*>", serialized):
        raise ValueError(
            f"{task} row {row.get('id', '<unknown>')!r} still contains placeholder values"
        )


def _positive_token_limit(value: Any, row_id: Any) -> int:
    limit = int(value)
    if limit <= 0:
        raise ValueError(f"quality benchmark row {row_id!r} requires max_new_tokens > 0")
    return limit


def _non_negative_decimal(value: Any, row_id: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"math row {row_id!r} has invalid absolute_tolerance") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(
            f"math row {row_id!r} requires a finite, non-negative absolute_tolerance"
        )
    return parsed


def _gsm8k_final_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+?)\s*$", answer)
    return match.group(1) if match else answer.strip()


CONTEXT_PROMPTS = [
    EvalPrompt(
        "ctx_needle_short",
        "needle_retrieval",
        "context",
        "short",
        "Remember this code: NEEDLE42. What code did I ask you to remember?",
        32,
        {"must_contain": ["NEEDLE42"]},
    ),
    EvalPrompt(
        "ctx_contradiction_medium",
        "contradiction_detection",
        "context",
        "medium",
        (
            "Document A says the launch is Monday. Document B says the launch is Friday. "
            "Identify the contradiction."
        ),
        64,
        {"must_contain_any": ["Monday", "Friday", "contradiction"]},
    ),
    EvalPrompt(
        "ctx_instruction_retention",
        "instruction_retention",
        "context",
        "long",
        (
            "Instruction: answer with exactly the word blue. Extra context: red green yellow. "
            "Answer now."
        ),
        16,
        {"must_contain": ["blue"]},
    ),
]
