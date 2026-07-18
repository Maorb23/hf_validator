from __future__ import annotations

import re
from dataclasses import dataclass
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
