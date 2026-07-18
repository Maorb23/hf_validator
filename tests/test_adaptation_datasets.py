from __future__ import annotations

import json
import sys
import types

import pytest

from hf_model_watch.finetune.datasets import (
    load_adaptation_dataset,
    load_all_adaptation_datasets,
    load_finance_analysis_dataset,
    load_finance_style_dataset,
    load_json_extraction_dataset,
)


def test_fixture_loading_still_works() -> None:
    examples = load_json_extraction_dataset()

    assert examples
    assert examples[0].task_type == "json_extraction"


def test_canonical_financial_analyst_style_fixture_works() -> None:
    examples = load_adaptation_dataset("financial_analyst_style", {"source": "fixture"})

    assert examples
    assert {example.task_type for example in examples} == {"financial_analyst_style"}
    assert load_finance_style_dataset() == examples


def test_old_finance_analysis_alias_still_works() -> None:
    examples = load_adaptation_dataset("finance_analysis", {"source": "fixture"})

    assert examples == load_finance_analysis_dataset()
    assert {example.task_type for example in examples} == {"financial_analyst_style"}


def test_local_jsonl_loading_works(tmp_path) -> None:
    path = tmp_path / "finance.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"question": "What changed?", "answer": "Revenue rose.", "context": "Q1"}
                ),
                json.dumps({"question": "Risk?", "answer": "Margins fell.", "context": "Q2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_adaptation_dataset(
        "financial_analyst_style",
        {
            "source": "local_jsonl",
            "local_jsonl_path": str(path),
            "max_train_examples": 10,
            "column_mapping": {
                "prompt_column": "question",
                "completion_column": "answer",
                "context_column": "context",
            },
        },
    )

    assert len(examples) == 2
    assert examples[0].prompt == "Context: Q1\nQuestion: What changed?"
    assert examples[0].completion == "Revenue rose."


def test_mocked_huggingface_loading_works(monkeypatch) -> None:
    calls = {}

    def load_dataset(name, config_name=None, split=None):
        calls.update({"name": name, "config_name": config_name, "split": split})
        return [{"prompt": "p", "completion": "c"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))

    examples = load_adaptation_dataset(
        "json_extraction",
        {
            "source": "huggingface",
            "hf_dataset_name": "org/dataset",
            "hf_config_name": "subset",
            "train_split": "train[:10]",
            "column_mapping": {"prompt_column": "prompt", "completion_column": "completion"},
        },
    )

    assert calls == {"name": "org/dataset", "config_name": "subset", "split": "train[:10]"}
    assert examples[0].prompt == "p"
    assert examples[0].completion == "c"


def test_structured_context_columns_and_completion_fallback_are_formatted(tmp_path) -> None:
    path = tmp_path / "finqa.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "What is the result?",
                "answer": "",
                "final_result": "12.5%",
                "pre_text": ["Annual report"],
                "table": [["Year", "Margin"], ["2025", "12.5%"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_adaptation_dataset(
        "financial_analyst_style",
        {
            "source": "local_jsonl",
            "local_jsonl_path": str(path),
            "column_mapping": {
                "prompt_column": "question",
                "completion_column": "answer",
                "completion_fallback_column": "final_result",
                "context_columns": ["pre_text", "table"],
            },
        },
    )

    assert examples[0].completion == "12.5%"
    assert "Pre Text: Annual report" in examples[0].prompt
    assert "Year | Margin\n2025 | 12.5%" in examples[0].prompt


def test_max_examples_is_enforced_for_eval_split(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        "".join(json.dumps({"prompt": f"p{i}", "completion": "c"}) + "\n" for i in range(3)),
        encoding="utf-8",
    )

    examples = load_adaptation_dataset(
        "json_extraction",
        {
            "source": "local_jsonl",
            "local_jsonl_path": str(path),
            "max_train_examples": 3,
            "max_eval_examples": 1,
        },
        split="validation",
    )

    assert len(examples) == 1


def test_missing_required_columns_raises_clear_error(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"prompt": "p"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required column"):
        load_adaptation_dataset(
            "json_extraction",
            {"source": "local_jsonl", "local_jsonl_path": str(path)},
        )


def test_unsupported_source_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported adaptation dataset source"):
        load_adaptation_dataset("json_extraction", {"source": "warehouse"})


def test_load_all_accepts_config_and_alias() -> None:
    datasets = load_all_adaptation_datasets(
        {
            "finance_analysis": {"source": "fixture"},
            "json_extraction": {"source": "disabled"},
        }
    )

    assert set(datasets) == {"financial_analyst_style", "json_extraction"}
    assert datasets["financial_analyst_style"]
    assert datasets["json_extraction"] == []
