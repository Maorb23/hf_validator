"""Read-only, model-independent task sizing guidance supplied by the user."""
from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskSizeInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    task: str | None = Field(default=None, min_length=1, max_length=4000)
    use_case_id: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def needs_lookup(self):
        if not self.task and not self.use_case_id:
            raise ValueError("Supply task or use_case_id")
        return self


def normalize(text: str) -> str:
    return " ".join("".join(
        char if char.isalnum() else " "
        for char in unicodedata.normalize("NFKC", text).casefold()
    ).split())


def lookup_task_size(**raw: Any) -> dict[str, Any]:
    inputs = TaskSizeInputs.model_validate(raw)
    catalog = json.loads(Path(__file__).with_name("task_size_catalog.json").read_text())
    entries = catalog["use_cases"]
    result: dict[str, Any] = {
        "catalog_version": catalog["schema_version"],
        "basis": "user_supplied_heuristic_not_benchmark_ranking",
        "status": "needs_clarification", "selected": None, "candidates": [],
        "guidance": "Size ranges are starting points, not capability guarantees. Preserve explicit user size constraints. Check hardware, license, context, modality and benchmark results before selection.",
    }
    if inputs.use_case_id:
        matches = [e for e in entries if e["id"] == inputs.use_case_id]
        if not matches:
            raise ValueError("Unknown use_case_id")
        reason = "exact_use_case_id"
    else:
        text = normalize(inputs.task or "")
        padded = f" {text} "
        matches = [e for e in entries if any(
            f" {normalize(alias)} " in padded
            for alias in [e["id"], e["use_case"], *e["match"]["aliases"]]
        )]
        reason = "complete_alias"
        if not matches:
            # Overlap retrieves options only; even a single option needs confirmation.
            tokens = set(text.split())
            scored = [(len(tokens & set(normalize(" ".join(
                [e["use_case"], *e["match"]["keywords"], *e["match"]["aliases"],
                 *e["match"]["example_requests"]]
            )).split())), e) for e in entries]
            matches = [e for score, e in sorted(scored, key=lambda row: -row[0]) if score >= 2][:5]
            reason = "keyword_shortlist_only"
    def view(entry):
        return {
            **{key: entry[key] for key in
               ("id", "use_case", "model_type", "recommended_size", "deployment_note")},
            "example_requests": entry["match"]["example_requests"],
        }
    result["match_reason"] = reason
    result["candidates"] = [view(e) for e in matches]
    if len(matches) == 1 and reason != "keyword_shortlist_only":
        result.update(status="matched", selected=view(matches[0]))
    else:
        result["clarification"] = "Which use case describes the task? Select a candidate ID or describe the intended inputs and outputs. Do not infer a single size range from keyword overlap."
    return result


TASK_SIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_task_size",
        "description": "Look up heuristic parameter-size ranges for a task. Returns no model names. Use when size is unspecified; ambiguous matches require clarification or semantic disambiguation, not arbitrary selection. Embeddings/rerankers are specialized, not chat generators.",
        "parameters": TaskSizeInputs.model_json_schema(),
    },
}


def infer_task_size(task: str, settings: dict[str, Any], *, model: str | None = None,
                    client: Any = None) -> dict[str, Any]:
    """Have Nemotron call the catalog; only catalog ranges can reach discovery."""
    from openai import OpenAI
    model = model or settings["nebius"]["default_model"]
    allowed = settings["nebius"].get("allowed_models") or []
    if allowed and model not in allowed:
        raise ValueError("Nemotron model override is not allowed by config")
    TaskSizeInputs(task=task)  # Bound input before any network request.
    if client is None:
        key = os.getenv("NEBIUS_API_KEY")
        if not key:
            raise ValueError("NEBIUS_API_KEY is required for Nemotron task sizing")
        client = OpenAI(base_url=settings["nebius"]["base_url"], api_key=key,
                        timeout=float(settings["nebius"]["timeout_seconds"]), max_retries=0)
    messages = [
        {"role": "system", "content": "Call lookup_task_size with the user's complete task unchanged. Do not invent a use_case_id, model name, or parameter size. The catalog is heuristic guidance only."},
        {"role": "user", "content": task},
    ]
    completion = client.chat.completions.create(
        model=model, messages=messages, tools=[TASK_SIZE_TOOL],
        tool_choice="auto",
        max_tokens=int(settings["nebius"]["max_output_tokens"]), temperature=0,
    )
    choice = completion.choices[0]
    calls = choice.message.tool_calls or []
    if choice.finish_reason == "length" or len(calls) != 1:
        raise ValueError("Nemotron did not return one complete task-size tool call")
    call = calls[0]
    args = TaskSizeInputs.model_validate_json(call.function.arguments)
    if (call.function.name != "lookup_task_size" or args.task is None
            or normalize(args.task) != normalize(task) or args.use_case_id):
        raise ValueError("Nemotron task-size call changed the task or selected an unconfirmed use case")
    result = lookup_task_size(task=task)
    # Range values and ambiguity decisions come from the catalog, never generated text.
    return {**result, "nemotron": {"model": model, "tool_called": call.function.name,
                                   "finish_reason": choice.finish_reason}}
