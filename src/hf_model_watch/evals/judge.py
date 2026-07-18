from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


def judge_quality(task_type: str, text: str, expected: dict[str, Any]) -> dict[str, Any]:
    normalized = text.strip()
    if task_type == "grade_school_math":
        return judge_numeric_answer(normalized, str(expected.get("final_answer") or ""))
    if task_type == "json_schema_adherence":
        return judge_json_schema(normalized, expected.get("required_keys", []))
    if task_type == "text_to_sql":
        return judge_sql(normalized, expected.get("sql_keywords", []))
    if task_type == "investment_caveat":
        return judge_terms(normalized, expected.get("caveat_terms", []), require_any=True)
    if expected.get("must_contain_any"):
        return judge_terms(normalized, expected["must_contain_any"], require_any=True)
    return judge_terms(normalized, expected.get("must_contain", []), require_any=False)


def judge_json_schema(text: str, required_keys: list[str]) -> dict[str, Any]:
    try:
        data = json.loads(_extract_json(text))
    except json.JSONDecodeError:
        return {"quality_score": 0.0, "success": False, "reason": "json_parse_failed"}
    missing = [key for key in required_keys if key not in data]
    score = 1.0 if not missing else max(0.0, 1.0 - len(missing) / max(len(required_keys), 1))
    return {
        "quality_score": score,
        "success": not missing,
        "reason": "ok" if not missing else "missing_required_keys",
        "missing_keys": missing,
    }


def judge_sql(text: str, keywords: list[str]) -> dict[str, Any]:
    compact = text.strip().strip("`").lower()
    has_select = bool(re.search(r"\bselect\b", compact))
    checks = [keyword.lower() in compact for keyword in keywords]
    score = (sum(checks) + (1 if has_select else 0)) / max(len(keywords) + 1, 1)
    return {
        "quality_score": score,
        "success": score >= 0.75,
        "reason": "ok" if score >= 0.75 else "sql_syntax_or_keyword_failed",
    }


def judge_terms(text: str, terms: list[str], require_any: bool) -> dict[str, Any]:
    if not terms:
        return {"quality_score": 1.0, "success": True, "reason": "no_expectations"}
    lowered = text.lower()
    hits = [term for term in terms if term.lower() in lowered]
    success = bool(hits) if require_any else len(hits) == len(terms)
    denominator = 1 if require_any else len(terms)
    score = min(1.0, len(hits) / denominator)
    return {
        "quality_score": score,
        "success": success,
        "reason": "ok" if success else "expected_terms_missing",
        "matched_terms": hits,
    }


def judge_numeric_answer(text: str, expected: str) -> dict[str, Any]:
    predicted_value = _last_numeric_value(text)
    expected_value = _last_numeric_value(expected)
    success = predicted_value is not None and predicted_value == expected_value
    return {
        "quality_score": 1.0 if success else 0.0,
        "success": success,
        "reason": "ok" if success else "numeric_answer_mismatch",
        "predicted_answer": str(predicted_value) if predicted_value is not None else None,
        "expected_answer": str(expected_value) if expected_value is not None else expected,
    }


def _last_numeric_value(text: str) -> Fraction | Decimal | None:
    answer_region = text.rsplit("####", maxsplit=1)[-1]
    matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?", answer_region)
    if not matches and answer_region != text:
        matches = re.findall(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    if not matches:
        return None
    value = matches[-1].replace("$", "").replace(",", "")
    try:
        return Fraction(value) if "/" in value else Decimal(value)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        return text[start : end + 1]
    return text


def optional_local_model_judge(*args: Any, **kwargs: Any) -> None:
    return None
