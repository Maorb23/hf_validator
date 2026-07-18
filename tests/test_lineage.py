from __future__ import annotations

from hf_model_watch.hub.lineage import extract_lineage


def test_lineage_extracts_family_and_size_from_obvious_name() -> None:
    lineage = extract_lineage({"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "tags": []})

    assert lineage["family"] == "Mistral"
    assert lineage["size_billion"] == 7
    assert lineage["size_band"] == "medium"
    assert lineage["version_or_date"] == "v0.3"


def test_lineage_returns_inconclusive_without_prior_tested_models() -> None:
    lineage = extract_lineage({"model_id": "fresh/upload-without-family", "tags": []})

    assert lineage["status"] == "inconclusive"
    assert lineage["predecessor_status"] == "inconclusive"
    assert lineage["related_tested_models"] == []
