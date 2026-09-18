from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hf_model_watch.hub import lookup


def _info(model_id: str, *, downloads: int = 0, tags: list[str] | None = None):
    return SimpleNamespace(
        id=model_id,
        author=model_id.split("/", 1)[0],
        tags=tags or ["text-generation"],
        siblings=[],
        config={},
        pipeline_tag="text-generation",
        downloads=downloads,
        likes=0,
        gated=False,
        private=False,
        sha="revision",
        card_data={},
    )


class _Api:
    def __init__(self) -> None:
        self.model_info_calls: list[str] = []

    def model_info(self, model_id: str, **_kwargs):
        self.model_info_calls.append(model_id)
        return _info(model_id)

    def list_models(
        self, *, search=None, sort=None, limit=None, full=None, cardData=None, fetch_config=None
    ):
        return [
            _info("Qwen/Qwen3-4B", downloads=100),
            _info("other/not-a-match", downloads=1),
        ]


@pytest.fixture(autouse=True)
def no_network_enrichment(monkeypatch):
    monkeypatch.setattr(lookup, "_enrich_candidate_metadata", lambda *_args, **_kwargs: None)


def test_exact_model_lookup_writes_a_reproducible_record(tmp_path) -> None:
    api = _Api()
    output = tmp_path / "lookup.jsonl"

    records = lookup.lookup_models(model_id="org/model", output_path=output, api=api)

    assert api.model_info_calls == ["org/model"]
    assert records[0]["model_id"] == "org/model"
    assert records[0]["lookup_match"]["mode"] == "exact_model_id"
    assert json.loads(output.read_text())["revision"] == "revision"


def test_family_lookup_matches_locally_and_excludes_non_matches(tmp_path) -> None:
    records = lookup.lookup_models(
        family="qwen3", output_path=tmp_path / "lookup.jsonl", api=_Api()
    )

    assert [record["model_id"] for record in records] == ["Qwen/Qwen3-4B"]
    assert records[0]["lookup_match"]["reasons"] == ["substring match in model_id"]


def test_fuzzy_lookup_is_opt_in(tmp_path) -> None:
    assert (
        lookup.lookup_models(query="qween3", output_path=tmp_path / "plain.jsonl", api=_Api()) == []
    )

    records = lookup.lookup_models(
        query="qween3", fuzzy=True, output_path=tmp_path / "fuzzy.jsonl", api=_Api()
    )

    assert [record["model_id"] for record in records] == ["Qwen/Qwen3-4B"]
    assert records[0]["lookup_match"]["mode"] == "fuzzy"


def test_lookup_requires_exactly_one_selector(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        lookup.lookup_models(
            model_id="org/model", family="qwen", output_path=tmp_path / "lookup.jsonl", api=_Api()
        )
