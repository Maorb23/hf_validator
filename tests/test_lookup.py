from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hf_model_watch.hub import lookup


def _info(
    model_id: str,
    *,
    downloads: int = 0,
    tags: list[str] | None = None,
    created_at: str | None = None,
    last_modified: str | None = None,
):
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
        created_at=created_at,
        last_modified=last_modified,
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


def test_family_lookup_can_request_newly_created_models(tmp_path) -> None:
    class RecentApi(_Api):
        def __init__(self) -> None:
            super().__init__()
            self.sort = None

        def list_models(self, **kwargs):
            self.sort = kwargs["sort"]
            return super().list_models(**kwargs)

    api = RecentApi()
    lookup.lookup_models(
        family="qwen", sort="createdAt", output_path=tmp_path / "lookup.jsonl", api=api
    )
    assert api.sort == "createdAt"


def test_family_lookup_can_be_scoped_to_original_author(tmp_path) -> None:
    class AuthorApi(_Api):
        def __init__(self) -> None:
            super().__init__()
            self.author = None

        def list_models(self, **kwargs):
            self.author = kwargs.get("author")
            return super().list_models(
                **{key: value for key, value in kwargs.items() if key != "author"}
            )

    api = AuthorApi()
    lookup.lookup_models(
        family="Qwen",
        author="Qwen",
        sort="createdAt",
        output_path=tmp_path / "lookup.jsonl",
        api=api,
    )

    assert api.author == "Qwen"


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


def test_exact_lookup_preserves_huggingface_snake_case_timestamps(tmp_path) -> None:
    class TimestampApi(_Api):
        def model_info(self, model_id: str, **_kwargs):
            return _info(
                model_id,
                created_at="2025-04-28T10:00:00+00:00",
                last_modified="2025-07-01T12:00:00+00:00",
            )

    records = lookup.lookup_models(
        model_id="Qwen/Qwen3-4B",
        output_path=tmp_path / "lookup.jsonl",
        api=TimestampApi(),
    )

    assert records[0]["created_at"] == "2025-04-28T10:00:00+00:00"
    assert records[0]["last_modified"] == "2025-07-01T12:00:00+00:00"


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
