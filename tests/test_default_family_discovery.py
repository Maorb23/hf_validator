import pytest

from hf_model_watch.recommender import discovery
from hf_model_watch.recommender.core import load_settings
from tests.test_recommender import _spec, _record


@pytest.mark.parametrize(("request_text", "families"), [
    ("Find a 2B conversational model", ["qwen", "gemma"]),
    ("Find models like Qwen3-2B", ["qwen", "gemma"]),
    ("Find models like Gemma-2B", ["qwen", "gemma"]),
    ("Find models like MiniCPM5-2B", ["minicpm", "qwen", "gemma"]),
])
def test_default_and_requested_families_are_scanned_once(tmp_path, request_text, families):
    called = []
    paths = []
    def lookup(*, family, output_path, **kwargs):
        called.append(family.lower())
        paths.append(output_path)
        return []
    result = discovery.discover_models(_spec(request=request_text, speculative_decoding="off"),
        load_settings(), tmp_path, scan=lambda *args: [], lookup=lookup,
        search=lambda *args, **kwargs: [])
    assert called == families
    assert len(paths) == len(set(paths))
    assert result["scan_diagnostics"]["gemma"]["found"] == 0
    if "qwen" in result["scan_diagnostics"]:
        assert result["scan_diagnostics"]["qwen"]["found"] == 0


def test_size_only_request_can_recommend_established_qwen_and_merge_recent(tmp_path, monkeypatch):
    record = _record("Qwen/model-2B", 2)
    record.update(has_tokenizer=True, has_safetensors=True, chat_template_present=True,
                  estimated_download_size_gb=4, downloads=100000)
    monkeypatch.setattr(discovery, "_validate_with_nemotron", lambda *args: None)
    def lookup(*, family, **kwargs):
        return [record] if family == "Qwen" else []
    result = discovery.discover_models(_spec(request="Find a 2B conversational model", speculative_decoding="off"),
        load_settings(), tmp_path, scan=lambda *args: [record], lookup=lookup,
        search=lambda *args, **kwargs: [])
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["model_id"] == record["model_id"]
    assert set(result["recommendations"][0]["discovery_strategies"]) == {"recent_similar", "qwen"}
    assert result["scan_diagnostics"]["qwen"]["eligible"] == 1
