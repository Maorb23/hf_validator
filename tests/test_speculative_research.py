import json
from types import SimpleNamespace

from hf_model_watch.recommender.utilities import documented_pairs, speculative_method_order
from hf_model_watch.recommender.speculative_research import ResearchBudget, research_methods
from hf_model_watch.recommender import discovery
from hf_model_watch.recommender.core import load_settings
from tests.test_recommender import _spec


def source(method, target="Qwen/Qwen3-8B"):
    draft = f"org/Qwen3-8B-{method}"
    return {"source_url": f"https://huggingface.co/{draft}", "score": .9,
            "retrieved_text": f"Draft model for `{target}`. Use SGLang with draft {draft}. vLLM is unrelated."}


def test_default_and_explicit_method_order():
    assert speculative_method_order("Prefer speculative decoding")[:2] == ["dspark", "dflash"]
    assert speculative_method_order("Prefer DFlash")[:2] == ["dflash", "dspark"]
    assert speculative_method_order("Use Medusa")[:3] == ["medusa", "dspark", "dflash"]
    assert speculative_method_order("Similar to org/model-DFlash")[:2] == ["dspark", "dflash"]


def test_separate_searches_find_dspark_without_hint_and_retain_dflash():
    record = {"model_id": "Qwen/Qwen3-8B"}
    settings = load_settings()
    seen = []
    def search(query, _settings, *, extract_urls=None):
        if extract_urls:
            return [source(m) for m in ("DSpark", "DFlash") if any(m.lower() in url.lower() for url in extract_urls)]
        seen.append(query)
        return [source("DSpark")] if " dspark " in query else [source("DFlash")] if " dflash " in query else []
    evidence = {record["model_id"]: []}
    budget = ResearchBudget(search, settings)
    order = speculative_method_order("Prefer speculative decoding")
    coverage = research_methods([{"model_id": record["model_id"], "record": record}], evidence, settings, order, budget)
    assert " dspark " in seen[0]
    assert " dflash " in seen[1]
    pairs = documented_pairs(record, evidence[record["model_id"]], order)
    assert [p["method"] for p in pairs][:2] == ["dspark", "dflash"]
    assert pairs[0]["runtimes"] == ["sglang"]  # Runtime mention alone must not imply support.
    assert coverage[record["model_id"]]["dspark"]["status"] == "documented"
    assert coverage[record["model_id"]]["mtp"]["status"] == "not_researched"
    assert documented_pairs({"model_id": "Qwen/Qwen3.5-8B"}, evidence[record["model_id"]], order) == []
    assert documented_pairs(record, evidence[record["model_id"]], ["dflash", "dspark"])[0]["method"] == "dflash"
    duplicate = {**source("DSpark"), "source_url": source("DSpark")["source_url"] + "?ref=another"}
    merged = documented_pairs(record, [source("DSpark"), duplicate], order)
    assert len(merged) == 1 and len(merged[0]["source_urls"]) == 2


def test_budgets_report_unsearched_instead_of_absence():
    settings = load_settings(); settings["tavily"].update(discovery_search_budget=1, discovery_extract_budget=0)
    record = {"model_id": "Qwen/Qwen3-8B"}; evidence = {record["model_id"]: []}
    budget = ResearchBudget(lambda *args, **kwargs: [source("DSpark", "other/model")], settings)
    result = research_methods([{"model_id": record["model_id"], "record": record}], evidence, settings,
                              ["dspark", "dflash", "eagle"], budget)
    assert budget.searches == 1 and budget.extracts == 0
    assert result[record["model_id"]]["dflash"]["status"] == "not_researched"
    assert result[record["model_id"]]["dflash"]["reason"] == "search_budget_exhausted"
    assert result[record["model_id"]]["dspark"]["status"] == "found_unverified"


def test_compact_validation_omits_repeated_deployment_text(tmp_path, monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "placeholder")
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            model_dump=lambda **kwargs: {"content": '{"assessments":[]}'}))])
    monkeypatch.setattr(discovery, "OpenAI", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    rows = [{"model_id": f"org/model{i}", "eligible": True, "hf_score": .8,
             "runtime_assessment": {"eligible": True, "evidence": [{"retrieved_text": "x" * 100000}],
                                    "hf_evidence": [{"retrieved_text": "y" * 100000}],
                                    "speculative_decoding": {"usable": False, "alternatives": [], "method_coverage": {}}},
             "task_fit": {"speculative_decoding": "unknown"}, "evidence": []} for i in range(8)]
    discovery._validate_with_nemotron(_spec(), rows, load_settings(), tmp_path)
    assert len(captured["messages"][1]["content"]) < 10000
    metadata = json.loads((tmp_path / "nemotron_discovery_input.json").read_text())
    assert metadata["prompt_chars"] < metadata["prompt_limit"]
    assert "hf_evidence" not in metadata["candidates"][0]["deployment"]


def test_markdown_target_and_vllm_speculative_config():
    from hf_model_watch.recommender.utilities import documented_launches
    for flag in ('--speculative-config', '--speculative_config'):
        draft = 'org/Qwen3-8B-DSpark'
        text = (f'Draft model for [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B).\n'
                f'vllm serve Qwen/Qwen3-8B {flag} '\
                f"'{{\"method\":\"dspark\",\"model\":\"{draft}\"}}'")
        pairs = documented_pairs({'model_id': 'Qwen/Qwen3-8B'},
                                 [{'source_url': f'https://huggingface.co/{draft}', 'retrieved_text': text}],
                                 ['dspark', 'dflash'])
        assert pairs[0]['draft_model'] == draft
        assert pairs[0]['runtimes'] == ['vllm']
        assert documented_launches(text)[0]['algorithm'] == 'dspark'
        assert not documented_pairs({'model_id': 'Qwen/Qwen3-4B'},
                                    [{'source_url': f'https://huggingface.co/{draft}', 'retrieved_text': text}], ['dspark'])


def test_verifier_table_and_malformed_config():
    from hf_model_watch.recommender.utilities import documented_launches
    e = {'source_url': 'https://huggingface.co/org/dspark',
         'retrieved_text': '| Verifier | `Qwen/Qwen3-8B` |\nDSpark draft. Use vLLM.'}
    assert documented_pairs({'model_id': 'Qwen/Qwen3-8B'}, [e], ['dspark'])[0]['runtimes'] == ['vllm']
    launch = documented_launches("vllm serve Qwen/Qwen3-8B --speculative-config '{bad}'")[0]
    assert 'draft_model' not in launch


def test_verified_pair_and_method_priority_precede_score():
    from hf_model_watch.recommender.utilities import speculative_rank
    rows = [
        {'model': 'high-score-mention', 'score': .99, 'pair': {'usable': False}},
        {'model': 'eagle', 'score': .98, 'pair': {'usable': True, 'draft_model': 'org/EAGLE3'}},
        {'model': 'dflash', 'score': .97, 'pair': {'usable': True, 'draft_model': 'org/DFlash'}},
        {'model': 'dspark', 'score': .70, 'pair': {'usable': True, 'draft_model': 'org/DSpark'}},
    ]
    def order(preference, methods):
        return [r['model'] for r in sorted(rows, key=lambda r: (*speculative_rank(r['pair'], preference, methods), -r['score']))]
    defaults = speculative_method_order('Prefer speculative decoding')
    assert order('prefer', defaults) == ['dspark', 'dflash', 'eagle', 'high-score-mention']
    assert order('require', defaults) == ['dspark', 'dflash', 'eagle', 'high-score-mention']
    assert order('prefer', speculative_method_order('Prefer EAGLE'))[0] == 'eagle'
    assert order('off', defaults)[0] == 'high-score-mention'
