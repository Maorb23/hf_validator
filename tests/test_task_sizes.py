import json
from pathlib import Path

import pytest

from hf_model_watch.recommender.task_sizes import lookup_task_size
from hf_model_watch.recommender import task_sizes
from hf_model_watch.recommender.core import TOOLS, _validate_tool_args
from hf_model_watch.nanoclaw_mcp import NanoClawMcpServer
from hf_model_watch.openclaw_adapter import OpenClawToolAdapter


def test_catalog_all_ranges_and_no_model_names():
    catalog = json.loads(Path(task_sizes.__file__).with_name('task_size_catalog.json').read_text())
    assert len(catalog['use_cases']) == 20
    assert 'model_id' not in json.dumps(catalog)
    for entry in catalog['use_cases']:
        result = lookup_task_size(use_case_id=entry['id'])
        assert result['status'] == 'matched'
        assert result['selected']['recommended_size'] == entry['recommended_size']
        for alias in entry['match']['aliases']:
            found = lookup_task_size(task=alias)
            assert entry['id'] in [r['id'] for r in found['candidates']]


def test_normalized_complete_alias_and_specialized_types():
    result = lookup_task_size(task='Please DRAFT_an—EMAIL!')
    assert result['selected']['id'] == 'email_rewriting'
    assert result['selected']['recommended_size']['min_b'] == 2
    assert lookup_task_size(task='create text embeddings')['selected']['model_type'] == 'embedding_model'
    assert lookup_task_size(task='rerank retrieved documents')['selected']['model_type'] == 'reranker_model'


def test_ambiguity_overlap_and_no_match_do_not_select():
    for task in ['draft an email and translate text into another language',
                 'support query search', 'zxqv', 'Qwen3.5-4B']:
        result = lookup_task_size(task=task)
        assert result['status'] == 'needs_clarification'
        assert result['selected'] is None
    assert lookup_task_size(use_case_id='text_to_json', task='draft an email')['selected']['id'] == 'text_to_json'


@pytest.mark.parametrize('raw', [{}, {'task': ''}, {'use_case_id': 'unknown'}, {'task': 'hello', 'model_id': 'org/model'}])
def test_invalid_inputs(raw):
    with pytest.raises(ValueError):
        lookup_task_size(**raw)


def test_nemotron_tool_arguments_and_mcp_call_without_research(tmp_path):
    assert any(t['function']['name'] == 'lookup_task_size' for t in TOOLS)
    args = _validate_tool_args('lookup_task_size', {'use_case_id': 'support_triage'}, 8)
    assert lookup_task_size(**args)['selected']['recommended_size']['max_b'] == 2
    def forbidden(*args, **kwargs):
        raise AssertionError('Catalog lookup must not start recommendation/research')
    server = NanoClawMcpServer(OpenClawToolAdapter(state_root=tmp_path, cli_runner=forbidden))
    reply = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
                           'params': {'name': 'lookup_task_size', 'arguments': {'task': 'draft an email'}}})
    result = json.loads(reply['result']['content'][0]['text'])
    assert result['selected']['recommended_size']['max_b'] == 4
    assert not list(tmp_path.iterdir())


def test_nemotron_research_loop_dispatches_size_tool(tmp_path):
    from hf_model_watch.recommender.core import recommend, load_settings
    from tests.test_recommender import _spec
    responses = iter([
        {'tool_calls': [{'id': 'size', 'function': {
            'name': 'lookup_task_size', 'arguments': '{"task":"draft an email"}'}}]},
        {'content': '{}'},
    ])
    result = recommend(_spec(request='draft an email'), load_settings(), tmp_path,
                       chat=lambda: next(responses), search=lambda *a, **k: [])
    assert any(e['event'] == 'tool_call' and e.get('name') == 'lookup_task_size'
               for e in result['events'])
    assert not any(e['event'] == 'tool_error' for e in result['events'])


def test_live_client_contract_and_untrusted_arguments():
    from types import SimpleNamespace as NS
    from hf_model_watch.recommender.task_sizes import infer_task_size
    from hf_model_watch.recommender.core import load_settings
    def client(args):
        def create(**kwargs):
            assert kwargs['tool_choice'] == 'auto'
            return NS(choices=[NS(finish_reason='tool_calls', message=NS(tool_calls=[
                NS(function=NS(name='lookup_task_size', arguments=json.dumps(args)))
            ]))])
        return NS(chat=NS(completions=NS(create=create)))
    settings = load_settings()
    r = infer_task_size('draft an email', settings, client=client({'task': 'draft an email'}))
    assert r['selected']['recommended_size']['min_b'] == 2
    assert r['nemotron']['tool_called'] == 'lookup_task_size'
    r = infer_task_size('Draft professional business emails.', settings,
                        client=client({'task': 'Draft professional business emails'}))
    assert r['selected']['id'] == 'email_rewriting'
    with pytest.raises(ValueError):
        infer_task_size('draft an email', settings, client=client({'task': 'write a Python function'}))
    with pytest.raises(ValueError):
        infer_task_size('draft an email', settings, client=client({'use_case_id': 'coding_assistance'}))


def test_cli_size_tool_without_scanning(monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    monkeypatch.setattr(task_sizes, 'infer_task_size', lambda task, *a, **k: lookup_task_size(task=task))
    result = CliRunner().invoke(app, ['task-size', 'draft an email'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['selected']['recommended_size']['max_b'] == 4


def test_cli_recommend_passes_band_and_preserves_explicit_size(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    from hf_model_watch.recommender import discovery
    calls = []
    monkeypatch.setattr(task_sizes, 'infer_task_size', lambda task, *a, **k: lookup_task_size(task='draft an email'))
    def discover(*a, **kw):
        calls.append(kw.get('size_range'))
        return {'status': 'insufficient_evidence', 'recommendations': []}
    monkeypatch.setattr(discovery, 'discover_models', discover)
    flags = ['--task-size-guidance', '--gpu-memory-gb', '24', '--gpu-hourly-cost-usd', '1',
             '--budget-usd-per-hour', '2', '--license', 'any', '--framework-mode', 'sweep',
             '--speculative-decoding', 'off', '--json', '--state-root', str(tmp_path)]
    for text in ['draft an email', 'draft an email with a 7B model']:
        result = CliRunner().invoke(app, ['recommend', text, *flags])
        assert result.exit_code == 0, result.output
    assert calls == [(2., 4.), None]


def test_discovery_enforces_catalog_range_before_research(tmp_path, monkeypatch):
    from hf_model_watch.recommender import discovery
    from hf_model_watch.recommender.core import load_settings
    from tests.test_recommender import _spec, _record
    rows = []
    for size in (1., 3., 7.):
        row = _record(f'Qwen/model-{size:g}B', size)
        row.update(has_tokenizer=True, has_safetensors=True, chat_template_present=True,
                   estimated_download_size_gb=size * 2, downloads=100000)
        rows.append(row)
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *a: None)
    r = discovery.discover_models(_spec(request='draft an email', speculative_decoding='off'),
        load_settings(), tmp_path, scan=lambda *a: rows, lookup=lambda **kw: [],
        search=lambda *a, **kw: [], size_range=(2., 4.))
    assert [row['model_id'] for row in r['recommendations']] == ['Qwen/model-3B']
    assert r['size_range']['min_b'] == 2
