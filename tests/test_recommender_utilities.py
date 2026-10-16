"""Check shared interpretation, feature enforcement, and adapter clarification paths."""
import json

import pytest
from typer.testing import CliRunner

from hf_model_watch.cli import app
from hf_model_watch.openclaw_adapter import OpenClawToolAdapter, RecommendationInputs
from hf_model_watch.recommender import core, discovery
from hf_model_watch.recommender.utilities import (
    detect_features, discovery_inputs, extract_constraints, normalize_prompt,
    speculative_bonus, speculative_preference,
)
from hf_model_watch.recommender.run_state import save_run, run_dir
from tests.test_recommender import _record, _spec


@pytest.mark.parametrize('phrase', ['speculative decuding', 'spec decoding', 'DSpark', 'DFlash', 'EAGLE-3', 'MTP', 'draft model'])
def test_both_flows_research_aliases(phrase):
    request = f'Find models similar to org/Qwen3-4B with {phrase}'
    assert 'speculative decoding' in detect_features(request)
    constraints = extract_constraints(request)
    assert constraints['original_request'] == request
    assert constraints['target_params_billion'] == 4
    assert constraints['speculative_decoding'] == 'prefer'
    assert 'draft_pair' in dict(core.model_research_queries(_record('org/Qwen3-4B', 4), request))


@pytest.mark.parametrize('user_request, explicit, expected', [
    ('must have speculative decuding', None, 'require'),
    ('prefer speculative decoding', None, 'prefer'),
    ('without speculative decoding', None, 'off'),
    ('fast inference', None, None),
    ('must have speculative decoding', 'off', 'off'),
    ('fast inference', 'y', 'prefer'),
    ('fast inference', 'n', 'off'),
])
def test_preference_wording_and_override(user_request, explicit, expected):
    assert speculative_preference(user_request, explicit) == expected


def test_common_inputs_keep_original_and_request_feature_clarification():
    supplied = {'gpu_memory_gb': 80, 'budget_usd_per_hour': 5,
                'gpu_hourly_cost_usd': 3, 'license': 'any', 'framework_mode': 'fast'}
    values, missing = discovery_inputs('Score posts', supplied)
    assert missing == ['speculative_decoding']
    values, missing = discovery_inputs('must have speculative decuding', supplied)
    assert not missing
    assert values['request'] == 'must have speculative decuding'
    assert values['speculative_decoding'] == 'require'


def test_shared_verified_bonus_and_required_filter():
    record = _record('org/target-4B', 4)
    draft = {'source_url': 'https://huggingface.co/org/target-4B-DSpark',
             'retrieved_text': 'DSpark draft model for `org/target-4B`. Run with SGLang.'}
    settings = core.load_settings()
    settings['selection']['speculative_decoding']['verified_bonus'] = 0.4
    unknown = core.rank_candidates([record], _spec(speculative_decoding='require'), {}, settings)[0]
    assert not unknown['eligible']
    assert 'speculative_decoding_unverified' in unknown['blocking_reasons']
    verified = core.rank_candidates([record], _spec(speculative_decoding='require'),
                                    {record['model_id']: [draft]}, settings)[0]
    assert verified['eligible']
    assert verified['runtime'] == 'sglang'
    assert verified['speculative_decoding']['status'] == 'verified'
    assert verified['feature_bonus'] == speculative_bonus('verified', 'require', settings) == 0.4
    off = core.rank_candidates([record], _spec(speculative_decoding='off'),
                               {record['model_id']: [draft]}, settings)[0]
    assert off['feature_bonus'] == 0
    assert off['estimated_bf16_memory_gb'] == 9


def test_json_clarification_resumes_without_terminal_input(tmp_path, monkeypatch):
    captured = []
    def discover(spec, *_args, **_kwargs):
        captured.append(spec)
        return {'status': 'insufficient_evidence', 'recommendations': []}
    monkeypatch.setattr(discovery, 'discover_models', discover)
    runner = CliRunner()
    args = ['recommend', 'Score posts', '--gpu-memory-gb', '80', '--budget-usd-per-hour', '5',
            '--gpu-hourly-cost-usd', '3', '--license', 'any', '--framework-mode', 'fast',
            '--json', '--state-root', str(tmp_path)]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    response = json.loads(first.stdout)
    assert response['needs_input'] == ['speculative_decoding']
    assert response['questions'][0]['default'] == 'prefer'
    second = runner.invoke(app, ['recommend', '--run-id', response['run_id'],
                               '--speculative-decoding', 'y', '--json', '--state-root', str(tmp_path)])
    assert second.exit_code == 0, second.output
    assert captured[0].request == 'Score posts'
    assert captured[0].speculative_preference == 'prefer'


def test_adapter_forwards_preference_on_initial_call_and_resume(tmp_path):
    calls = []
    adapter = OpenClawToolAdapter(state_root=tmp_path,
        cli_runner=lambda args, _timeout: calls.append(args) or {'status': 'needs_input'})
    adapter.recommend_models({'request': 'Score posts', 'speculative_decoding': 'require'})
    assert calls[0][calls[0].index('--speculative-decoding') + 1] == 'require'
    assert '--json' in calls[0]
    run_id = 'run_speculative'
    save_run(run_dir(tmp_path, run_id), {'run_id': run_id, 'status': 'needs_input',
                                      'needs_input': ['speculative_decoding']})
    adapter.provide_recommendation_input({'run_id': run_id, 'answers': {'speculative_decoding': 'n'}})
    assert calls[1][calls[1].index('--speculative-decoding') + 1] == 'n'


def test_discovery_required_feature_excludes_unverified_candidates(tmp_path, monkeypatch):
    record = _record('org/target-4B', 4)
    monkeypatch.setattr(discovery, 'score_candidate', lambda row, *_args:
                        {'model_id': row['model_id'], 'score': .8, 'eligible': True})
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *_args: None)
    result = discovery.discover_models(_spec(speculative_decoding='require'), core.load_settings(),
        tmp_path, scan=lambda *_args: [record], lookup=lambda **_kwargs: [],
        search=lambda *_args, **_kwargs: [])
    assert result['status'] == 'insufficient_evidence'
    assert not result['recommendations']


def test_preference_affects_shortlist_and_shared_configurable_bonus(tmp_path, monkeypatch):
    rows = [_record(f'org/model-{i}-4B', 4) for i in range(4)]
    rows[-1]['card_excerpt'] = 'DSpark speculative decoding draft model available.'
    scores = dict(zip([row['model_id'] for row in rows], [.8, .75, .7, .6]))
    monkeypatch.setattr(discovery, 'score_candidate', lambda row, *_args:
        {'model_id': row['model_id'], 'score': scores[row['model_id']], 'eligible': True})
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *_args: None)
    settings = core.load_settings()
    settings['selection']['speculative_decoding'].update(inferred_bonus=.3, research_candidates=3)
    queries = []
    def search(query, *_args, **_kwargs):
        queries.append(query)
        return []
    result = discovery.discover_models(_spec(speculative_decoding='prefer'), settings, tmp_path,
        scan=lambda *_args: rows, lookup=lambda **_kwargs: [], search=search)
    assert result['recommendations'][0]['model_id'] == rows[0]['model_id']
    assert not any(rows[-1]['model_id'] in query for query in queries)
    assert any(rows[2]['model_id'] in query for query in queries)
    assert result['recommendations'][0]['final_score'] == .9


def test_interactive_confirmation_defaults_to_prefer(tmp_path, monkeypatch):
    captured = []
    def discover(spec, *_args, **_kwargs):
        captured.append(spec)
        return {'status': 'insufficient_evidence', 'recommendations': []}
    monkeypatch.setattr(discovery, 'discover_models', discover)
    result = CliRunner().invoke(app, ['recommend', 'Score posts', '--gpu-memory-gb', '80',
        '--budget-usd-per-hour', '5', '--gpu-hourly-cost-usd', '3', '--license', 'any',
        '--framework-mode', 'fast', '--state-root', str(tmp_path)], input='\n')
    assert result.exit_code == 0, result.output
    assert 'Prefer speculative decoding when supported? [Y/n]' in result.output
    assert captured[0].speculative_preference == 'prefer'


@pytest.mark.parametrize('phrase, canonical', [
    ('specolative decoding', 'speculative decoding'),
    ('speculative decodign', 'speculative decoding'),
    ('tool caling', 'tool calling'),
])
def test_fuzzy_feature_phrases(phrase, canonical):
    user_request = f'Recommend org/model-2B; must have {phrase}'
    assert canonical in detect_features(user_request)
    assert extract_constraints(user_request)['original_request'] == user_request
    assert canonical in normalize_prompt(user_request)
    if canonical == 'speculative decoding':
        assert speculative_preference(user_request) == 'require'
        assert 'draft_pair' in dict(core.model_research_queries(_record('org/model-2B', 2), user_request))
        assert speculative_preference(f'without {phrase}') == 'off'


@pytest.mark.parametrize('text', [
    'Use org/specolative-decoding-2B',
    'See https://example.com/specolative-decoding',
    'Prefer speculative encoding',
    'Read posts with tool cooling',
    'Reading and vision models',
])
def test_fuzzy_matching_preserves_identifiers_and_unrelated_phrases(text):
    assert normalize_prompt(text) == text


MINICPM_LAUNCH = '''vllm serve openbmb/MiniCPM5-2B --port 8000
python -m sglang.launch_server \\
  --model-path openbmb/MiniCPM5-2B \\
  --trust-remote-code \\
  --speculative-algorithm DSPARK \\
  --speculative-draft-model-path openbmb/MiniCPM5-2B-DSpark \\
  --speculative-dspark-block-size 7 \\
  --port 30000
'''


def test_target_card_verifies_exact_draft_and_runtime():
    record = _record('openbmb/MiniCPM5-2B', 2)
    source = {'source_url': 'https://huggingface.co/openbmb/MiniCPM5-2B',
              'retrieved_text': MINICPM_LAUNCH}
    runtime = core.runtime_assessment(record, [source])
    assert runtime['vllm']['status'] == 'compatible'
    assert runtime['sglang']['status'] == 'compatible'
    assert runtime['speculative_decoding']['draft_model'] == 'openbmb/MiniCPM5-2B-DSpark'
    assert runtime['speculative_decoding']['runtimes'] == ['sglang']
    assert runtime['speculative_decoding']['status'] == 'verified'
    ranked = core.rank_candidates([record], _spec(speculative_decoding='require'),
                                  {record['model_id']: [source]}, core.load_settings())[0]
    assert ranked['eligible']
    assert ranked['runtime'] == 'sglang'


@pytest.mark.parametrize('url, body', [
    ('https://example.com/blog', MINICPM_LAUNCH),
    ('https://huggingface.co/other/MiniCPM5-2B', MINICPM_LAUNCH),
    ('https://huggingface.co/openbmb/MiniCPM5-2B', MINICPM_LAUNCH.replace('--model-path openbmb/MiniCPM5-2B ', '--model-path other/target ')),
    ('https://huggingface.co/openbmb/MiniCPM5-2B', MINICPM_LAUNCH.replace('--speculative-algorithm DSPARK', '')),
])
def test_launch_verification_rejects_wrong_provenance_target_and_incomplete_pair(url, body):
    record = _record('openbmb/MiniCPM5-2B', 2)
    runtime = core.runtime_assessment(record, [{'source_url': url, 'retrieved_text': body}])
    assert not runtime['speculative_decoding']['usable']


def test_card_excerpt_preserves_launch_commands_after_long_introduction():
    from hf_model_watch.recommender.utilities import evidence_excerpt
    text = 'Introduction. ' * 1000 + '\n```bash\n' + MINICPM_LAUNCH + '```\n'
    excerpt = evidence_excerpt(text, 3000)
    assert len(excerpt) <= 3000
    record = _record('openbmb/MiniCPM5-2B', 2)
    record['card_excerpt'] = excerpt
    runtime = core.runtime_assessment(record, [])
    assert runtime['speculative_decoding']['usable']
    assert runtime['vllm']['status'] == 'compatible'


def test_discovery_require_accepts_target_card_pair(tmp_path, monkeypatch):
    record = _record('openbmb/MiniCPM5-2B', 2)
    monkeypatch.setattr(discovery, 'score_candidate', lambda row, *_args:
        {'model_id': row['model_id'], 'score': .8, 'eligible': True})
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *_args: None)
    source = {'source_url': 'https://huggingface.co/openbmb/MiniCPM5-2B', 'retrieved_text': MINICPM_LAUNCH}
    result = discovery.discover_models(_spec(speculative_decoding='require'), core.load_settings(),
        tmp_path, scan=lambda *_args: [record], lookup=lambda **_kwargs: [],
        search=lambda *_args, **_kwargs: [source])
    assert result['recommendations'][0]['task_fit']['speculative_decoding'] == 'verified'
    assert result['recommendations'][0]['runtime_assessment']['runtime'] == 'sglang'


@pytest.mark.parametrize('style', ['inline', 'bare'])
def test_long_extractor_text_preserves_unfenced_commands(style):
    from hf_model_watch.recommender.utilities import evidence_excerpt
    commands = MINICPM_LAUNCH
    if style == 'inline':
        commands = '`' + commands.replace('\npython', '`\n\n`python').rstrip() + '`'
    excerpt = evidence_excerpt('Introduction. ' * 1000 + commands, 3000)
    record = _record('openbmb/MiniCPM5-2B', 2)
    source = {'source_url': 'https://huggingface.co/openbmb/MiniCPM5-2B',
              'retrieved_text': excerpt}
    runtime = core.runtime_assessment(record, [source])
    assert len(excerpt) <= 3000
    assert runtime['vllm']['status'] == 'compatible'
    assert runtime['sglang']['status'] == 'compatible'
    assert runtime['speculative_decoding']['usable']


@pytest.mark.parametrize('preference', ['prefer', 'require'])
def test_both_flows_retain_runtime_only_extraction_without_speculative_pair(tmp_path, monkeypatch, preference):
    record = _record('openbmb/MiniCPM5-1B', 1)
    source = {'source_url': 'https://huggingface.co/openbmb/MiniCPM5-1B',
              'retrieved_text': 'vllm serve openbmb/MiniCPM5-1B --port 8000\n'
                                'python -m sglang.launch_server --model-path openbmb/MiniCPM5-1B --port 30000'}
    # Search returns no runtime evidence; only direct target-card extraction supplies it.
    def search(_query, _settings, *, extract_urls=None):
        return [source] if extract_urls and source['source_url'] in extract_urls else []
    monkeypatch.setattr(discovery, 'score_candidate', lambda row, *_args:
        {'model_id': row['model_id'], 'score': .7, 'eligible': True})
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *_args: None)
    settings = core.load_settings()
    spec = _spec(speculative_decoding=preference)
    discovered = discovery.discover_models(spec, settings, tmp_path / 'discovery',
        scan=lambda *_args: [record], lookup=lambda **_kwargs: [], search=search)
    responses = iter([
        {'tool_calls': [{'id': 'extract', 'type': 'function', 'function': {
            'name': 'tavily_extract', 'arguments': json.dumps({'urls': [source['source_url']]})}}]},
        {'content': json.dumps({'viable_model_ids': [record['model_id']]})},
    ])
    explicit = core.recommend(spec, settings, tmp_path / 'explicit', initial_records=[record],
        explicit_model_id=record['model_id'], search=search, chat=lambda: next(responses))
    checked = explicit['evaluated_candidates'][0]
    assert {'vllm', 'sglang'} <= set(checked['compatible_frameworks'])
    assert checked['speculative_decoding']['status'] == 'unknown'
    assert not checked['speculative_decoding']['usable']
    assert source in checked['evidence']
    if preference == 'prefer':
        assert explicit['recommendation']
        row = discovered['recommendations'][0]
        assert source in row['evidence']
        assert row['task_fit']['speculative_decoding'] == 'unknown'
        assert row['runtime_assessment']['compatible_frameworks'] == checked['compatible_frameworks']
        assert row['runtime_assessment']['runtime'] == checked['runtime'] == 'vllm'
    else:
        assert explicit['recommendation'] is None
        assert not discovered['recommendations']


@pytest.mark.parametrize('metadata, wording, expected', [
    ('llama3.2', 'the Llama 3.2 Community License, Copyright Meta.', 'consistent'),
    ('llama3.1', 'the Meta Llama 3.1 Community License.', 'consistent'),
    ('apache-2.0', 'the Apache License, Version 2.0.', 'consistent'),
    ('apache-2.0', 'the Apache License 2.0.', 'consistent'),
    ('apache-2.0', 'MIT.', 'conflicting'),
    ('llama3.2', 'the Llama 3.1 Community License.', 'conflicting'),
    ('llama3.2', 'the project terms.', 'unknown'),
    ('llama3', 'the Llama 3.9 Community License.', 'unknown'),
])
def test_shared_license_claims(metadata, wording, expected):
    from hf_model_watch.recommender.utilities import license_assessment
    source = {'source_url': 'https://example.com/license', 'retrieved_text': f'Licensed under {wording}'}
    checked = license_assessment(metadata, [source])
    assert checked['status'] == expected
    assert checked['conflict'] == (expected == 'conflicting')
    if checked['claims']:
        assert checked['claims'][0]['source_url'] == source['source_url']


def test_both_flows_use_shared_license_normalization(tmp_path, monkeypatch):
    from hf_model_watch.recommender.utilities import normalize_license
    assert normalize_license('the Llama 3.2 Community License') == 'llama3.2'
    record = _record('org/llama-2B', 2)
    record['license'] = 'llama3.2'
    source = {'source_url': 'https://huggingface.co/org/llama-2B',
              'retrieved_text': 'Licensed under the Llama 3.2 Community License, Copyright Meta.'}
    spec = _spec(license='any', speculative_decoding='off')
    settings = core.load_settings()
    checked = core.rank_candidates([record], spec, {record['model_id']: [source]}, settings)[0]
    assert checked['eligible']
    assert not checked['license_claims_conflict']
    assert checked['license_assessment']['status'] == 'consistent'
    monkeypatch.setattr(discovery, 'score_candidate', lambda row, *_args:
        {'model_id': row['model_id'], 'score': .66, 'eligible': True})
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *_args: None)
    result = discovery.discover_models(spec, settings, tmp_path, scan=lambda *_args: [record],
        lookup=lambda **_kwargs: [], search=lambda *_args, **_kwargs: [source])
    assert result['recommendations'][0]['runtime_assessment']['license_assessment'] == checked['license_assessment']


@pytest.mark.parametrize('updates, reason', [
    ({'model_id': 'Devseis/endpoint-auditor-1.5b'}, 'model_keyword_rejected'),
    ({'tags': ['AUDITOR']}, 'model_keyword_rejected'),
    ({'model_id': 'org/model-MLX-4bit'}, 'unsupported_finetune_format'),
    ({'tags': ['gguf']}, 'unsupported_finetune_format'),
    ({'config': {'quantization_config': {'format': 'compressed_tensors'}}}, 'unsupported_finetune_format'),
])
def test_shared_rejections_apply_to_hub_and_both_recommender_flows(tmp_path, monkeypatch, updates, reason):
    from hf_model_watch.recommender.utilities import model_rejection_reasons
    from hf_model_watch.hub.candidate_filter import filter_candidate
    record = _record('org/model-2B', 2)
    record.update(updates)
    assert reason in model_rejection_reasons(record)
    assert reason in filter_candidate(record, {}).blocking_reasons
    spec = _spec(speculative_decoding='off')
    row = core.rank_candidates([record], spec, {}, core.load_settings())[0]
    assert not row['eligible']
    assert reason in row['blocking_reasons']
    monkeypatch.setattr(discovery, 'score_candidate', lambda candidate, *_args:
        {'model_id': candidate['model_id'], 'score': 1, 'eligible': True})
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *_args: None)
    def no_research(*_args, **_kwargs):
        pytest.fail('Rejected models must not reach discovery research')
    result = discovery.discover_models(spec, core.load_settings(), tmp_path,
        scan=lambda *_args: [record], lookup=lambda **_kwargs: [], search=no_research)
    assert not result['recommendations']


def test_shared_reject_policy_does_not_reject_ordinary_models():
    from hf_model_watch.recommender.utilities import model_rejection_reasons
    assert not model_rejection_reasons(_record('openbmb/MiniCPM5-2B', 2))
    assert not model_rejection_reasons({'model_id': 'org/ordinary', 'tags': [],
        'config': {'quantization_config': {'quant_method': 'awq'}}})
