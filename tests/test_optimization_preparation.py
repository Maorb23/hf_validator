import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from hf_model_watch.optimization.preparation import prepare, serving_arguments
from hf_model_watch.conversation import ConversationAdapter
from hf_model_watch.openclaw_adapter import OpenClawToolAdapter
from hf_model_watch.recommender.run_state import load_run, save_run

MODEL = 'org/target'
REV = 'a' * 40
URL = f'https://huggingface.co/{MODEL}/blob/{REV}/README.md'


def report(command):
    return {'model': MODEL, 'revision': REV, 'sources': [{'source_url': URL, 'retrieved_text': command}],
            'inspection': {'claims': [], 'heads': [{'target_model': MODEL, 'draft_model': 'org/draft',
                'target_revision': REV, 'serving_image': 'pinned:image',
                'runtime': 'vllm' if command.startswith('vllm') else 'sglang', 'status': 'verified', 'source_urls': [URL]}]}}


def inputs():
    return {'gpu_type': 'gpu-h100-sxm', 'input_tokens': 128, 'output_tokens': 32,
            'concurrency': 4, 'gpu_count': 1, 'gpu_hourly_cost_usd': 2, 'budget_usd_per_hour': 4}


def execution():
    return {'images': {'vllm': 'pinned:image'}, 'bucket': 'bucket', 'prefix': 'study',
            'preset': '1gpu-16vcpu-96gb', 'timeout_minutes': 120, 'dry_run': True, 'optuna': {'trials': 3, 'sampler': 'TPESampler'}}


def selected():
    return {'run_id': 'run_prepare', 'selected_model_id': MODEL, 'selection': {'revision': REV},
            'request_spec': {'gpu_count': 1}, 'recommendations': [{'model_id': MODEL}]}


def test_cli_preparation_does_not_reuse_or_update_conversation(tmp_path, monkeypatch):
    from hf_model_watch.cli import _prepare_selected_optimization

    monkeypatch.delenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', raising=False)
    monkeypatch.setattr('hf_model_watch.nanoclaw_mcp.load_optimizer_config', lambda _: None)
    run = selected()
    save_run(tmp_path / run['run_id'], run)
    context = {'workload': {**inputs(), 'frameworks': ['vllm'],
                            'optimizer_bucket': 'other-group-bucket'},
               'active_run_id': 'run_other'}
    path = tmp_path / 'conversation.json'
    path.write_text(json.dumps(context))
    original = path.read_bytes()

    result = _prepare_selected_optimization(tmp_path, run['run_id'], display=False)

    assert result['status'] == 'needs_input'
    assert 'gpu_type' in result['needs_input']
    assert path.read_bytes() == original
    _, saved = load_run(tmp_path, run['run_id'])
    assert 'gpu_type' not in saved['optimization_choices']


def test_cli_recommendation_ignores_conversation_defaults(tmp_path):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app

    path = tmp_path / 'conversation.json'
    # Even malformed group memory must not affect standalone CLI discovery.
    path.write_text('invalid JSON belonging to a different interface')
    result = CliRunner().invoke(app, ['recommend', '--model-id', MODEL,
                                    '--json', '--state-root', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['status'] == 'needs_input'


def test_interactive_cli_collects_run_choices_without_changing_group_memory(tmp_path, monkeypatch):
    from hf_model_watch.cli import _prepare_selected_optimization

    monkeypatch.delenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', raising=False)
    monkeypatch.setattr('hf_model_watch.nanoclaw_mcp.load_optimizer_config', lambda _: None)
    monkeypatch.setattr('hf_model_watch.optimization.preparation.research', lambda *a: report(VLLM))
    run = selected()
    save_run(tmp_path / run['run_id'], run)
    memory = tmp_path / 'conversation.json'
    memory.write_text('{"workload": {"gpu_type": "gpu-other"}}')
    original = memory.read_bytes()
    prompts = []
    # An invalid platform is retried locally. The H100 preset and default image
    # should resolve automatically after hardware and frameworks are supplied.
    answers = iter(['invalid', 'gpu-h100-sxm', 128, 32, 4, 5.0, 'vllm', 'my-bucket'])
    def prompt(label, **kwargs):
        prompts.append(label)
        return next(answers)
    monkeypatch.setattr('hf_model_watch.cli.typer.prompt', prompt)

    result = _prepare_selected_optimization(tmp_path, run['run_id'], interactive=True)

    assert result['status'] == 'prepared'
    assert result['summary']['bucket'] == 'my-bucket'
    assert result['summary']['preset'] == '1gpu-16vcpu-200gb'
    assert len(prompts) == 8
    assert memory.read_bytes() == original
    _, saved = load_run(tmp_path, run['run_id'])
    assert saved['optimization_choices']['input_tokens'] == 128
    assert saved['optimization_execution']['bucket'] == 'my-bucket'
    assert not saved.get('optimization_reference')


def test_interactive_cli_budget_retry_collects_both_cost_answers(tmp_path, monkeypatch):
    from hf_model_watch.cli import _prepare_selected_optimization

    monkeypatch.setenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', json.dumps(execution()))
    monkeypatch.setattr('hf_model_watch.nanoclaw_mcp.load_optimizer_config', lambda _: None)
    monkeypatch.setattr('hf_model_watch.optimization.preparation.research', lambda *a: report(VLLM))
    run = selected()
    run['optimization_choices'] = {**inputs(), 'budget_usd_per_hour': 1}
    save_run(tmp_path / run['run_id'], run)
    answers = iter([2.0, 4.0])
    monkeypatch.setattr('hf_model_watch.cli.typer.prompt', lambda *a, **kw: next(answers))

    result = _prepare_selected_optimization(tmp_path, run['run_id'], interactive=True)

    assert result['status'] == 'prepared'
    assert load_run(tmp_path, run['run_id'])[1]['optimization_choices']['budget_usd_per_hour'] == 4


VLLM = "vllm serve org/target --speculative-config '{\"model\":\"org/draft\",\"method\":\"eagle3\",\"num_speculative_tokens\":5}'"
SGLANG = 'python -m sglang.launch_server --model-path org/target --speculative-algorithm EAGLE3 --speculative-draft-model-path org/draft --speculative-num-steps 5 --speculative-eagle-topk 1'


@pytest.mark.parametrize('command,runtime', [(VLLM, 'vllm'), (SGLANG, 'sglang')])
def test_verified_launch_translates_actual_arguments(command, runtime):
    data = report(command)
    args, heads = serving_arguments(MODEL, [runtime], data['sources'], data['inspection']['heads'])
    values = {a.split('=', 1)[0]: ast.literal_eval(a.split('=', 1)[1])[0] for a in args[runtime]}
    if runtime == 'vllm':
        assert json.loads(values['speculative_config'])['num_speculative_tokens'] == 5
    else:
        assert values['speculative_num_steps'] == 5
        assert values['speculative_draft_model_path'] == 'org/draft'
    assert heads[0]['source_url'] == URL


@pytest.mark.parametrize('change', ['metadata', 'runtime', 'target', 'inferred', 'unsourced'])
def test_mentions_and_mismatched_pairs_do_not_enable_heads(change):
    data = report(VLLM)
    if change == 'metadata':
        data['sources'][0]['retrieved_text'] = 'draft_model=org/draft; compatible with vllm'
    elif change == 'runtime':
        data['inspection']['heads'][0]['runtime'] = 'sglang'
    elif change == 'target':
        data['sources'][0]['retrieved_text'] = VLLM.replace('org/target', 'org/other')
    elif change == 'inferred':
        data['inspection']['heads'][0]['status'] = 'inferred'
    else:
        data['sources'][0]['source_url'] = 'https://example.org/mention'
    assert serving_arguments(MODEL, ['vllm'], data['sources'], data['inspection']['heads']) == ({}, [])


def test_preparation_saves_exact_intent_request_and_reuses_saved_preferences(tmp_path):
    pytest.importorskip('llm_optimizer.remote.contracts')
    researcher = Mock(return_value=report(VLLM))
    result = prepare(selected(), inputs(), execution(), tmp_path, tmp_path, researcher)
    assert result['status'] == 'prepared'
    intent = json.loads((tmp_path / 'intent.json').read_text())
    request = json.loads((tmp_path / 'request.json').read_text())
    assert intent['context']['revision'] == request['revision'] == REV
    assert request['images'] == execution()['images']
    assert request['timeout_minutes'] == 120
    assert request['optuna']['trials'] == 3
    assert 'random_input_len=128' in request['client_args']
    assert any(a.startswith('speculative_config=') for a in request['server_args']['vllm'])
    assert result['summary']['maximum_gpu_cost_usd'] == 4
    prepare(selected(), inputs(), execution(), tmp_path, tmp_path, researcher)
    researcher.assert_called_once()


def test_only_missing_choices_are_requested_before_research(tmp_path):
    research = Mock(side_effect=AssertionError('research should wait for required choices'))
    values = inputs()
    del values['input_tokens']
    result = prepare(selected(), values, execution(), tmp_path, tmp_path, research)
    assert result['needs_input'] == ['input_tokens']
    assert not (tmp_path / 'request.json').exists()


def test_unverified_defaults_cannot_enable_speculation(tmp_path):
    config = execution()
    config['server_args'] = {'vllm': ['speculative_config=bad']}
    with pytest.raises(ValueError, match='Speculative defaults'):
        prepare(selected(), inputs(), config, tmp_path, tmp_path, lambda *a: report(VLLM))


def test_selection_automatically_prepares_and_missing_choices_survive_restart(tmp_path, monkeypatch):
    import hf_model_watch.optimization.preparation as preparation
    monkeypatch.setenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', json.dumps(execution()))
    monkeypatch.setattr(preparation, 'research', lambda *a: report(VLLM))
    # Replace the CLI boundary only; actual NanoClaw selection/preparation executes.
    def select(self, raw):
        run = selected()
        save_run(tmp_path / run['run_id'], run)
        return run
    monkeypatch.setattr(OpenClawToolAdapter, 'select_model', select)
    save_run(tmp_path / 'run_prepare', selected())
    adapter = ConversationAdapter(state_root=tmp_path)
    remembered = inputs()
    del remembered['input_tokens']
    adapter.update_workload(remembered)
    result = adapter.select_model({'run_id': 'run_prepare', 'model_id': '1', 'confirmed': True})
    assert result['optimization_preparation']['needs_input'] == ['input_tokens']
    # Avoid network by replacing the default researcher bound at function definition.
    original = preparation.prepare
    monkeypatch.setattr(preparation, 'prepare', lambda *a: original(*a, researcher=lambda *a: report(VLLM)))
    resumed = ConversationAdapter(state_root=tmp_path)
    assert resumed.prepare_optimization({'run_id': 'run_prepare', 'input_tokens': 128})['status'] == 'prepared'
    assert load_run(tmp_path, 'run_prepare')[1]['optimization_choices']['input_tokens'] == 128
    assert not load_run(tmp_path, 'run_prepare')[1].get('optimization_reference')
    assert resumed.get_run_status({'run_id': 'run_prepare'})['optimization_preparation']['status'] == 'prepared'


def test_mcp_approval_submits_prepared_arguments_only(tmp_path, monkeypatch):
    from hf_model_watch.nanoclaw_mcp import NanoClawMcpServer
    from hf_model_watch.optimization.workflow import OptimizationWorkflow
    from hf_model_watch.orchestration.optimization_adapter import OptimizationAdapter
    run = selected()
    run['optimization_choices'] = inputs()
    directory = tmp_path / 'run_prepare' / 'optimization-prepared'
    directory.mkdir(parents=True)
    run['optimization_preparation'] = prepare(run, {}, execution(), directory, tmp_path, lambda *a: report(VLLM))
    save_run(tmp_path / 'run_prepare', run)
    controller = Mock()
    controller.submit.return_value = {'run_id': 'remote', 'state': 'queued'}
    adapter = ConversationAdapter(state_root=tmp_path, optimization_workflow=OptimizationWorkflow(
        tmp_path, OptimizationAdapter(controller, execution())))
    monkeypatch.setenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', json.dumps(execution()))
    monkeypatch.setattr(adapter, 'get_optimization_connection', lambda _: {'connected': True})
    server = NanoClawMcpServer(adapter)
    args = {'run_id': 'run_prepare', 'approved_gpu_type': 'gpu-h100-sxm',
            'approved_gpu_hourly_cost_usd': 2, 'approved_timeout_minutes': 120, 'max_cost_usd': 4}
    def call():
        return server.handle({'id': 1, 'method': 'tools/call', 'params': {'name': 'start_optimization', 'arguments': args}})['result']
    assert call()['isError']
    controller.submit.assert_not_called()
    args['user_approved'] = True
    assert not call().get('isError')
    controller.submit.assert_called_once()
    submitted = controller.submit.call_args.args[0]
    assert submitted == json.loads((directory / 'request.json').read_text())
    assert any(a.startswith('speculative_config=') for a in submitted['server_args']['vllm'])


@pytest.mark.parametrize('field,value', [('target_revision', 'b' * 40), ('serving_image', 'other:image')])
def test_head_requires_exact_revision_and_runtime_image(tmp_path, field, value):
    data = report(VLLM)
    data['inspection']['heads'][0][field] = value
    result = prepare(selected(), inputs(), execution(), tmp_path, tmp_path, lambda *a: data)
    assert result['summary']['speculative_heads'] == []
    assert not any(a.startswith('speculative_config=') for a in json.loads((tmp_path / 'request.json').read_text())['server_args']['vllm'])


def test_required_speculation_without_verified_head_blocks_preparation(tmp_path):
    run = selected()
    run['request_spec']['speculative_decoding'] = 'require'
    data = report(VLLM)
    data['inspection']['heads'][0]['status'] = 'inferred'
    result = prepare(run, inputs(), execution(), tmp_path, tmp_path, lambda *a: data)
    assert result['status'] == 'insufficient_evidence'
    assert not (tmp_path / 'request.json').exists()


def test_nemotron_inspects_pinned_card_and_cites_only_supplied_urls(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    import openai
    from hf_model_watch.optimization.preparation import research
    from hf_model_watch.recommender import core
    card = tmp_path / 'README.md'
    card.write_text(VLLM)
    download = Mock(return_value=str(card))
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    monkeypatch.setenv('NEBIUS_API_KEY', 'placeholder')
    monkeypatch.setattr(core, 'tavily_research', Mock(return_value=[]))
    claim = {'claim': 'Documented launch', 'status': 'verified', 'source_urls': [URL]}
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
        message=SimpleNamespace(content=json.dumps({'claims': [claim], 'heads': []})))]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: client)
    result = research(MODEL, REV, tmp_path, core.Path('config/recommender.yaml'), {'vllm': 'pinned:image'})
    download.assert_called_once_with(MODEL, 'README.md', revision=REV)
    messages = create.call_args.kwargs['messages']
    payload = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
    assert any('do not propose max_model_len' in m['content']
               for m in messages if m['role'] == 'system')
    assert payload['revision'] == REV
    assert payload['serving_images'] == {'vllm': 'pinned:image'}
    assert result['inspection']['claims'] == [claim]
    claim['source_urls'] = ['https://invented.example']
    create.return_value.choices[0].message.content = json.dumps({'claims': [claim]})
    with pytest.raises(ValueError, match='cite supplied sources'):
        research(MODEL, REV, tmp_path, core.Path('config/recommender.yaml'), {'vllm': 'pinned:image'})


def test_selected_framework_uses_only_its_configured_search_defaults(tmp_path):
    config = execution()
    config['images']['sglang'] = 'other:image'
    config['optuna_by_framework'] = {'vllm': {'trials': 7}, 'sglang': {'trials': 9}}
    values = {**inputs(), 'frameworks': ['vllm']}
    prepare(selected(), values, config, tmp_path, tmp_path, lambda *a: report(VLLM))
    payload = json.loads((tmp_path / 'request.json').read_text())
    assert payload['frameworks'] == ['vllm']
    assert payload['optuna_by_framework'] == {'vllm': {'trials': 7}}


def test_missing_infrastructure_is_reported_without_research(tmp_path):
    researcher = Mock(side_effect=AssertionError('must not research'))
    result = prepare(selected(), inputs(), {'images': {'vllm': 'image'}}, tmp_path, tmp_path, researcher)
    assert result['needs_input'] == ['execution.bucket']
    researcher.assert_not_called()


def test_cli_selection_prepares_and_start_requires_approval(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    import hf_model_watch.optimization.preparation as preparation
    monkeypatch.setenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', json.dumps(execution()))
    monkeypatch.setattr(preparation, 'research', lambda *a: report(VLLM))
    run = selected()
    run['status'] = 'recommended'
    run['recommendations'][0]['revision'] = REV
    run['request_spec'] = inputs()
    save_run(tmp_path / run['run_id'], run)
    runner = CliRunner()
    result = runner.invoke(app, ['select', '--run-id', run['run_id'], '--model-id', MODEL,
                                 '--state-root', str(tmp_path), '--json'])
    assert result.exit_code == 0, result.output
    prepared = json.loads(result.output)['optimization_preparation']
    assert prepared['status'] == 'prepared'
    assert '--user-approved' in prepared['next_command']
    assert not load_run(tmp_path, run['run_id'])[1].get('optimization_reference')
    denied = runner.invoke(app, ['optimization', 'start', '--run-id', run['run_id'],
                                 '--state-root', str(tmp_path), '--request-file', prepared['intent_file']])
    assert denied.exit_code == 1
    assert 'explicitly approve' in denied.output


def test_speculative_json_survives_worker_and_serving_command_serialization(tmp_path):
    import shlex
    from llm_optimizer.remote.worker import optimizer_command
    from llm_optimizer import args as optimizer_args, predefined
    prepare(selected(), inputs(), execution(), tmp_path, tmp_path, lambda *a: report(VLLM))
    request = json.loads((tmp_path / 'request.json').read_text())
    command = optimizer_command({'request': request, 'framework': 'vllm'}, tmp_path / 'out.json')
    value = next(command[i+1] for i,a in enumerate(command) if a == '--server-args' and command[i+1].startswith('speculative_config='))
    sets = optimizer_args.parse_args_str(value, scope=optimizer_args.ArgScope.SERVER,
                                         configs=predefined.SERVER_CONFIGS['vllm'], strict=False)
    arg = sets[0].get_all_possible_arg_values()[0]
    rendered = predefined.get_server_cmd_args('vllm', [arg])
    argv = shlex.split('vllm serve org/target ' + ' '.join(rendered))
    config = next(a.split('=', 1)[1] for a in argv if a.startswith('--speculative-config='))
    assert json.loads(config) == {'model': 'org/draft', 'method': 'eagle3', 'num_speculative_tokens': 5}


def test_nemotron_serving_settings_require_documented_values(tmp_path):
    data = report(VLLM + ' --dtype bfloat16 --max-model-len 8192')
    data['inspection']['serving_settings'] = [{'runtime': 'vllm', 'target_revision': REV,
        'serving_image': 'pinned:image', 'status': 'verified', 'source_urls': [URL],
        'arguments': {'dtype': 'bfloat16', 'max_model_len': 4096, 'unrelated': 'ignored'}}]
    result = prepare(selected(), inputs(), execution(), tmp_path, tmp_path, lambda *a: data)
    assert result['summary']['serving_settings'][0]['arguments'] == {'dtype': 'bfloat16'}
    payload = json.loads((tmp_path / 'request.json').read_text())
    assert "dtype=['bfloat16']" in payload['server_args']['vllm']
    assert not any(a.startswith('max_model_len=') for a in payload['server_args']['vllm'])


@pytest.mark.parametrize('runtime', ['vllm', 'sglang', 'trtllm'])
def test_research_never_injects_context_length(tmp_path, runtime):
    data = report('serve org/target --max-model-len 262144 --gpu-memory-utilization 0.8')
    data['inspection']['serving_settings'] = [{
        'runtime': runtime, 'target_revision': REV, 'serving_image': 'pinned:image',
        'status': 'verified', 'source_urls': [URL],
        'arguments': {'max_model_len': 262144, 'gpu_memory_utilization': 0.8}}]
    config = execution()
    config['images'] = {runtime: 'pinned:image'}
    result = prepare(selected(), {**inputs(), 'frameworks': [runtime]}, config,
                     tmp_path, tmp_path, lambda *a: data)
    args = json.loads((tmp_path / 'request.json').read_text())['server_args'][runtime]
    assert not any(a.startswith('max_model_len=') for a in args)
    if runtime != 'vllm':
        assert not any(a.startswith('gpu_memory_utilization=') for a in args)
        assert result['summary']['serving_settings'] == []


@pytest.mark.parametrize('proposed', [['bfloat16', 'bfloat16'], ['bfloat16', 'float16']])
def test_research_deduplicates_settings_and_omits_conflicts(tmp_path, proposed):
    data = report('vllm serve org/target --dtype bfloat16\nvllm serve org/target --dtype float16')
    data['inspection']['serving_settings'] = [{
        'runtime': 'vllm', 'target_revision': REV, 'serving_image': 'pinned:image',
        'status': 'verified', 'source_urls': [URL], 'arguments': {'dtype': dtype},
    } for dtype in proposed]
    result = prepare(selected(), inputs(), execution(), tmp_path, tmp_path, lambda *a: data)
    args = json.loads((tmp_path / 'request.json').read_text())['server_args']['vllm']
    assert [a for a in args if a.startswith('dtype=')] == (
        ["dtype=['bfloat16']"] if len(set(proposed)) == 1 else [])
    if len(set(proposed)) > 1:
        assert result['summary']['serving_settings'] == []


@pytest.mark.parametrize('scope', ['server_args', 'optuna', 'optuna_by_framework'])
def test_explicit_settings_take_precedence_over_research(tmp_path, scope):
    data = report('vllm serve org/target --dtype bfloat16 --max-model-len 262144')
    data['inspection']['serving_settings'] = [{
        'runtime': 'vllm', 'target_revision': REV, 'serving_image': 'pinned:image',
        'status': 'verified', 'source_urls': [URL],
        'arguments': {'dtype': 'bfloat16', 'max_model_len': 262144}}]
    config = execution()
    args = ["dtype=['float16']", 'max_model_len=[8192]']
    if scope == 'server_args':
        config[scope] = {'vllm': args}
    elif scope == 'optuna':
        config[scope]['server_args'] = args
    else:
        config[scope] = {'vllm': {'server_args': args}}
    result = prepare(selected(), inputs(), config, tmp_path, tmp_path, lambda *a: data)
    payload = json.loads((tmp_path / 'request.json').read_text())
    effective = [*payload['server_args']['vllm'],
                 *payload['optuna'].get('server_args', []),
                 *payload['optuna_by_framework'].get('vllm', {}).get('server_args', [])]
    assert [a for a in effective if a.startswith('dtype=')] == ["dtype=['float16']"]
    assert [a for a in effective if a.startswith('max_model_len=')] == ['max_model_len=[8192]']
    assert result['summary']['serving_settings'] == []


def test_research_failure_preserves_selection_and_reports_incomplete(tmp_path, monkeypatch):
    import hf_model_watch.optimization.preparation as preparation
    run = selected()
    run['optimization_choices'] = inputs()
    save_run(tmp_path / run['run_id'], run)
    monkeypatch.setenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', json.dumps(execution()))
    monkeypatch.setattr(preparation, 'research', Mock(side_effect=ValueError('Incomplete Nemotron inspection')))
    result = ConversationAdapter(state_root=tmp_path).prepare_optimization({'run_id': run['run_id']})
    assert result['status'] == 'preparation_failed'
    assert result['reason'] == 'Incomplete Nemotron inspection'
    saved = load_run(tmp_path, run['run_id'])[1]
    assert saved['selected_model_id'] == MODEL
    assert not saved.get('optimization_reference')


def test_interactive_discovery_defers_optimization_choices_until_selection(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    from hf_model_watch.recommender import discovery
    import hf_model_watch.optimization.preparation as preparation
    monkeypatch.delenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', raising=False)
    monkeypatch.setattr(preparation, 'research', lambda *a: report(VLLM))
    def discover(*args, **kwargs):
        print('RESEARCH STARTED')
        return {'status': 'recommended', 'recommendations': [
            {'model_id': MODEL, 'revision': REV, 'hf_score': .8, 'why_it_fits': ['fits'], 'limitations': []}]}
    monkeypatch.setattr(discovery, 'discover_models', discover)
    args = ['recommend', 'coding model', '--gpu-memory-gb', '80', '--gpu-count', '1',
            '--budget-usd-per-hour', '4', '--gpu-hourly-cost-usd', '2', '--license', 'any',
            '--framework-mode', 'sweep', '--speculative-decoding', 'off', '--state-root', str(tmp_path)]
    result = CliRunner().invoke(app, args, input='1\ngpu-h100-sxm\n128\n32\n4\nvllm\nmy-bucket\n')
    assert result.exit_code == 0, result.output
    assert 'RESEARCH STARTED' in result.output
    assert 'Optimization preparation: prepared' in result.output
    assert result.output.index('Selected model:') < result.output.index('Nebius GPU platform')
    assert list(tmp_path.glob('run_*/optimization-prepared/request.json'))
    assert not (tmp_path / 'conversation.json').exists()


def test_nanoclaw_discovers_without_optimizer_intake(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    from hf_model_watch.recommender import discovery
    from hf_model_watch.nanoclaw_mcp import NanoClawMcpServer
    monkeypatch.delenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', raising=False)
    discover = Mock(return_value={'status': 'recommended', 'recommendations': [
        {'model_id': MODEL, 'revision': REV}]})
    monkeypatch.setattr(discovery, 'discover_models', discover)
    def cli(argv, timeout):
        result = CliRunner().invoke(app, argv)
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)
    server = NanoClawMcpServer(ConversationAdapter(state_root=tmp_path, cli_runner=cli))
    def call(name, arguments):
        response = server.handle({'id': 1, 'method': 'tools/call', 'params': {
            'name': name, 'arguments': arguments}})['result']
        assert not response.get('isError'), response
        return json.loads(response['content'][0]['text'])
    initial = call('recommend_models', {'request': 'coding model', 'gpu_memory_gb': 80,
        'budget_usd_per_hour': 4, 'gpu_hourly_cost_usd': 2, 'license': 'any',
        'framework_mode': 'sweep', 'speculative_decoding': 'off'})
    assert initial['status'] == 'recommended'
    discover.assert_called_once()
    assert 'needs_input' not in initial or not initial['needs_input']
    assert not load_run(tmp_path, initial['run_id'])[1].get('optimization_reference')


def test_intake_reuses_configured_infrastructure_and_saved_workload():
    from hf_model_watch.optimization.intake import intake
    values, config, missing = intake(inputs(), execution())
    assert missing == []
    assert values['frameworks'] == ['vllm']
    assert all(config[key] == value for key, value in execution().items())
    _, _, missing = intake({**inputs(), 'frameworks': ['sglang']}, execution())
    assert missing == []
    from hf_model_watch.optimization.intake import DEFAULT_IMAGES
    assert intake({**inputs(), 'frameworks': ['sglang']}, execution())[1]['images']['sglang'] == DEFAULT_IMAGES['sglang']


@pytest.mark.parametrize('gpu,count,preset', [
    ('gpu-h100-sxm', 1, '1gpu-16vcpu-200gb'),
    ('gpu-h200-sxm', 1, '1gpu-16vcpu-200gb'),
    ('gpu-h100-sxm', 8, '8gpu-128vcpu-1600gb'),
    ('gpu-h200-sxm', 8, '8gpu-128vcpu-1600gb'),
    ('gpu-l40s-a', 1, '1gpu-8vcpu-32gb'),
    ('gpu-l40s-d', 1, '1gpu-16vcpu-96gb'),
])
def test_authorized_image_and_gpu_preset_defaults(gpu, count, preset):
    from hf_model_watch.optimization.intake import intake, DEFAULT_IMAGES
    values = {**inputs(), 'gpu_type': gpu, 'gpu_count': count,
              'frameworks': ['sglang', 'vllm', 'trtllm']}
    _, config, missing = intake(values, {'bucket': 'bucket'})
    assert not missing
    assert config['preset'] == preset
    assert config['images'] == DEFAULT_IMAGES
    assert config['images'] == {
        framework: f'cr.eu-north1.nebius.cloud/e00p24e8mwqab11v1e/{framework}:sf261025-1'
        for framework in ('vllm', 'sglang', 'trtllm')
    }
    assert all(len(image) <= 64 for image in config['images'].values())
    _, custom, missing = intake({**values, 'optimizer_preset': f'{count}gpu-custom'},
                                {'bucket': 'bucket', 'images': {'vllm': 'custom:image'}})
    assert not missing
    assert custom['preset'] == f'{count}gpu-custom'
    assert custom['images']['vllm'] == 'custom:image'


@pytest.mark.parametrize('gpu,count', [('gpu-l40s-d', 8), ('gpu-l40s-a', 8), ('gpu-h100-sxm', 2)])
def test_unlisted_gpu_counts_require_explicit_preset(gpu, count):
    from hf_model_watch.optimization.intake import intake
    _, config, missing = intake({**inputs(), 'gpu_type': gpu, 'gpu_count': count,
                                 'frameworks': ['vllm']}, {'bucket': 'bucket'})
    assert missing == ['optimizer_preset']
    assert 'preset' not in config


def test_single_prepared_request_fans_out_to_isolated_framework_workers(tmp_path, monkeypatch):
    """Exercise real controller/command builders with all cloud boundaries mocked."""
    import base64
    from types import SimpleNamespace
    from llm_optimizer.remote.adapters import NebiusAdapter
    from llm_optimizer.remote.controller import Controller
    from llm_optimizer.remote.worker import optimizer_command
    from hf_model_watch.optimization.intake import DEFAULT_IMAGES

    monkeypatch.delenv('NEBIUS_DEBUG_SSH_PUBLIC_KEY', raising=False)
    frameworks = ['vllm', 'sglang', 'trtllm']
    config = {**execution(), 'images': DEFAULT_IMAGES, 'dry_run': False,
              'server_args': {f: [f'max_model_len={1024 * (i + 1)}'] for i, f in enumerate(frameworks)},
              'optuna_by_framework': {f: {'trials': i + 5} for i, f in enumerate(frameworks)}}
    values = {**inputs(), 'frameworks': frameworks, 'budget_usd_per_hour': 6}
    empty_research = lambda *a: {'sources': [], 'inspection': {'heads': []}}
    prepared = prepare(selected(), values, config, tmp_path / 'prepared', tmp_path, empty_research)
    assert prepared['status'] == 'prepared'
    request = json.loads((tmp_path / 'prepared/request.json').read_text())
    assert request['frameworks'] == frameworks
    assert len(list((tmp_path / 'prepared').glob('request.json'))) == 1

    # Real Nebius command generation, fake execution: no subprocess or GPU launch.
    commands = []
    def capture(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({'id': f'aijob-test{len(commands)}'}))
    provider = Mock(wraps=NebiusAdapter('test-project', run_command=capture))
    provider.status.return_value = 'running'
    store = Mock()
    store.progress.return_value = None
    store_factory = Mock(return_value=store)
    database = tmp_path / 'test-controller.sqlite'
    controller = Controller(database, nebius=provider, s3_factory=store_factory, concurrency=3)
    run = controller.submit(request, idempotency_key='fanout-test')
    assert {j['framework'] for j in run['jobs']} == set(frameworks)
    assert all(j['state'] == 'queued' for j in run['jobs'])
    assert not commands
    controller.tick()
    assert len(commands) == 3

    for framework in frameworks:
        command = next(c for c in commands if c[c.index('--name') + 1].endswith('-' + framework))
        assert command[command.index('--image') + 1] == DEFAULT_IMAGES[framework]
        assert command[command.index('--platform') + 1] == values['gpu_type']
        assert command[command.index('--preset') + 1] == request['preset']
        encoded = next(v for v in command if v.startswith(('LLM_OPTIMIZER_PAYLOAD_B64=', 'LLM_OPTIMIZER_PAYLOAD_ZLIB_B64=')))
        import zlib
        raw = base64.b64decode(encoded.split('=', 1)[1])
        payload = json.loads(zlib.decompress(raw) if 'ZLIB' in encoded else raw)
        assert payload['framework'] == framework
        worker = optimizer_command(payload, tmp_path / f'{framework}.json')
        assert worker[worker.index('--framework') + 1] == framework
        assert worker[worker.index('--optuna-trials') + 1] == str(config['optuna_by_framework'][framework]['trials'])
        server_args = [worker[i + 1] for i, arg in enumerate(worker) if arg == '--server-args']
        revision_key = 'revision'
        assert f'{revision_key}={REV}' in server_args
        assert config['server_args'][framework][0] in server_args
        for other in set(frameworks) - {framework}:
            assert config['server_args'][other][0] not in server_args
        client_args = [worker[i + 1] for i, arg in enumerate(worker) if arg == '--client-args']
        assert 'random_input_len=128' in client_args
        assert 'random_output_len=32' in client_args
        assert 'max_concurrency=4' in client_args

    # Restart and repeat submission must neither duplicate the study nor its jobs.
    restarted = Controller(database, nebius=provider, s3_factory=store_factory, concurrency=3)
    assert restarted.submit(request, idempotency_key='fanout-test')['run_id'] == run['run_id']
    restarted.tick()
    assert len(commands) == 3


@pytest.mark.parametrize('policy', [None, 'spot-policy-example'])
def test_spot_defaults_are_saved_and_reach_nebius_command(tmp_path, monkeypatch, policy):
    from types import SimpleNamespace
    import base64
    from llm_optimizer.remote.adapters import NebiusAdapter
    if policy:
        monkeypatch.setenv('NEBIUS_SPOT_PRICING_POLICY_ID', policy)
    else:
        monkeypatch.delenv('NEBIUS_SPOT_PRICING_POLICY_ID', raising=False)
    monkeypatch.delenv('NEBIUS_DEBUG_SSH_PUBLIC_KEY', raising=False)
    prepared = prepare(selected(), inputs(), execution(), tmp_path, tmp_path,
                       lambda *a: {'sources': [], 'inspection': {}})
    request = json.loads((tmp_path / 'request.json').read_text())
    assert request['preemptible'] is True
    assert request['spot_pricing_policy_id'] == policy
    assert request['follows_spot_price'] is (policy is None)
    assert prepared['summary']['spot_pricing_policy_id'] == policy
    assert prepared['summary']['follows_spot_price'] is (policy is None)
    # A later controller environment change must not change the reviewed policy.
    monkeypatch.setenv('NEBIUS_SPOT_PRICING_POLICY_ID', 'different-policy')
    run = Mock(return_value=SimpleNamespace(stdout='{"id":"aijob-test"}'))
    adapter = NebiusAdapter('test-project', run_command=run)
    adapter.submit({'run_id': 'test', 'framework': 'vllm', 'request': {**request, 'dry_run': False}})
    command = run.call_args.args[0]
    assert '--preemptible' in command
    if policy:
        assert command[command.index('--spot-pricing-policy-id') + 1] == policy
        assert '--follows-spot-price' not in command
    else:
        assert '--follows-spot-price' in command
        assert '--spot-pricing-policy-id' not in command
    encoded = next(v.split('=', 1)[1] for v in command if v.startswith('LLM_OPTIMIZER_PAYLOAD_B64='))
    worker = json.loads(base64.b64decode(encoded))['request']
    assert not {'preemptible', 'spot_pricing_policy_id', 'follows_spot_price'} & worker.keys()


def test_nonpreemptible_override_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('NEBIUS_SPOT_PRICING_POLICY_ID', 'spot-policy-example')
    config = {**execution(), 'preemptible': False}
    prepare(selected(), inputs(), config, tmp_path, tmp_path, lambda *a: {'sources': [], 'inspection': {}})
    request = json.loads((tmp_path / 'request.json').read_text())
    assert request['preemptible'] is False
    assert request['spot_pricing_policy_id'] is None
    assert request['follows_spot_price'] is False


@pytest.mark.parametrize('pricing', [
    {'preemptible': True, 'spot_pricing_policy_id': 'policy', 'follows_spot_price': True},
    {'preemptible': False, 'follows_spot_price': True},
    {'preemptible': True, 'follows_spot_price': False},
    {'preemptible': True, 'spot_pricing_policy_id': ''},
    {'preemptible': True, 'follows_spot_price': 'yes'},
])
def test_invalid_spot_pricing_combinations_are_rejected(pricing):
    from llm_optimizer.remote.contracts import OptimizationRequest
    with pytest.raises(ValueError, match='spot|Spot|Preemptible'):
        OptimizationRequest.parse({**execution(), 'model': MODEL, 'frameworks': ['vllm'],
                                   'gpu_type': 'gpu-h100-sxm', **pricing})



def test_default_spot_estimate_allows_three_concurrent_frameworks(tmp_path):
    values = {**inputs(), 'frameworks': ['sglang', 'trtllm', 'vllm'],
              'concurrency': 128, 'budget_usd_per_hour': 5}
    del values['gpu_hourly_cost_usd']
    result = prepare(selected(), values, {'bucket': 'bucket'}, tmp_path, tmp_path,
                     lambda *a: {'sources': [], 'inspection': {}})
    assert result['status'] == 'prepared'
    assert result['summary']['gpu_hourly_cost_usd'] == 1
    assert result['summary']['maximum_hourly_gpu_cost_usd'] == 3
    request = json.loads((tmp_path / 'request.json').read_text())
    assert request['frameworks'] == ['sglang', 'trtllm', 'vllm']
    assert 'max_concurrency=128' in request['client_args']
    assert 'max_parallel_frameworks' not in request


def test_explicit_gpu_price_is_not_replaced_by_spot_default(tmp_path):
    values = {**inputs(), 'frameworks': ['sglang', 'trtllm', 'vllm'],
              'gpu_hourly_cost_usd': 5, 'budget_usd_per_hour': 5}
    research = Mock(side_effect=AssertionError('over-budget preparation must stop before research'))
    result = prepare(selected(), values, {'bucket': 'bucket'}, tmp_path, tmp_path, research)
    assert result['status'] == 'needs_input'
    assert result['needs_input'] == ['gpu_hourly_cost_usd', 'budget_usd_per_hour']
    research.assert_not_called()


def test_discovery_does_not_invent_gpu_price(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    from hf_model_watch.recommender import discovery
    captured = {}
    def discover(spec, *args, **kwargs):
        captured['price'] = spec.gpu_hourly_cost_usd
        return {'status': 'recommended', 'recommendations': [{'model_id': MODEL, 'revision': REV}]}
    monkeypatch.setattr(discovery, 'discover_models', discover)
    args = ['recommend', 'coding model', '--json', '--state-root', str(tmp_path),
            '--gpu-type', 'gpu-h100-sxm', '--gpu-memory-gb', '80', '--budget-usd-per-hour', '5',
            '--license', 'any', '--framework-mode', 'sweep', '--speculative-decoding', 'off',
            '--input-tokens', '512', '--output-tokens', '128', '--concurrency', '128',
            '--frameworks', 'sglang,trtllm,vllm', '--optimizer-bucket', 'bucket']
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response['status'] == 'recommended'
    assert captured['price'] is None
    saved = load_run(tmp_path, response['run_id'])[1]
    assert saved['request_spec']['gpu_hourly_cost_usd'] is None


def test_prepare_cli_corrects_saved_price_without_rerunning_discovery(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    import hf_model_watch.optimization.preparation as preparation
    run = selected()
    run['request_spec'] = {**inputs(), 'frameworks': ['sglang', 'trtllm', 'vllm'],
                           'gpu_hourly_cost_usd': 5, 'budget_usd_per_hour': 5}
    run['optimization_execution'] = {'bucket': 'bucket'}
    save_run(tmp_path / run['run_id'], run)
    monkeypatch.setattr(preparation, 'research', lambda *a: {'sources': [], 'inspection': {}})
    result = CliRunner().invoke(app, ['optimization', 'prepare', '--run-id', run['run_id'],
        '--state-root', str(tmp_path), '--gpu-hourly-cost-usd', '1'])
    assert result.exit_code == 0, result.output
    response = json.loads(result.stdout)
    assert response['status'] == 'prepared'
    assert response['summary']['maximum_hourly_gpu_cost_usd'] == 3
    saved = load_run(tmp_path, run['run_id'])[1]
    assert saved['optimization_choices']['gpu_hourly_cost_usd'] == 1
    assert not saved.get('optimization_reference')


def test_legacy_string_numeric_preferences_are_normalized_before_budget_check(tmp_path, monkeypatch):
    import hf_model_watch.optimization.preparation as preparation
    run = selected()
    run['request_spec'] = {**inputs(), 'frameworks': ['sglang', 'vllm', 'trtllm'],
        'gpu_count': '1', 'gpu_hourly_cost_usd': '5', 'budget_usd_per_hour': '5',
        'input_tokens': '512', 'output_tokens': '128', 'concurrency': '32'}
    researcher = Mock(return_value={'sources': [], 'inspection': {}})
    result = prepare(run, {}, {'bucket': 'bucket'}, tmp_path / 'direct', tmp_path, researcher)
    assert result['status'] == 'needs_input'
    researcher.assert_not_called()
    run['optimization_choices'] = {'budget_usd_per_hour': '5', 'gpu_hourly_cost_usd': '5'}
    run['optimization_execution'] = {'bucket': 'bucket'}
    save_run(tmp_path / run['run_id'], run)
    monkeypatch.setattr(preparation, 'research', researcher)
    adapter = ConversationAdapter(state_root=tmp_path)
    result = adapter.prepare_optimization({'run_id': run['run_id'], 'gpu_hourly_cost_usd': 1})
    assert result['status'] == 'prepared'
    assert result['summary']['maximum_hourly_gpu_cost_usd'] == 3
    saved = load_run(tmp_path, run['run_id'])[1]['optimization_choices']
    assert saved['budget_usd_per_hour'] == 5.0
    assert isinstance(saved['budget_usd_per_hour'], float)
    assert saved['gpu_count'] == 1


def test_interactive_numeric_answers_are_saved_as_numbers(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from hf_model_watch.cli import app
    from hf_model_watch.recommender import discovery
    import hf_model_watch.optimization.preparation as preparation
    monkeypatch.delenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', raising=False)
    monkeypatch.setattr(preparation, 'research', lambda *a: {'sources': [], 'inspection': {}})
    monkeypatch.setattr(discovery, 'discover_models', lambda *a, **kw: {
        'status': 'recommended', 'recommendations': [{'model_id': MODEL, 'revision': REV,
            'hf_score': .8, 'why_it_fits': ['fits'], 'limitations': []}]})
    args = ['recommend', 'coding model', '--state-root', str(tmp_path), '--gpu-type', 'gpu-h100-sxm',
        '--input-tokens', '512', '--output-tokens', '128', '--concurrency', '32',
        '--frameworks', 'sglang,vllm,trtllm', '--optimizer-bucket', 'bucket',
        '--license', 'any', '--framework-mode', 'sweep', '--speculative-decoding', 'off',
        '--gpu-memory-gb', '80', '--budget-usd-per-hour', '5', '--gpu-hourly-cost-usd', '1']
    result = CliRunner().invoke(app, args, input='1\n')
    assert result.exit_code == 0, result.output
    assert 'Optimization preparation: prepared' in result.output
    path = next(tmp_path.glob('run_*/recommendation_run.json'))
    state = json.loads(path.read_text())
    assert state['request_spec']['gpu_memory_gb'] == 80.0
    assert isinstance(state['request_spec']['budget_usd_per_hour'], float)
    assert state['request_spec']['gpu_hourly_cost_usd'] == 1.0
    assert state['optimization_preparation']['summary']['maximum_hourly_gpu_cost_usd'] == 3
    request = json.loads(Path(state['optimization_preparation']['request_file']).read_text())
    args = request['server_args']['sglang']
    names = [a.partition('=')[0].partition(':')[0] for a in args]
    assert len(names) == len(set(names))
    space = {a.partition('=')[0].partition(':')[0]: ast.literal_eval(a.partition('=')[2])
             for a in args}
    assert space['cuda_graph_backend_decode'] == ['disabled', 'full']
    assert space['cuda_graph_backend_prefill'] == ['disabled', 'breakable']
    assert space['attention_backend'] == ['triton', 'fa3', 'flashinfer']
    assert space['disable_radix_cache*schedule_policy'] == [
        [False, 'fcfs'], [False, 'lpm'], [True, 'fcfs'], [True, 'lpm']]
    assert not any(n in names for n in ('disable_radix_cache', 'schedule_policy'))


def test_preparation_reuses_full_optimizer_expanded_spaces_and_keeps_workload_fixed(tmp_path):
    from llm_optimizer.search_profiles import TOKEN_BUDGETS_128, CAPACITIES_128
    values = {**inputs(), 'frameworks': ['sglang', 'vllm', 'trtllm'],
              'concurrency': 32, 'budget_usd_per_hour': 6}
    result = prepare(selected(), values, {'bucket': 'bucket'}, tmp_path, tmp_path,
                     lambda *a: {'sources': [], 'inspection': {}})
    request = json.loads((tmp_path / 'request.json').read_text())
    assert result['status'] == 'prepared'
    assert result['summary']['search_profile'] == 'expanded'
    assert result['summary']['search_regime'] == 128
    assert request['optuna']['trials'] == 30
    assert request['optuna']['sampler'] == 'TPESampler'
    assert request['optuna']['objectives'] == ['output_throughput', 'mean_e2e_latency_ms']
    assert 'max_concurrency=32' in request['client_args']
    assert not request['optuna'].get('client_args')
    vllm = {a.partition('=')[0].partition(':')[0]: ast.literal_eval(a.partition('=')[2])
            for a in request['server_args']['vllm']}
    assert vllm['max_num_seqs'] == CAPACITIES_128
    assert vllm['max_num_batched_tokens'] == TOKEN_BUDGETS_128
    assert len(vllm['attention_backend']) == 3
    assert request['server_config_files']['trtllm']
    assert result['summary']['server_config_file_counts']['trtllm'] > 1


def test_explicit_framework_knobs_override_expanded_defaults_without_duplicates(tmp_path):
    from llm_optimizer.search_profiles import argument_names
    config = execution()
    config['server_args'] = {'vllm': ['max_num_batched_tokens=[2048,4096]', "attention_backend=['FLASHINFER']"]}
    config['optuna_by_framework'] = {'vllm': {'trials': 7, 'server_args': ['max_num_seqs=int(16,32,step=8)']}}
    prepare(selected(), inputs(), config, tmp_path, tmp_path, lambda *a: {'sources': [], 'inspection': {}})
    request = json.loads((tmp_path / 'request.json').read_text())
    args = request['server_args']['vllm']
    assert 'max_num_batched_tokens=[2048,4096]' in args
    assert "attention_backend=['FLASHINFER']" in args
    assert not any('max_num_seqs' in argument_names(a) for a in args)
    assert request['optuna_by_framework']['vllm']['trials'] == 7


def test_nanoclaw_live_and_trial_controls_reprepare_and_persist(tmp_path, monkeypatch):
    from hf_model_watch.optimization import preparation
    from hf_model_watch.nanoclaw_mcp import NanoClawMcpServer
    config = execution()
    config['optuna']['trials'] = 16
    config['optuna']['objectives'] = ['output_throughput', 'mean_e2e_latency_ms']
    config['optuna_by_framework'] = {'vllm': {'trials': 12}}
    monkeypatch.setenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', json.dumps(config))
    monkeypatch.setattr(preparation, 'research', lambda *a: {'sources': [], 'inspection': {}})
    run = selected()
    run['request_spec'].update(inputs())
    save_run(tmp_path / run['run_id'], run)
    adapter = ConversationAdapter(state_root=tmp_path)
    first = adapter.prepare_optimization({'run_id': run['run_id']})
    assert first['summary']['dry_run'] is True
    server = NanoClawMcpServer(adapter)
    response = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {
        'name': 'prepare_optimization', 'arguments': {'run_id': run['run_id'], 'dry_run': False, 'trials': 2}}})
    result = json.loads(response['result']['content'][0]['text'])
    assert result['status'] == 'prepared'
    assert result['summary']['dry_run'] is False
    assert result['summary']['optuna']['trials'] == 2
    assert result['request_sha256'] != first['request_sha256']
    request = json.loads(Path(result['request_file']).read_text())
    assert request['optuna']['objectives'] == config['optuna']['objectives']
    assert request['optuna']['sampler'] == 'TPESampler'
    assert request['optuna_by_framework']['vllm']['trials'] == 2
    assert request['timeout_minutes'] == config['timeout_minutes']
    again = adapter.prepare_optimization({'run_id': run['run_id']})
    assert again['summary']['dry_run'] is False
    assert again['summary']['optuna']['trials'] == 2
    _, saved = load_run(tmp_path, run['run_id'])
    assert not saved.get('optimization_reference')


def test_controller_allows_only_one_active_study_and_terminal_retry(tmp_path):
    from llm_optimizer.remote.controller import Controller
    prepare(selected(), inputs(), execution(), tmp_path / 'prepared', tmp_path, lambda *a: report(VLLM))
    request = json.loads((tmp_path / 'prepared/request.json').read_text())
    controller = Controller(tmp_path / 'queue.sqlite')
    first = controller.submit(request, idempotency_key='first')
    assert controller.submit(request, idempotency_key='first')['run_id'] == first['run_id']
    with pytest.raises(ValueError, match='still active'):
        controller.submit(request, idempotency_key='second')
    controller.cancel(first['run_id'])
    assert controller.submit(request, idempotency_key='second')['run_id'] != first['run_id']


def test_pending_controls_and_budget_answer_survive_fresh_adapters(tmp_path, monkeypatch):
    import hf_model_watch.optimization.preparation as preparation
    run = selected()
    run.pop('selected_model_id', None)
    run.pop('selection', None)
    run['request_spec'] = {**inputs(), 'budget_usd_per_hour': 0.5}
    save_run(tmp_path / 'run_prepare', run)
    monkeypatch.setenv('HF_MODEL_WATCH_OPTIMIZER_EXECUTION_JSON', json.dumps(execution()))
    monkeypatch.setattr(preparation, 'research', lambda *a: report(VLLM))
    pending = ConversationAdapter(state_root=tmp_path).prepare_optimization(
        {'run_id': 'run_prepare', 'dry_run': True, 'trials': 2})
    assert pending['needs_input'] == ['selected_model_id']
    # Match this fixture's actual selected model rather than invoke the CLI.
    model_id = selected()['selected_model_id']
    def select(self, raw):
        _, saved = load_run(tmp_path, raw['run_id'])
        saved.update(selected_model_id=model_id, selection=selected()['selection'])
        save_run(tmp_path / raw['run_id'], saved)
        return saved
    monkeypatch.setattr(OpenClawToolAdapter, 'select_model', select)
    view = ConversationAdapter(state_root=tmp_path).select_model(
        {'run_id': 'run_prepare', 'model_id': model_id, 'confirmed': True})
    assert view['optimization_preparation']['status'] == 'needs_input'
    ctx = ConversationAdapter(state_root=tmp_path).get_conversation_context({})
    assert ctx['run']['selected_model_id'] == model_id
    assert ctx['optimization_controls']['optuna']['trials'] == 2
    result = ConversationAdapter(state_root=tmp_path).provide_active_input(
        {'answers': {'gpu_hourly_cost_usd': 2, 'budget_usd_per_hour': 7}})
    assert result['status'] == 'prepared'
    assert result['summary']['dry_run'] is True
    assert result['summary']['optuna']['trials'] == 2
    assert ConversationAdapter(state_root=tmp_path).get_conversation_context({})['next_tool'] == 'start_optimization'

@pytest.mark.parametrize('mode', ['off', 'prefer'])
@pytest.mark.parametrize('model', ['org/target', 'Qwen/Qwen3-4B'])
def test_baseline_modes_do_not_use_explicit_dspark(tmp_path, monkeypatch, mode, model):
    def forbidden(*args):
        raise AssertionError('Baseline must not resolve a mapped head')
    monkeypatch.setattr('hf_model_watch.optimization.dspark.resolve', forbidden)
    monkeypatch.setattr('hf_model_watch.optimization.dspark.pinned_draft', forbidden)
    run = selected()
    run['selected_model_id'] = model
    run['request_spec'].update(speculative_method='auto', speculative_decoding=mode)
    data = {'model': model, 'revision': REV, 'sources': [], 'inspection': {'heads': [], 'claims': []}}
    researcher = Mock(return_value=data)
    result = prepare(run, inputs(), execution(), tmp_path, tmp_path, researcher)
    assert result['status'] == 'prepared'
    researcher.assert_called_once()
    payload = json.loads((tmp_path / 'request.json').read_text())
    assert not any('speculative' in a for a in payload['server_args']['vllm'])
    assert result['summary']['speculative_heads'] == []
