import json
import pytest

from hf_model_watch.recommender import discovery
from hf_model_watch.recommender.core import rank_candidates, load_settings
from hf_model_watch.recommender.capacity import estimate_memory
from hf_model_watch.recommender.utilities import discovery_inputs, speculative_bonus
from hf_model_watch.hub.metadata import parameter_estimate
from hf_model_watch.openclaw_adapter import OpenClawToolAdapter
from hf_model_watch.conversation import ConversationAdapter, write_json, run_worker
from tests.test_recommender import _spec, _record


def test_small_gpu_and_draft_fallback():
    assert estimate_memory(1)[0] == 2
    assert estimate_memory(16, True)[0] == pytest.approx(21.6)
    assert estimate_memory(16, True, 1)[0] == pytest.approx(18.6)
    r = rank_candidates([_record('org/small', .5)], _spec(gpu_memory_gb=8, speculative_decoding='off'), {}, load_settings())[0]
    assert r['eligible']


def test_multigpu_parsing_total_price_and_runtime_capacity():
    values, missing = discovery_inputs('Use two GPUs with 60 GB each costing $2/hour per GPU, or $4/hour total. Budget $5. Any license. Use sweep mode. Prefer speculative decoding.', {})
    assert not missing
    assert values['gpu_count'] == 2
    assert values['gpu_hourly_cost_usd'] == 2
    row = _record('Qwen/model-32B', 32)
    evidence = {row['model_id']: [{'source_url': 'https://huggingface.co/Qwen/model-32B',
                  'retrieved_text': 'vllm serve Qwen/model-32B'}]}
    r = rank_candidates([row], _spec(gpu_memory_gb=60, gpu_count=2, gpu_hourly_cost_usd=2, budget_usd_per_hour=5), evidence, load_settings())[0]
    assert r['eligible']
    assert r['estimated_hourly_usd'] == 4
    assert r['available_memory_gb'] == 102
    assert 'budget_unverified_or_exceeded' in rank_candidates([row], _spec(gpu_count=2, gpu_hourly_cost_usd=3, budget_usd_per_hour=5), evidence, load_settings())[0]['blocking_reasons']


@pytest.mark.parametrize('ident,expected', [('nvidia/Gemma-4-31B-IT-NVFP4',31), ('nvidia/Qwen3.6-35B-A3B-NVFP4',35)])
def test_quantized_counts_are_not_packed_tensor_elements(ident, expected):
    assert parameter_estimate({'quantization_config': {'bits':4}}, [], ident, 18.684) == expected


def test_generic_inferred_and_wrong_method_do_not_receive_bonus():
    s = load_settings()
    s['selection']['speculative_decoding']['inferred_bonus'] = 100
    assert speculative_bonus('inferred','prefer',s) == 0
    row = _record('Qwen/model-4B',4)
    source = {'source_url':'https://huggingface.co/org/DFlash',
              'retrieved_text':'Draft model for Qwen/model-4B. Use vllm.'}
    r = rank_candidates([row], _spec(request='Prefer Medusa speculative decoding'), {row['model_id']:[source]}, s)[0]
    assert r['feature_bonus'] == 0


def test_reporting_uses_effective_inputs_and_keeps_rejections():
    result = OpenClawToolAdapter._safe_run_view({'constraints': {'gpu_memory_gb':None},
        'request_spec': {'gpu_memory_gb':60,'gpu_count':2}, 'candidate_rejections': {'org/model':['license_blocked']}})
    assert result['constraints']['gpu_count'] == 2
    assert result['constraints']['gpu_memory_gb'] == 60
    assert result['candidate_rejections']


def test_closest_candidates_complete_methods_and_smaller_fallback_is_labeled(tmp_path, monkeypatch):
    rows=[]
    for n in [4,8,13,14]:
        row=_record(f'Qwen/model-{n}B',n)
        row.update(has_tokenizer=True,has_safetensors=True,chat_template_present=True,
                   estimated_download_size_gb=n*2,downloads=100000)
        rows.append(row)
    monkeypatch.setattr(discovery, '_validate_with_nemotron', lambda *a: None)
    calls=[]
    def search(query,*a,extract_urls=None):
        if not extract_urls:calls.append(query)
        return []
    s=load_settings()
    r=discovery.discover_models(_spec(request='Find a 14B reasoning model, prefer speculative decoding'),s,tmp_path,
        scan=lambda *a:rows,lookup=lambda **kw:[],search=search)
    assert all(row['model_id'] in {'Qwen/model-13B','Qwen/model-14B'} for row in r['recommendations'])
    for model, coverage in r['research']['method_coverage'].items():
        assert all(coverage[m]['status'] != 'not_researched' for m in ['dspark','dflash','eagle'])
    r=discovery.discover_models(_spec(request='Find a 14B model',speculative_decoding='off'),s,tmp_path/'fallback',
        scan=lambda *a:rows[:2],lookup=lambda **kw:[],search=search)
    assert r['size_fallback']
    assert any('Size fallback' in t for t in r['recommendations'][0]['limitations'])


def test_new_request_supersedes_old_clarification_and_ignores_profile_task(tmp_path,monkeypatch):
    write_json(tmp_path/'conversation.json', {'workload':{'request':'old task','gpu_memory_gb':80},'active_job':'job_old'})
    write_json(tmp_path/'jobs/job_old.json', {'job_id':'job_old','status':'completed','arguments':{'request':'old task'},'result':{'status':'needs_input'}})
    adapter=ConversationAdapter(state_root=tmp_path)
    captured={}
    monkeypatch.setattr(adapter,'_launch_locked',lambda c,o,a:captured.update(a) or {})
    adapter.start_recommendation({'request':'New 14B reasoning task'})
    assert captured['request']=='New 14B reasoning task'


def test_worker_rejects_mismatched_result(tmp_path,monkeypatch):
    path=tmp_path/'jobs/job_test.json'
    write_json(path,{'job_id':'job_test','status':'starting','project_root':str(tmp_path),'timeout':10,
        'operation':'recommend_models','arguments':{'request':'new task'}})
    monkeypatch.setattr(OpenClawToolAdapter,'recommend_models',lambda *a:{'task_request':'old task','run_id':'run_old'})
    run_worker(path)
    assert json.loads(path.read_text())['status']=='failed'
