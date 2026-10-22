"""Import selected optimizer exports. Retain runs with incomplete provenance.

Only inference metrics and sanitized context are published. No jobs are launched.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ENGINES = {'vllm': 'vLLM', 'sglang': 'SGLang', 'trtllm': 'TensorRT-LLM'}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if not any(s in k.lower() for s in ('path', 'secret', 'password', 'endpoint', 'api_key'))}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        if re.search(r'replace_with|placeholder|^unknown$|^none$|^null$', value, re.I):
            return None
        if value.startswith(('/', '\\', '~')) or re.search(r'[A-Za-z]:[\\/]|https?://', value):
            return None
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def positive(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 else None


def convert(payload, source_hash):
    studies, rows = [], []
    for index, run in enumerate(payload['runs']):
        params = run.get('params') or {}
        env = params.get('environment') or {}
        workload = clean(params.get('workload') or {})
        if workload.get('seed') is None:
            workload['seed'] = env.get('seed')
        model = clean(params.get('target_model') or env.get('target_model') or env.get('model'))
        draft = clean(params.get('draft_model') or env.get('draft_model'))
        run_id = str(run.get('id') or f'run-{index}')
        context = {'model': model, 'draftModel': draft, 'workload': workload,
                   'requestsSha256': clean(env.get('requests_sha256')), 'workloadSha256': clean(env.get('workload_sha256')),
                   'gpu': clean(env.get('gpu_type')), 'gpuCount': env.get('gpu_count'),
                   'revision': clean(env.get('model_revision')), 'engineVersion': clean(env.get('framework_version')),
                   'cudaVersion': clean(env.get('cuda_version')), 'driverVersion': clean(env.get('driver_version'))}
        missing = [key for key, value in context.items() if value is None and key != 'draftModel']
        for key in ('seed', 'num_prompts', 'random_input_len', 'random_output_len', 'max_concurrency'):
            if workload.get(key) is None:
                missing.append('workload.' + key)
        engine = ENGINES.get(params.get('framework'), clean(params.get('framework')) or 'Unreported')
        study = {'id': run_id, 'name': clean(run.get('name')) or run_id,
                 'state': run.get('state', 'unknown'), 'engine': engine,
                 'sampler': clean(params.get('sampler')), 'context': context,
                 'missingMetadata': sorted(set(missing)), 'draftModelStatus': 'recorded' if draft else 'not reported',
                 'sourceSha256': source_hash, 'trialCount': len(run.get('trial_history') or [])}
        for position, trial in enumerate(run.get('trial_history') or []):
            config = clean({k[len('trial/config.'):]: v for k, v in trial.items() if k.startswith('trial/config.')})
            trial_draft = config.get('draft_model') or config.get('speculative_model') or draft
            trial_context = {**context, 'draftModel': trial_draft}
            identity = {k: trial_context[k] for k in ('model', 'draftModel', 'workload', 'requestsSha256', 'workloadSha256', 'gpu', 'gpuCount', 'revision')}
            cohort = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            state = trial.get('trial/state', 'UNKNOWN')
            latency = positive(trial.get('benchmark/p95_e2e_latency_ms'))
            throughput = positive(trial.get('benchmark/output_throughput'))
            measured = str(trial.get('benchmark/source', '')).startswith('MEASURED')
            failed = trial.get('benchmark/failed')
            eligible = state == 'COMPLETE' and measured and latency is not None and throughput is not None and failed in (None, 0)
            rows.append({'id': f'{run_id}-{position}', 'runId': run_id, 'number': trial.get('trial/number', position),
                         'state': state, 'kind': 'measurement', 'engine': engine, 'sampler': study['sampler'],
                         'context': trial_context, 'cohort': cohort, 'configuration': config,
                         'latency': latency, 'throughput': throughput, 'ttft': positive(trial.get('benchmark/p95_ttft_ms')),
                         'measurementSource': clean(trial.get('benchmark/source')), 'completed': trial.get('benchmark/completed'), 'failed': failed,
                         'sloFeasible': trial.get('trial/slo_feasible'), 'plotEligible': eligible})
        studies.append(study)
    return {'schemaVersion': 1, 'kind': 'measurement', 'sourceSha256': source_hash,
            'status': 'Imported inference measurements with disclosed provenance gaps',
            'note': 'Missing metadata does not exclude runs. Draft model not reported does not establish that none was used. Replay follows exported record order. Results are not independently validated.',
            'metricDefinitions': {'latency': 'p95 E2E latency · ms', 'throughput': 'Output throughput · tok/s', 'ttft': 'p95 TTFT · ms'},
            'studies': studies, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = convert(json.loads(raw), hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'runs': len(result['studies']), 'trials': len(result['rows']), 'plottable': sum(r['plotEligible'] for r in result['rows'])}))


if __name__ == '__main__':
    main()
