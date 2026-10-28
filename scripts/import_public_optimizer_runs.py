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


def convert_legacy(payload, source_hash):
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


def snapshot(value):
    match = re.search(r'models--([^/\\]+)[/\\]snapshots[/\\]([a-f0-9]{40})', str(value or ''))
    return {'model': match[1].replace('--', '/'), 'revision': match[2]} if match else None


def public_url(value):
    return value if isinstance(value, str) and re.fullmatch(r'https://(?:wandb\.ai|huggingface\.co)/[\w./-]+', value) else None


def sanitize(value):
    """Redact credentials and machine locations, including embedded log text."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if not any(s in k.lower() for s in
                ('path', 'secret', 'password', 'endpoint', 'api_key', 'authorization', 'command', 'cmd_args', 'access_token'))}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        identity = snapshot(value)
        if identity and '\n' not in value and ' ' not in value:
            return identity['model'] + '@' + identity['revision']
        if re.search(r'replace_with|placeholder|^unknown$|^none$|^null$', value, re.I):
            return None
        value = re.sub(r'(?i)(?:bearer\s+\S+|(?:hf_|sk-)[A-Za-z0-9_-]{8,}|(?:api[_-]?key|password|token|secret)\s*[=:]\s*\S+)', '[redacted]', value)
        return re.sub(r'https?://[^\s\"\'<>]+|[A-Za-z]:[\\/][^\s\"\'<>]+|(?<!\w)/(?:home|root|tmp|mnt|var|opt|usr|workspace|data)[^\s\"\'<>]*', '[redacted]', value)
    return clean(value)


def recorded_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def resolve_context(params, execution=None):
    env = dict(params.get('environment') or {})
    for key, value in (execution or {}).items():
        if key.startswith('execution.environment.') and value is not None:
            env[key.removeprefix('execution.environment.')] = value
    conflicts = []

    def conflict(field, values, resolved, reason):
        known = {k: v for k, v in values.items() if v is not None}
        if len({json.dumps(v, sort_keys=True) for v in known.values()}) > 1:
            conflicts.append({'field': field, 'recorded': sanitize(known), 'resolved': resolved, 'reason': reason})

    snap = snapshot((execution or {}).get('execution.model')) or snapshot(env.get('model'))
    tokenizer = snapshot(env.get('tokenizer'))
    models = {'hf_target_model_id': clean(env.get('hf_target_model_id')), 'snapshot': (snap or {}).get('model'),
              'tokenizerSnapshot': (tokenizer or {}).get('model'), 'target_model': clean(params.get('target_model')),
              'environment.target_model': clean(env.get('target_model'))}
    model = models['snapshot'] or models['hf_target_model_id'] or models['target_model'] or models['environment.target_model']
    conflict('model', models, model, 'Execution snapshot takes precedence, cross-checked against hf_target_model_id and tokenizer; copied target_model is lower confidence.')
    revisions = {'model_revision': clean(env.get('model_revision')), 'snapshot': (snap or {}).get('revision')}
    revision = revisions['snapshot'] or revisions['model_revision']
    conflict('revision', revisions, revision, 'Immutable execution snapshot takes precedence.')
    workload = sanitize(params.get('workload') or {})
    if workload.get('seed') is None:
        workload['seed'] = env.get('seed')
    concurrency = {'workload.max_concurrency': workload.get('max_concurrency'), 'config.concurrency': params.get('concurrency'),
                   'environment.concurrency': env.get('concurrency'),
                   'execution.client': (execution or {}).get('execution.config.client_args.max_concurrency'),
                   'execution.normalized': (execution or {}).get('execution.config.normalized_config.max_concurrency')}
    resolved = concurrency['execution.client'] or concurrency['execution.normalized'] or concurrency['workload.max_concurrency'] or concurrency['environment.concurrency'] or concurrency['config.concurrency']
    conflict('concurrency', concurrency, resolved, 'Execution client concurrency takes precedence, then normalized execution and workload. Server sequence limits are not client concurrency.')
    workload['max_concurrency'] = resolved
    fixed = env.get('fixed_server') or {}
    spec = fixed.get('speculative_config') or {}
    declared = env.get('speculative_decoding') or {}
    method = (execution or {}).get('execution.config.server_args.speculative_algorithm') or spec.get('method') or spec.get('decoding_type') or fixed.get('speculative_algorithm')
    execution_spec = (execution or {}).get('execution.config.server_args.speculative_config')
    if isinstance(execution_spec, dict):
        spec = execution_spec
        method = spec.get('method') or spec.get('decoding_type') or method
    draft_snap = snapshot(spec.get('model') or spec.get('speculative_model') or fixed.get('speculative_draft_model_path'))
    draft = (draft_snap or {}).get('model') or env.get('draft_model_id') or declared.get('draft_model_id') or params.get('draft_model')
    draft_revision = (draft_snap or {}).get('revision') or env.get('draft_revision') or declared.get('draft_revision')
    conflict('draftModel', {'snapshot': (draft_snap or {}).get('model'), 'draft_model_id': env.get('draft_model_id'), 'declared': declared.get('draft_model_id')}, draft, 'Execution draft snapshot takes precedence.')
    conflict('draftRevision', {'snapshot': (draft_snap or {}).get('revision'), 'draft_revision': env.get('draft_revision'), 'declared': declared.get('draft_revision')}, draft_revision, 'Execution draft snapshot takes precedence.')
    conflict('speculativeMethod', {'execution': str(method).lower() if method else None, 'declared': str(declared.get('algorithm')).lower() if declared.get('algorithm') else None}, method, 'Native fixed-server execution setting takes precedence over descriptive algorithm metadata.')
    variant = 'DSpark' if str(method).lower() == 'dspark' else (str(method) if method else 'Baseline' if not draft and not declared else 'Unreported')
    context = {'model': model, 'variant': variant,
               'variantBasis': 'Native speculative execution setting' if method else 'Inferred baseline: no draft or speculative settings recorded' if variant == 'Baseline' else 'Insufficient execution evidence',
               'draftModel': clean(draft), 'draftRevision': clean(draft_revision),
               'speculativeMethod': method, 'speculativeEvidence': sanitize({'fixedServer': fixed, 'declared': declared}),
               'identityEvidence': {**models, 'snapshotRevision': (snap or {}).get('revision'), 'tokenizerRevision': env.get('tokenizer_revision')},
               'concurrencyEvidence': concurrency, 'workload': workload,
               'requestsSha256': clean(env.get('requests_sha256')), 'workloadSha256': clean(env.get('workload_sha256')),
               'gpu': clean(env.get('gpu_type')), 'gpuCount': env.get('gpu_count'), 'revision': revision,
               'engineVersion': clean(env.get('framework_version')), 'cudaVersion': clean(env.get('cuda_version')),
               'driverVersion': clean(env.get('driver_version')), 'packageVersions': sanitize(env.get('package_versions')),
               'pythonVersion': env.get('python_version') or env.get('python'), 'platform': sanitize(env.get('platform'))}
    return sanitize(context), conflicts


def cohort_for(context, engine, source, run_id):
    keys = ('model', 'variant', 'draftModel', 'draftRevision', 'speculativeMethod', 'workload', 'requestsSha256', 'workloadSha256', 'gpu', 'gpuCount', 'revision', 'engineVersion', 'cudaVersion', 'driverVersion', 'packageVersions')
    identity = {k: context.get(k) for k in keys}
    identity.update(engine=engine, measurementSource=source, latencyStatistic='p95')
    incomplete_draft = context.get('draftModel') and not context.get('draftRevision')
    if incomplete_draft or any(context.get(k) is None for k in ('model', 'gpu', 'gpuCount', 'revision', 'engineVersion', 'requestsSha256')):
        identity['unverifiedStudy'] = run_id
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]


def convert(payload, source_hash):
    result = convert_legacy({'runs': [r for r in payload['runs'] if 'config' not in r]}, source_hash)
    for index, run in enumerate(payload['runs']):
        if 'config' not in run:
            continue
        params, summary = run.get('config') or {}, run.get('summary') or {}
        run_id = str(run.get('id') or f'{source_hash[:12]}-run-{index}')
        context, conflicts = resolve_context(params)
        summary_concurrency = summary.get('study.concurrency')
        if summary_concurrency is not None and summary_concurrency != context['workload']['max_concurrency']:
            conflicts.append({'field': 'concurrency', 'recorded': {'summary': summary_concurrency, 'workload': context['workload']['max_concurrency']}, 'resolved': context['workload']['max_concurrency'], 'reason': 'Execution/workload evidence takes precedence over summary.'})
        engine = ENGINES.get(params.get('framework'), clean(params.get('framework')) or 'Unreported')
        sampler_settings = sanitize(params.get('sampler_settings'))
        sampler = 'tpe_multivariate' if (sampler_settings or {}).get('multivariate') else clean(params.get('sampler'))
        study = {'id': run_id, 'name': sanitize(run.get('name')) or run_id, 'state': run.get('run_state', 'unknown'),
                 'engine': engine, 'sampler': sampler, 'context': context, 'metadataConflicts': conflicts,
                 'missingMetadata': [k for k in ('model', 'revision', 'gpu', 'gpuCount', 'engineVersion', 'requestsSha256') if context.get(k) is None],
                 'draftModelStatus': 'recorded' if context['draftModel'] else 'not reported', 'sourceSha256': source_hash,
                 'sourceUrl': public_url(run.get('url')), 'createdAt': run.get('created_at'),
                 'durationSeconds': recorded_number(summary.get('_runtime')), 'durationSource': '_runtime' if recorded_number(summary.get('_runtime')) is not None else None,
                 'terminationReason': sanitize(run.get('termination_reason') or summary.get('termination_reason') or summary.get('study.termination_reason')),
                 'interrupted': summary.get('interrupted'), 'trialCount': 0,
                 'search': sanitize({k: v for k, v in params.items() if k not in ('environment', 'target_model', 'workload')}),
                 'timings': sanitize({k: v for k, v in summary.items() if k in ('study.allocated_gpu_seconds', 'study.host_search_and_storage_seconds', 'cost/cumulative_gpu_seconds')})}
        seen = {}
        for position, trial in enumerate(run.get('trials') or []):
            number = trial.get('number', position)
            trial_id = f'{run_id}-{number}'
            if trial_id in seen:
                if seen[trial_id] != trial:
                    raise ValueError(f'Conflicting duplicate trial {trial_id}')
                continue
            seen[trial_id] = trial
            metrics, metadata = trial.get('metrics') or {}, trial.get('trial_metadata') or {}
            tc, trial_conflicts = resolve_context(params, metrics)
            config = trial.get('effective_config') or {k.removeprefix('config.'): v for k, v in metadata.items() if k.startswith('config.')}
            if not config:
                config = trial.get('params') or {}
            state = str(trial.get('state') or 'UNKNOWN').upper()
            latency, throughput = positive(metrics.get('p95_e2e_latency_ms')), positive(metrics.get('output_throughput'))
            source = metrics.get('source')
            failed, completed = metrics.get('failed'), metrics.get('completed')
            expected, total = tc['workload'].get('num_prompts'), metrics.get('total_requests')
            reasons = []
            if state != 'COMPLETE': reasons.append('Trial state is ' + state)
            if not str(source or '').startswith('MEASURED'): reasons.append('No measured source')
            if latency is None or throughput is None: reasons.append('Missing positive finite p95 E2E latency or throughput')
            if failed not in (None, 0): reasons.append('Failed requests')
            if completed is not None and any(isinstance(n, (int, float)) and completed < n for n in (expected, total)):
                reasons.append('Incomplete request measurements')
            if metrics.get('execution.status') in ('failed', 'incomplete', 'interrupted'):
                reasons.append('Incomplete execution')
            result['rows'].append({'id': trial_id, 'runId': run_id, 'number': number, 'state': state, 'kind': 'measurement',
                'engine': engine, 'sampler': sampler, 'context': tc, 'metadataConflicts': trial_conflicts,
                'cohort': cohort_for(tc, engine, source, run_id), 'configuration': sanitize(config), 'parameters': sanitize(trial.get('params')),
                'latency': latency, 'throughput': throughput, 'ttft': positive(metrics.get('p95_ttft_ms')),
                'meanLatency': positive(metrics.get('mean_e2e_latency_ms')), 'meanTtft': positive(metrics.get('mean_ttft_ms')),
                'measurementSource': source, 'completed': completed, 'failed': failed, 'sloFeasible': metadata.get('slo_feasible'),
                'plotEligible': not reasons, 'diagnosticReasons': reasons, 'failureReason': sanitize(trial.get('failure_reason')),
                'objectives': sanitize(trial.get('objectives')), 'requestAccounting': sanitize(trial.get('request_accounting')),
                'trialMetadata': sanitize(metadata), 'metrics': sanitize(metrics), 'cost': sanitize(trial.get('cost'))})
        study['trialCount'] = len(seen)
        result['studies'].append(study)
    return result


def merge_results(existing, additions):
    """Merge stable run/trial IDs, enriching overlap without dropping history."""
    result = dict(existing or additions[0])
    studies = {s['id']: s for s in (existing or {}).get('studies', [])}
    rows = {(r['runId'], r['number']): r for r in (existing or {}).get('rows', [])}
    sources = {s['sha256']: s for s in (existing or {}).get('sources', [])}
    if existing and existing.get('sourceSha256') and not sources:
        sources[existing['sourceSha256']] = {'file': 'Previously published optimizer export', 'sha256': existing['sourceSha256'], 'studyCount': len(studies), 'trialCount': len(rows)}
    duplicates = {'studies': 0, 'trials': 0}
    for addition in additions:
        for study in addition['studies']:
            prior = studies.get(study['id'], {})
            study = {**study, 'sourceHashes': sorted(set(prior.get('sourceHashes', []) + [h for h in (prior.get('sourceSha256'), study.get('sourceSha256')) if h]))}
            if study['id'] in studies:
                duplicates['studies'] += 1
            studies[study['id']] = {**studies.get(study['id'], {}), **study}
        for row in addition['rows']:
            key_id = (row['runId'], row['number'])
            if key_id in rows:
                duplicates['trials'] += 1
                old = rows[key_id]
                for key in ('number', 'latency', 'throughput', 'ttft'):
                    if old.get(key) is not None and row.get(key) is not None and old[key] != row[key]:
                        raise ValueError(f'Conflicting duplicate trial {row["id"]}: {key}')
                row = {**row, 'id': old['id']}
            rows[key_id] = {**rows.get(key_id, {}), **row}
        for source in addition.get('sources', []):
            sources[source['sha256']] = source
    result.update(studies=list(studies.values()), rows=list(rows.values()), sources=list(sources.values()), schemaVersion=2)
    for row in result['rows']:
        row['cohort'] = cohort_for(row['context'], row['engine'], row.get('measurementSource'), row['runId'])
    if len({r['id'] for r in result['rows']}) != len(result['rows']):
        raise ValueError('Distinct trial numbers produced colliding row IDs')
    for study in result['studies']:
        study['trialCount'] = sum(r['runId'] == study['id'] for r in result['rows'])
    result['importSummary'] = {'overlappingStudies': duplicates['studies'], 'overlappingTrials': duplicates['trials']}
    result['note'] = additions[0]['note'] + ' No study-duration cutoff is applied. Native execution and immutable snapshots take precedence over copied metadata; conflicts are disclosed. Unreported termination reasons remain unavailable.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--merge', type=Path, help='Existing sanitized publication to preserve')
    args = parser.parse_args()
    additions = []
    for path in args.input:
        if path.suffix != '.json':
            raise ValueError('Only explicitly selected .json exports are accepted; backups are ignored')
        raw = path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        result = convert(json.loads(raw), source_hash)
        result['sources'] = [{'file': path.name, 'sha256': source_hash, 'studyCount': len(result['studies']), 'trialCount': len(result['rows'])}]
        additions.append(result)
    existing = json.loads(args.merge.read_text()) if args.merge else None
    result = merge_results(existing, additions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'runs': len(result['studies']), 'trials': len(result['rows']), 'plottable': sum(r['plotEligible'] for r in result['rows']), **result['importSummary']}))


if __name__ == '__main__':
    main()
