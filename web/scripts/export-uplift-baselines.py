"""Export public baseline measurements without local paths or provider credentials."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
args = parser.parse_args()
raw = args.source.read_bytes()
source = json.loads(raw)
rows = []
for run in source['runs']:
    config, summary = run['config'], run['summary']
    if run['run_state'] != 'finished' or summary.get('benchmark_valid') is not True:
        continue
    env = config.get('environment', {})
    rows.append({
        'model': config['model'], 'runId': run['id'], 'url': run['url'],
        'framework': config['framework'], 'workload': config['workload'],
        'gpu': env.get('gpu_type'), 'gpuCount': env.get('gpu_count'),
        'revision': env.get('model_revision'),
        'throughput': summary.get('output_throughput'),
        'latency': summary.get('p95_e2e_latency_ms'),
        'ttft': summary.get('p95_ttft_ms'),
    })
output = Path(__file__).resolve().parents[1] / 'public/data/uplift-baselines.json'
output.write_text(json.dumps({'source': args.source.name,
    'sourceSha256': hashlib.sha256(raw).hexdigest(), 'rows': rows}, indent=2) + '\n')
