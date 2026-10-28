"""Audit source counts, checksums, trial metrics and preservation independently of conversion."""
import json,pathlib,hashlib,collections,math,re
import argparse
parser=argparse.ArgumentParser(description='Independently check selected source exports and prior published measurements; no GPU execution.')
parser.add_argument('--publication', type=pathlib.Path, required=True)
parser.add_argument('--before', type=pathlib.Path, required=True)
parser.add_argument('--source-dir', type=pathlib.Path, required=True)
parser.add_argument('--report', type=pathlib.Path, required=True)
args=parser.parse_args()
pub=json.loads(args.publication.read_text()); before=json.loads(args.before.read_text()); folder=args.source_dir
rows={(r['runId'],r['number']):r for r in pub['rows']}; studies={s['id']:s for s in pub['studies']};assert len(rows)==len(pub['rows']);assert len(set(r['id'] for r in pub['rows']))==len(rows)
report=[];runids=set();oldkeys={(r['runId'],r['number']) for r in before['rows']}
for source in pub['sources']:
 if not source['file'].endswith('.json'):continue
 raw=(folder/source['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256'];runs=json.loads(raw)['runs'];count=valid=short=0;states=collections.Counter()
 for run in runs:
  runids.add(run['id']);study=studies[run['id']];c=run['config'];e=c['environment'];assert study['state']==run['run_state'];assert study['durationSeconds']==run['summary'].get('_runtime');assert study['sourceSha256']==source['sha256'];short+=study['durationSeconds'] is not None and study['durationSeconds']<300
  assert study['context']['model']==e['hf_target_model_id'];assert study['context']['revision']==e['model_revision'];assert study['context']['workload']['max_concurrency']==128
  for trial in run['trials']:
   count+=1;r=rows[(run['id'],trial['number'])];states[trial['state']]+=1;valid+=r['plotEligible'];assert r['context']['model']==e['hf_target_model_id'];assert r['context']['workload']['max_concurrency']==128;assert r['state']==trial['state']
   for key,rawkey in [('latency','p95_e2e_latency_ms'),('ttft','p95_ttft_ms'),('throughput','output_throughput')]:
    v=trial['metrics'].get(rawkey);v=v if isinstance(v,(int,float)) and math.isfinite(v) and v>0 else None;assert r[key]==v
   if r['plotEligible']:assert r['state']=='COMPLETE' and r['latency']>0 and r['throughput']>0
  assert study['trialCount']==len(run['trials'])
 report.append({'file':source['file'],'studies':len(runs),'trials':count,'validMeasurements':valid,'diagnostics':count-valid,'underFiveMinutes':short,'states':dict(states)})
for old in before['rows']:
 r=rows[(old['runId'],old['number'])]
 for key in ['id','number','state','latency','throughput','ttft','completed','failed','plotEligible']:
  assert old.get(key)==r.get(key),(old['id'],key)
assert all(s['id'] in studies for s in before['studies'])
new=[studies[i] for i in sorted(runids)];short=[s for s in new if s['durationSeconds'] is not None and s['durationSeconds']<300]
s=json.dumps(pub);assert not re.search(r'/home/|/root/|/tmp/|REPLACE_WITH|api_key|Bearer\s|hf_[A-Za-z0-9]{10}',s)
output={'sources':report,'totalStudies':len(studies),'totalTrials':len(rows),'validMeasurements':sum(r['plotEligible'] for r in rows.values()),'diagnostics':sum(not r['plotEligible'] for r in rows.values()),'preservedStudies':len(before['studies']),'preservedTrials':len(before['rows']),'overlappingStudies':len(runids & {s['id'] for s in before['studies']}),'overlappingTrials':sum((r['runId'],r['number']) in oldkeys for r in pub['rows'] if r['runId'] in runids),'conflicts':dict(collections.Counter(c['field'] for st in new for c in st['metadataConflicts'])),'shortStudies':[{'id':st['id'],'seconds':st['durationSeconds'],'validMeasurements':sum(r['plotEligible'] for r in pub['rows'] if r['runId']==st['id'])} for st in short],'allSelectedConcurrency':128,'unchangedSourceHashes':True,'originalMeasurementsPreserved':True,'missingTerminationReasons':sum(s['terminationReason'] is None for s in new)}
args.report.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
