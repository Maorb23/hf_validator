import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { measuredCohorts, resolveMeasuredFilters, matchingMeasuredRows, measuredPareto } from '../src/components/measured-filters.mjs';
import { pareto, comparable } from '../src/components/analysis.mjs';

const load = async name => JSON.parse(await readFile(new URL(`../public/data/${name}`, import.meta.url)));
test('recorded sampler traces preserve trial budgets, bounds, and best-so-far values', async () => {
  const data = await load('simulations.json');
  assert.equal(data.kind, 'simulation');
  assert.equal(data.versions.optuna, '4.9.0');
  assert.equal(data.experiments.length, 4);
  for (const e of data.experiments) for (const [strategy, rows] of Object.entries(e.traces)) {
    assert.equal(rows.length, 24);
    let best = Infinity;
    rows.forEach((row, i) => {
      assert.equal(row.number, i);
      assert.ok(row.x >= 0 && row.x <= 1 && row.y >= 0 && row.y <= 1);
      assert.ok(Number.isFinite(row.value));
      best = Math.min(best, row.value);
      assert.equal(row.best, best, `${strategy} has incorrect best-so-far`);
    });
  }
});
test('Pareto calculation preserves ties and rejects dominated points', () => {
  const a={id:'a',latency:10,throughput:100}, b={id:'b',latency:20,throughput:90}, c={id:'c',latency:30,throughput:200}, d={...a,id:'d'};
  assert.deepEqual(pareto([a,b,c,d]).map(x=>x.id), ['a','c','d']);
  assert.equal(comparable({cohort:'short'},{cohort:'long'}),false);
});
test('inference examples are simulated, finite, and isolated by workload', async () => {
  const data = await load('inference-examples.json');
  assert.equal(data.kind,'simulation');assert.equal(data.rows.length,72);
  for (const row of data.rows) {
    assert.equal(row.kind,'simulation');
    assert.ok(row.latency>0 && row.throughput>0 && row.ttft>0);
    assert.ok(!('quality' in row));
  }
  const v=data.rows.filter(r=>r.workload==='short'&&r.engine==='vLLM');
  const s=data.rows.filter(r=>r.workload==='short'&&r.engine==='SGLang');
  assert.deepEqual(v.map(r=>r.throughput),s.map(r=>r.throughput),'synthetic labels must not imply engine rankings');
});
test('imported measurements retain every run and disclose incomplete metadata', async () => {
  const p=await load('publication.json');
  assert.equal(p.kind,'measurement');assert.equal(p.studies.length,190);assert.equal(p.rows.length,3302);
  assert.equal(p.studies.filter(s=>s.state==='crashed').length,10);
  assert.equal(p.rows.filter(r=>r.plotEligible).length,2492);
  assert.ok(p.studies.some(s=>s.missingMetadata.includes('revision')));
  assert.deepEqual([...new Set(p.studies.map(s=>s.context.model))].sort(),['Qwen/Qwen3-4B','Qwen/Qwen3-8B','Qwen/Qwen3.5-4B','google/gemma-4-12B-it','openbmb/MiniCPM5-2B']);
  for(const row of p.rows) {
    assert.ok(p.studies.some(s=>s.id===row.runId));
    if(row.plotEligible){assert.equal(row.state,'COMPLETE');assert.ok(row.throughput>0&&row.latency>0);}
  }
  assert.match(p.metricDefinitions.latency,/p95/);
  assert.doesNotMatch(JSON.stringify(p),/REPLACE_WITH|\/home\/ubuntu|dataset_path/);
});

test('engine filtering includes every matching data group and the actual TRT throughput winner', async () => {
  const {rows}=await load('publication.json');
  let selection={model:rows[0].context.model,engine:'all',sampler:'all',cohort:rows[0].cohort,runId:'all'};
  for(const engine of ['TensorRT-LLM','vLLM','SGLang']) {
    selection=resolveMeasuredFilters(rows,selection,{engine});
    assert.equal(selection.cohort,'all');
    const matching=matchingMeasuredRows(rows,selection);
    assert.equal(matching.length,rows.filter(r=>r.context.model===selection.model&&r.engine===engine).length);
    assert.ok(matching.some(r=>r.plotEligible));
    if(engine==='TensorRT-LLM'){
      const winner=matching.filter(r=>r.plotEligible).sort((a,b)=>b.throughput-a.throughput)[0];
      assert.equal(winner.id,'9y5kjcb9-13');
      assert.equal(winner.throughput,7911.50377256421);
      const group=measuredCohorts(rows,selection.model,engine)[0].cohort;
      const restricted=matchingMeasuredRows(rows,{...selection,cohort:group});
      assert.ok(restricted.length<matching.length);
      assert.ok(restricted.every(r=>r.cohort===group));
    }
  }
});
test('dependent filter changes clear hidden group restrictions and incompatible strategies', () => {
  const rows=[
    {cohort:'a',engine:'vLLM',sampler:'tpe',context:{model:'A'}},
    {cohort:'b',engine:'TensorRT-LLM',sampler:'gp',context:{model:'A'}},
    {cohort:'c',engine:'vLLM',sampler:'tpe',context:{model:'B'}},
  ];
  const selection={model:'A',engine:'all',sampler:'tpe',cohort:'a'};
  assert.equal(resolveMeasuredFilters(rows,selection,{engine:'vLLM'}).cohort,'all');
  assert.deepEqual(resolveMeasuredFilters(rows,selection,{engine:'TensorRT-LLM'}),{model:'A',engine:'TensorRT-LLM',sampler:'all',cohort:'all'});
  assert.deepEqual(resolveMeasuredFilters(rows,selection,{model:'B'}),{model:'B',engine:'all',sampler:'all',cohort:'all'});
  assert.equal(resolveMeasuredFilters(rows,selection,{cohort:'a'}).cohort,'a');
  assert.deepEqual(measuredCohorts(rows,'B','TensorRT-LLM'),[]);
});
test('multi-group browsing preserves Pareto points within each context', () => {
  const rows=[{id:'a',cohort:'A',latency:10,throughput:100},{id:'b',cohort:'A',latency:20,throughput:90},{id:'c',cohort:'B',latency:30,throughput:80}];
  assert.deepEqual(measuredPareto(rows).map(r=>r.id),['a','c']);
});


test('new variants expose all trials, preserve concurrency and exclude diagnostics from frontiers', async () => {
  const p=await load('publication.json');
  assert.equal(new Set(p.rows.map(r=>r.id)).size,p.rows.length);
  assert.equal(new Set(p.rows.map(r=>`${r.runId}:${r.number}`)).size,p.rows.length);
  assert.equal(p.sources.length,10);
  const studies=p.studies.filter(s=>s.search);
  assert.equal(studies.length,94);
  assert.equal(studies.filter(s=>s.durationSeconds<300).length,4);
  assert.ok(studies.every(s=>s.context.workload.max_concurrency===128));
  assert.equal(studies.filter(s=>s.metadataConflicts.some(c=>c.field==='model')).length,34);
  assert.equal(studies.filter(s=>s.metadataConflicts.some(c=>c.field==='speculativeMethod')).length,16);
  assert.ok(studies.every(s=>s.terminationReason===null));
  for(const [model,variants] of [['openbmb/MiniCPM5-2B',['Baseline']],['google/gemma-4-12B-it',['Baseline','DSpark']],['Qwen/Qwen3-4B',['Baseline','DSpark']],['Qwen/Qwen3-8B',['Baseline','DSpark']]]) {
    for(const variant of variants) {
      const rows=matchingMeasuredRows(p.rows,{model,variant,engine:'all',sampler:'all',cohort:'all',runId:'all'});
      assert.ok(rows.length>0,`${model} ${variant}`);
      assert.ok(rows.every(r=>r.context.variant===variant));
      assert.ok(rows.some(r=>r.plotEligible));
      assert.ok(measuredPareto(rows).every(r=>r.plotEligible));
      if(variant==='DSpark')assert.ok(rows.every(r=>r.context.draftModel&&r.context.draftRevision));
    }
  }
  const short=studies.filter(s=>s.durationSeconds>0&&s.durationSeconds<300);
  assert.equal(short.length,3);
  assert.ok(short.some(s=>p.rows.some(r=>r.runId===s.id&&r.plotEligible)));
});
