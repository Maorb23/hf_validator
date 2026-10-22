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
  assert.equal(p.kind,'measurement');assert.equal(p.studies.length,114);assert.equal(p.rows.length,1948);
  assert.equal(p.studies.filter(s=>s.state==='crashed').length,6);
  assert.equal(p.rows.filter(r=>r.plotEligible).length,1524);
  assert.ok(p.studies.some(s=>s.missingMetadata.includes('revision')));
  assert.deepEqual([...new Set(p.studies.map(s=>s.context.model))].sort(),['Qwen/Qwen3-4B','Qwen/Qwen3.5-4B']);
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
