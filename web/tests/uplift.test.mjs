import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {objectiveUplifts,upliftGroups} from '../src/components/uplift.mjs';
const workload={dataset_name:'random',seed:2026,num_prompts:1000,random_input_len:512,random_output_len:128,random_range_ratio:1,max_concurrency:128,request_rate:'inf'};
const baseline={model:'model',workload,throughput:100,latency:20,ttft:10};
const row=(id,engine,throughput,latency,ttft,extra={})=>({id,engine,plotEligible:true,context:{model:'model',workload},throughput,latency,ttft,...extra});
test('independent best metrics across all frameworks use measured baseline',()=>{
 const rows=[row('a','vLLM',150,8,6),row('b','SGLang',200,12,2),row('c','TensorRT-LLM',180,10,4)];
 const result=objectiveUplifts(rows,[baseline]);
 assert.deepEqual(result.map(r=>r.best.id),['b','a','b']);
 assert.deepEqual(result.map(r=>r.percent),[100,60,80]);
 assert.deepEqual(result.map(r=>r.reference),[100,20,10]);
 // Extra losing trials cannot change the denominator or best values.
 assert.deepEqual(objectiveUplifts([...rows,row('d','vLLM',1,100,100)],[baseline]).map(r=>r.percent),[100,60,80]);
});
test('models, failed trials and mismatched workloads cannot enter a comparison',()=>{
 const rows=[row('a','vLLM',150,10,5),row('failed','SGLang',999,1,1,{plotEligible:false}),
 row('other workload','SGLang',999,1,1,{context:{model:'model',workload:{...workload,num_prompts:896}}})];
 assert.deepEqual(objectiveUplifts(rows,[baseline]).map(r=>r.best.id),['a','a','a']);
 assert.equal(upliftGroups([...rows,row('other','vLLM',1,1,1,{context:{model:'other',workload}})]).length,2);
 assert.deepEqual(objectiveUplifts([...rows,row('other','vLLM',1,1,1,{context:{model:'other',workload}})],[baseline]),[]);
});
test('missing, ambiguous, invalid and zero baselines never fall back to trial averages',()=>{
 const rows=[row('a','vLLM',100,10,2)];
 for(const references of [[],[baseline,baseline],[{...baseline,throughput:0,latency:null,ttft:NaN}]])
  assert.ok(objectiveUplifts(rows,references).every(s=>s.percent===null));
});
test('regressions retain negative percentages and one trial is sufficient',()=>{
 assert.deepEqual(objectiveUplifts([row('a','vLLM',50,30,15)],[baseline]).map(r=>r.percent),[-50,-50,-50]);
});
test('published five-model comparisons match independently computed extrema',()=>{
 const publication=JSON.parse(readFileSync(new URL('../public/data/publication.json',import.meta.url)));
 const baselines=JSON.parse(readFileSync(new URL('../public/data/uplift-baselines.json',import.meta.url))).rows;
 assert.equal(baselines.length,5);
 for(const baseline of baselines){
  const rows=publication.rows.filter(r=>r.context.model===baseline.model);
  const candidates=rows.filter(r=>r.plotEligible&&Object.keys(workload).every(k=>r.context.workload[k]===baseline.workload[k]));
  const summaries=objectiveUplifts(rows,baselines);
  for(const s of summaries){
   assert.ok(s.best,baseline.model);
   const values=candidates.map(r=>r[s.key]).filter(v=>Number.isFinite(v)&&v>=0);
   assert.equal(s.best[s.key],s.direction==='maximize'?Math.max(...values):Math.min(...values));
   assert.equal(s.reference,baseline[s.key]);
  }
 }
});
