// Independent metric winners across frameworks, relative to measured model baselines.
export const upliftMetrics = [
  {key:'throughput',label:'Output throughput',unit:'tok/s',direction:'maximize'},
  {key:'latency',label:'p95 end-to-end latency',unit:'ms',direction:'minimize'},
  {key:'ttft',label:'p95 time to first token',unit:'ms',direction:'minimize'},
];
const workloadKeys=['dataset_name','seed','num_prompts','random_input_len','random_output_len','random_range_ratio','max_concurrency','request_rate'];
export function matchesBaseline(row, baseline) {
  const c=row.context,w=c?.workload,b=baseline.workload;
  return row.plotEligible!==false && c?.model===baseline.model && w && b &&
    workloadKeys.every(k=>w[k]!==undefined && w[k]===b[k]) &&
    (!c.revision || !baseline.revision || c.revision===baseline.revision) &&
    (!c.gpuCount || !baseline.gpuCount || c.gpuCount===baseline.gpuCount);
}
export function upliftGroups(rows) {
  const models=[...new Set(rows.map(r=>r.context?.model).filter(Boolean))].sort();
  return models.map(model=>({key:model,model,rows:rows.filter(r=>r.context?.model===model)}));
}
export function objectiveUplifts(rows, baselines=[]) {
  const groups=upliftGroups(rows);
  if(groups.length!==1)return [];
  const matches=baselines.filter(b=>b.model===groups[0].model);
  // Never silently choose a denominator when the source has multiple baselines.
  const baseline=matches.length===1?matches[0]:null;
  const candidates=baseline?rows.filter(r=>matchesBaseline(r,baseline)):[];
  return upliftMetrics.map(spec=>{
    const valid=candidates.filter(r=>Number.isFinite(r[spec.key])&&r[spec.key]>=0);
    const reference=baseline?.[spec.key];
    const best=valid.reduce((a,b)=>!a||(spec.direction==='maximize'?b[spec.key]>a[spec.key]:b[spec.key]<a[spec.key])?b:a,null);
    const delta=best?(spec.direction==='maximize'?best[spec.key]-reference:reference-best[spec.key]):null;
    return {...spec,count:valid.length,reference:Number.isFinite(reference)?reference:null,baseline,best,
      percent:best&&Number.isFinite(reference)&&reference>0?delta/reference*100:null};
  });
}
