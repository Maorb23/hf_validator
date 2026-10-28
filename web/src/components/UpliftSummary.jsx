import React, {useId, useState} from 'react';
import baselines from '../../public/data/uplift-baselines.json';
import {objectiveUplifts, upliftGroups} from './uplift.mjs';
const fmt=n=>n.toLocaleString('en',{maximumFractionDigits:1});
export default function UpliftSummary({rows,onInspect,compact=false,initialModel='Qwen/Qwen3.5-4B'}){
 const [model,setModel]=useState(initialModel),id=useId();
 const groups=upliftGroups(rows),group=groups.find(g=>g.model===model)||groups[0];
 if(!group)return null;
 const summaries=objectiveUplifts(group.rows,baselines.rows),baseline=summaries[0]?.baseline;
 return <section className={`uplift-summary ${compact?'compact':''}`} aria-label="Best results versus measured baseline">
 <h3>Best results vs. model baseline</h3>
 {groups.length>1?<div className="uplift-group"><label htmlFor={id}>Uplift model</label><select id={id} value={group.model} onChange={e=>setModel(e.target.value)}>{groups.map(g=><option key={g.key} value={g.model}>{g.model}</option>)}</select></div>:<p className="small muted">{group.model}</p>}
 {baseline&&<p className="small muted">{baseline.workload.random_input_len} input / {baseline.workload.random_output_len} output tokens · {baseline.workload.num_prompts} prompts · concurrency {baseline.workload.max_concurrency} · seed {baseline.workload.seed} · <a href={baseline.url}>Measured {baseline.framework} baseline ({baseline.runId})</a></p>}
 <div className="uplift-metrics">{summaries.map(s=><div className="uplift-metric" key={s.key}><span>{s.label}</span><strong>{s.percent===null?'Unavailable':`${fmt(s.percent)}%`}</strong><span>{s.direction==='maximize'?'throughput uplift':'latency reduction'}</span>{s.best&&s.reference!==null&&<small>Best {fmt(s.best[s.key])} ({s.best.engine}) vs. baseline {fmt(s.reference)} {s.unit}</small>}{s.best&&onInspect&&<button className="table-link" onClick={()=>onInspect(s.best.id)}>Inspect best configuration</button>}</div>)}</div>
 <p className="uplift-note">Each metric selects its own best completed result across all frameworks for this model, with matching recorded workload settings. Chart filters and replay do not change these model-wide results. Uplift uses the measured baseline from {baselines.source}, never an average of trials. Negative percentages indicate a regression. Missing revision, hardware, runtime or chat-template metadata remain unverified.</p>
 </section>;
}
