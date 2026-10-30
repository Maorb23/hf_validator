import UpliftSummary from './UpliftSummary.jsx';
import React, {useEffect, useMemo, useState} from 'react';
const engines=[['vLLM','#a7f3cf'],['SGLang','#81b9ff'],['TensorRT-LLM','#d5adf6']];
const fmt=n=>Number(n).toLocaleString('en',{maximumFractionDigits:0});
export default function BenchmarkPlot({rows,upliftRows=rows,upliftSummaries,initialModel,animated=false}) {
 const [engine,setEngine]=useState('all'),[metric,setMetric]=useState('latency'),[pareto,setPareto]=useState(false),[selected,setSelected]=useState(null),[count,setCount]=useState(rows.length),[playing,setPlaying]=useState(false);
 useEffect(()=>{if(animated&&!window.matchMedia('(prefers-reduced-motion: reduce)').matches){setCount(0);setPlaying(true);}},[animated]);
 useEffect(()=>{if(!playing)return; const timer=setInterval(()=>setCount(n=>{if(n>=rows.length){setPlaying(false);return rows.length;}return Math.min(rows.length,n+12);}),80);return()=>clearInterval(timer);},[playing,rows.length]);
 const filtered=rows.filter(r=>engine==='all'||r.engine===engine), visible=filtered.filter(r=>rows.indexOf(r)<count&&Number.isFinite(r[metric]));
 const fronts=useMemo(()=>engines.flatMap(([name])=>{const group=visible.filter(r=>r.engine===name);return group.filter(p=>!group.some(q=>q.cohort===p.cohort&&q[metric]<=p[metric]&&q.throughput>=p.throughput&&(q[metric]<p[metric]||q.throughput>p.throughput)));}),[rows,count,engine,metric]);
 const ids=new Set(fronts.map(r=>r.id)),maxX=Math.max(1,...filtered.map(r=>r[metric]||0))*1.08,maxY=Math.max(1,...filtered.map(r=>r.throughput))*1.08;
 const active=visible.find(r=>r.id===selected),x=r=>76+r[metric]/maxX*458,y=r=>205-r.throughput/maxY*167;
 return <div className="interactive-benchmark">
 <div className="benchmark-controls"><label>Latency metric<select value={metric} onChange={e=>{setMetric(e.target.value);setSelected(null);}}><option value="latency">p95 E2E latency</option><option value="ttft">p95 time to first token</option></select></label><button className="benchmark-toggle" aria-pressed={pareto} onClick={()=>setPareto(!pareto)}>Pareto frontiers</button></div>
 <UpliftSummary rows={upliftRows} preparedGroups={upliftSummaries} initialModel={initialModel} compact/>
 <svg viewBox="0 0 560 265" role="group" aria-label="Interactive recorded throughput and latency trials">
 {[0,.25,.5,.75,1].map(t=><g key={t}><line x1="76" y1={205-t*167} x2="534" y2={205-t*167} className="plot-grid"/><text x="65" y={209-t*167} textAnchor="end" className="chart-tick">{fmt(t*maxY)}</text><text x={76+t*458} y="226" textAnchor="middle" className="chart-tick">{fmt(t*maxX)}</text></g>)}
 <path d="M76 38V205H534" className="plot-axis" fill="none"/>
 {engines.map(([name,color])=><g key={name}>{pareto&&<path className="benchmark-frontier" d={[...new Set(fronts.filter(r=>r.engine===name).map(r=>r.cohort))].map(cohort=>fronts.filter(r=>r.engine===name&&r.cohort===cohort).sort((a,b)=>a[metric]-b[metric]).map((r,i)=>`${i?'L':'M'}${x(r)},${y(r)}`).join(' ')).join(' ')} stroke={color} fill="none" strokeWidth="2"/>}{visible.filter(r=>r.engine===name).map(r=><circle key={r.id} cx={x(r)} cy={y(r)} r={selected===r.id?7:pareto&&ids.has(r.id)?4.5:3} fill={color} opacity={pareto&&!ids.has(r.id)?.12:.65} onMouseEnter={()=>setSelected(r.id)} onClick={()=>setSelected(r.id)}><title>{`${name}: ${fmt(r.throughput)} tokens/s · ${fmt(r[metric])} ms`}</title></circle>)}</g>)}
 <text x="305" y="255" textAnchor="middle" className="plot-label">{metric==='latency'?'p95 E2E latency':'p95 TTFT'} · ms</text><text transform="translate(17 125) rotate(-90)" textAnchor="middle" className="plot-label">Output throughput · tokens/s</text></svg>
 <div className="benchmark-engines" aria-label="Filter framework"><button aria-pressed={engine==='all'} onClick={()=>setEngine('all')}>All</button>{engines.map(([name,color])=><button key={name} aria-pressed={engine===name} onClick={()=>{setEngine(name);setSelected(null);}}><i style={{background:color}}/>{name}</button>)}</div>
 <div className="benchmark-readout" aria-live="polite">{active?<><strong>{active.engine}</strong><span>{fmt(active.throughput)} tok/s</span><span>{fmt(active[metric])} ms</span></>:<span>{visible.length} trials · hover or tap a point to inspect</span>}</div>
 <div className="benchmark-playback"><button onClick={()=>{if(count>=rows.length)setCount(0);setPlaying(!playing);}}>{playing?'Pause animation':count>=rows.length?'Replay trials':'Play trials'}</button><label>Trials<input aria-label="Visible benchmark trials" type="range" min="0" max={rows.length} value={count} onChange={e=>{setPlaying(false);setCount(Number(e.target.value));}}/></label><output>{count} / {rows.length}</output></div>
 {pareto&&<p className="benchmark-note">Frontiers are calculated within each framework and comparable data group. Missing hardware, revision, and runtime metadata prevent a general engine ranking.</p>}
 </div>;
}
