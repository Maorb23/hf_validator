import React, {useEffect, useMemo, useState} from 'react';
const metrics=[['throughput','Output throughput','tok/s','Higher is better'],['latency','p95 end-to-end latency','ms','Lower is better'],['ttft','p95 time to first token','ms','Lower is better']];
const fmt=n=>Number(n).toLocaleString('en',{maximumFractionDigits:0});
export default function MetricStory({rows}) {
 const [engine,setEngine]=useState('vLLM'),[frontOnly,setFrontOnly]=useState(false),[index,setIndex]=useState(0),[playing,setPlaying]=useState(false);
 useEffect(()=>{setPlaying(!window.matchMedia('(prefers-reduced-motion: reduce)').matches);},[]);
 // Every comparison stays inside one recorded workload/hardware cohort.
 const cohort=rows.find(r=>r.engine===engine)?.cohort;
 const group=useMemo(()=>rows.filter(r=>r.engine===engine&&r.cohort===cohort&&metrics.every(([key])=>Number.isFinite(r[key]))).sort((a,b)=>a.latency-b.latency),[rows,engine,cohort]);
 const frontier=useMemo(()=>group.filter(p=>!group.some(q=>q.latency<=p.latency&&q.throughput>=p.throughput&&(q.latency<p.latency||q.throughput>p.throughput))),[group]);
 const pool=frontOnly?frontier:group;
 const samples=useMemo(()=>Array.from(new Set(Array.from({length:Math.min(6,pool.length)},(_,i)=>pool[Math.round(i*(pool.length-1)/Math.max(1,Math.min(6,pool.length)-1))]))),[pool]);
 useEffect(()=>{if(!playing||samples.length<2)return;const timer=setInterval(()=>setIndex(i=>(i+1)%samples.length),3200);return()=>clearInterval(timer);},[playing,samples.length]);
 const row=samples[index%samples.length];
 if(!row)return <p className="benchmark-note">No complete metric records in this group.</p>;
 return <div className="metric-story">
 <div className="metric-story-heading"><h3>Configuration matters.</h3><button aria-label={playing?'Pause metric animation':'Play metric animation'} onClick={()=>setPlaying(!playing)}>{playing?'Pause':'Play'}</button></div>
 <div className="metric-story-controls"><label>Framework<select value={engine} onChange={e=>{setEngine(e.target.value);setIndex(0);}}>{['vLLM','SGLang','TensorRT-LLM'].map(name=><option key={name}>{name}</option>)}</select></label><button aria-pressed={frontOnly} onClick={()=>{setFrontOnly(!frontOnly);setIndex(0);}}>Pareto only</button></div>
 <div className="metric-story-meters">{metrics.map(([key,label,unit,direction])=><div className="metric-story-meter" key={key}><div><span>{label}</span><small>{direction}</small></div><strong key={row.id}>{fmt(row[key])}<small>{unit}</small></strong><div className="metric-story-track" role="meter" aria-label={label} aria-valuenow={row[key]} aria-valuemin={0} aria-valuemax={Math.max(...group.map(r=>r[key]))}><i style={{width:`${row[key]/Math.max(...group.map(r=>r[key]))*100}%`}}/></div></div>)}</div>
 <div className="metric-story-status"><span className={frontier.includes(row)?'on-frontier':''}>{frontier.includes(row)?'On the Pareto frontier':'A different configuration improves this tradeoff'}</span><code>{row.id}</code></div>
 <div className="metric-story-trials" aria-label="Recorded configurations">{samples.map((r,i)=><button key={r.id} aria-pressed={row.id===r.id} aria-label={`Inspect configuration ${i+1}`} onClick={()=>{setIndex(i);setPlaying(false);}}><span>{String(i+1).padStart(2,'0')}</span><i style={{height:`${20+r.throughput/Math.max(...group.map(r=>r.throughput))*45}px`}}/></button>)}</div>
 <details className="card-data-notes"><summary>How to read these metrics</summary><p>Up to six snapshots across one data group. Bars show each metric relative to its group maximum; they are not a combined score. Pareto uses throughput and E2E latency.</p></details>
 </div>;
}
