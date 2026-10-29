import UpliftSummary from './UpliftSummary.jsx';
import React, { useEffect, useMemo, useState } from 'react';
import { useDataset, DataState } from './data';
import { number } from './analysis.mjs';
import { measuredCohorts, resolveMeasuredFilters, matchingMeasuredRows, measuredPareto } from './measured-filters.mjs';

const colors = {'vLLM':'#a7f3cf','SGLang':'#81b9ff','TensorRT-LLM':'#d5adf6'};
const show = value => value === null || value === undefined ? 'Not reported' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const metric = value => value === null || value === undefined ? '—' : number(value);
function label(row) {
  const c=row.context,w=c.workload;
  return `${c.variant || 'Variant unreported'} · ${w.dataset_name || 'Unreported data'} · ${show(w.random_input_len)} in / ${show(w.random_output_len)} out · ${show(w.num_prompts)} prompts · c${show(w.max_concurrency)} · seed ${show(w.seed)} · draft ${c.draftModel || 'not reported'} · ${c.gpu || 'GPU not reported'} · ${row.cohort.slice(0,6)}`;
}

export default function MeasuredExplorer({replay=false}) {
  const {data,error,retry}=useDataset('publication.json');
  if(!data||error)return <DataState error={error} retry={retry}/>;
  return <Measurements data={data} replay={replay}/>;
}

function Measurements({data,replay}) {
  const initialRow=(replay && data.rows.find(r=>r.plotEligible)) || data.rows[0];
  const initialFilters={model:initialRow?.context.model || '',cohort:'all',engine:'all',sampler:'all',variant:'all',runId:replay ? initialRow?.runId || 'all' : 'all',frontOnly:false};
  const [filters,setFilters]=useState(initialFilters);
  const [pending,setPending]=useState(initialFilters);
  const {model,cohort:activeCohort,engine,sampler,runId,frontOnly}=filters;
  const [selected,setSelected]=useState(null),[comparison,setComparison]=useState([]),[page,setPage]=useState(0);
  const [step,setStep]=useState(null),[playing,setPlaying]=useState(false);
  const [filterNotice,setFilterNotice]=useState('');
  const models=[...new Set(data.studies.map(s=>s.context.model || ''))];
  const cohorts=measuredCohorts(data.rows,pending.model,pending.engine,pending.sampler,pending.variant);
  const modelRows=data.rows.filter(r=>(r.context.model || '')===pending.model);
  const availableSamplers=[...new Set(modelRows.filter(r=>pending.engine==='all'||r.engine===pending.engine).map(r=>r.sampler))];
  const study=data.studies.find(s=>s.id===runId);
  const allTrials=useMemo(()=>matchingMeasuredRows(data.rows,filters),[data,filters]);
  const appliedCohorts=[...new Map(allTrials.map(r=>[r.cohort,r])).values()];
  const hasPending=JSON.stringify(pending)!==JSON.stringify(filters);
  const observed=step===null?allTrials.length:Math.min(step,allTrials.length);
  const visible=allTrials.slice(0,observed);
  const eligible=visible.filter(r=>r.plotEligible);
  const front=useMemo(()=>measuredPareto(eligible),[allTrials,observed]);
  const best=eligible.reduce((winner,row)=>!winner||row.throughput>winner.throughput?row:winner,null);
  const frontIds=new Set(front.map(r=>r.id));
  const plotted=frontOnly?front:eligible;
  const displayed=[...(frontOnly?front:visible)].sort((a,b)=>Number(b.plotEligible)-Number(a.plotEligible)||(a.plotEligible?(b.throughput??0)-(a.throughput??0):0));
  const tableCount=frontOnly?front.length:visible.length;
  const inspected=data.rows.find(r=>r.id===selected);
  const inspectedStudy=data.studies.find(s=>s.id===inspected?.runId);
  const chosen=comparison.map(id=>data.rows.find(r=>r.id===id)).filter(Boolean);
  const maxX=Math.max(1,...eligible.map(r=>r.latency))*1.08,maxY=Math.max(1,...eligible.map(r=>r.throughput))*1.08;
  const maxPage=Math.max(0,Math.ceil(tableCount/12)-1);
  const currentPage=Math.min(page,maxPage);
  const tableRows=displayed.slice(currentPage*12,currentPage*12+12);
  function reset(){setSelected(null);setComparison([]);setPage(0);setStep(null);setPlaying(false);}
  function chooseRun(id){
    if(id==='all'){setPending(p=>({...p,runId:'all'}));return;}
    const study=data.studies.find(s=>s.id===id);
    setPending(p=>({...p,model:study.context.model||'',engine:study.engine,sampler:study.sampler||'all',variant:study.context.variant||'Unreported',cohort:'all',runId:id}));
  }
  function changeFilters(changes){setPending(p=>resolveMeasuredFilters(data.rows,p,changes));}
  function applyFilters(event){
    event.preventDefault();reset();setFilters({...pending});
    const count=matchingMeasuredRows(data.rows,pending).length;
    setFilterNotice(`Filters applied. ${count.toLocaleString()} recorded trials match this selection.`);
  }
  function resetFilters(){
    const next={...initialFilters,runId:'all'};reset();setFilters(next);setPending(next);setFilterNotice('Filters reset. Showing all data groups for this model.');
  }
  function canCompare(row){return row.plotEligible&&(!chosen.length||chosen[0].cohort===row.cohort);}
  function toggle(id){
    const row=data.rows.find(r=>r.id===id);
    if(!comparison.includes(id)&&!canCompare(row))return;
    setComparison(c=>c.includes(id)?c.filter(x=>x!==id):c.length<3?[...c,id]:c);
  }
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setStep(s=>{const next=(s??0)+1;if(next>=allTrials.length)setPlaying(false);return Math.min(next,allTrials.length);}),650);return()=>clearInterval(timer);},[playing,allTrials.length]);
  return <div className="measured-explorer">
    <div className="provenance-notice"><span className="provenance-tag">MEASURED</span><p>{data.studies.length} runs · {data.rows.length.toLocaleString()} recorded trials · {data.rows.filter(r=>r.plotEligible).length.toLocaleString()} completed measurements. Runs with missing metadata are retained; provenance gaps are disclosed, not guessed.</p></div>
    <form onSubmit={applyFilters} aria-label="Measured benchmark filters">
    <div className="control-bar measured-controls">
      <label>Model<select aria-label="Measured model" value={pending.model} onChange={e=>changeFilters({model:e.target.value})}>{models.map(m=><option key={m} value={m}>{m||'Not reported'}</option>)}</select></label>
      <label>Variant<select aria-label="Measured variant" value={pending.variant} onChange={e=>changeFilters({variant:e.target.value})}><option value="all">All variants</option>{[...new Set(data.studies.filter(s=>(s.context.model||'')===pending.model).map(s=>s.context.variant||'Unreported'))].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Data, seed and draft model<select aria-label="Measured cohort" value={pending.cohort} onChange={e=>changeFilters({cohort:e.target.value})}><option value="all">All matching data groups</option>{cohorts.map(r=><option key={r.cohort} value={r.cohort}>{label(r)}</option>)}</select></label>
      <label>Engine<select aria-label="Measured engine" value={pending.engine} onChange={e=>changeFilters({engine:e.target.value})}><option value="all">All engines</option>{Object.keys(colors).map(e=><option key={e} disabled={!modelRows.some(r=>r.engine===e)}>{e}</option>)}</select></label>
      <label>Search strategy<select aria-label="Measured sampler" value={pending.sampler} onChange={e=>changeFilters({sampler:e.target.value})}><option value="all">All strategies</option>{availableSamplers.map(s=><option key={s}>{s}</option>)}</select></label>
      <label>Run archive<select aria-label="Measured run" value={pending.runId} onChange={e=>chooseRun(e.target.value)}><option value="all">All runs in this selection</option>{data.studies.map(s=><option key={s.id} value={s.id}>{s.context.model} · {s.context.variant||'Variant unreported'} · {s.engine} · {s.sampler} · {s.id} · {s.state} · {s.trialCount} trials</option>)}</select></label>
      <label className="check-control"><input type="checkbox" checked={pending.frontOnly} onChange={e=>setPending(p=>({...p,frontOnly:e.target.checked}))}/> Measured Pareto points only</label>
    </div>
    <div className="button-row measured-filter-actions"><button className="primary-button" type="submit">Apply filters</button><button className="secondary-button" type="button" onClick={resetFilters}>Reset filters</button><span className="small muted">{hasPending?'Changes pending. Apply filters to update results.':'Showing applied filters.'}</span></div>
    </form>
    {filterNotice&&<p className="small muted" role="status">{filterNotice}</p>}
    <p className="small muted">Showing {allTrials.length.toLocaleString()} recorded trials across {appliedCohorts.length} data {appliedCohorts.length===1?'group':'groups'}. All matching groups are included unless you select a specific one. Pareto frontiers and configuration comparisons stay within each group.</p>
    <details className="technical-details"><summary>Selected model, draft model, seed and data · {appliedCohorts.length} groups</summary>{appliedCohorts.map(r=><div key={r.cohort}><p>{r.cohort} · {label(r)}</p><pre className="measured-json">{JSON.stringify(r.context,null,2)}</pre></div>)}</details>
    {best&&<section className="panel measured-best"><h2>Highest output throughput in this selection.</h2><p><strong className="measured-best-value">{number(best.throughput,2)} output tok/s</strong></p><p>{best.context.model} · {best.engine} · run {best.runId} · trial {best.number} · p95 E2E {number(best.latency,2)} ms</p><p className="small muted">{label(best)}. SLO feasible: {show(best.sloFeasible)}. This is the highest observed value in this selection, not a normalized ranking across workloads.</p><button className="secondary-button" onClick={()=>setSelected(best.id)}>Inspect highest-throughput trial</button></section>}
    <section className="panel measured-replay"><h2>{study?'Replay this run.':'Explore the recorded trials.'}</h2><p>{study?`${study.id} · ${study.engine} · ${study.sampler} · ${study.state}`:'Select any run, including crashed runs, to inspect its recorded history.'} Replay follows export order, not wall-clock time. Failed and pruned trials remain in the table.</p><div className="button-row"><button className="secondary-button" onClick={()=>{if(!playing&&observed===allTrials.length)setStep(0);setPlaying(!playing);}} disabled={!allTrials.length}>{playing?'Pause replay':'Play replay'}</button><button className="secondary-button" onClick={()=>{setPlaying(false);setStep(Math.min(allTrials.length,(step??0)+1));setPage(0);}} disabled={!allTrials.length}>Step trial →</button><button className="secondary-button" onClick={()=>{setPlaying(false);setStep(null);setPage(0);}}>Show all trials</button><output>{observed} / {allTrials.length} trials</output></div><label className="range-control">Recorded trials<input aria-label="Measured replay progress" type="range" min="0" max={allTrials.length} value={observed} onChange={e=>{setPlaying(false);setStep(Number(e.target.value));setPage(0);}}/></label>
    {study&&<details><summary>Recorded context and missing metadata</summary><p>Draft model: {study.context.draftModel || 'Not reported; absence is not confirmed.'}</p><p>Missing metadata: {study.missingMetadata.join(', ') || 'None in the exported context fields'}.</p><StudyRecord study={study}/></details>}</section>
    <div className="explorer-layout"><section className="panel chart-panel"><div className="chart-heading"><h2>Throughput and p95 latency.</h2><span>{plotted.length} measured points</span></div>
    <UpliftSummary rows={data.rows.filter(r=>r.context.model===model)} onInspect={setSelected}/>
    {eligible.length?<svg className="science-chart" viewBox="0 0 700 390" role="group" aria-label="Measured output throughput versus p95 end-to-end latency">{[0,.25,.5,.75,1].map(t=><g key={t}><line x1="75" y1={320-t*270} x2="665" y2={320-t*270} className="plot-grid"/><text x="63" y={324-t*270} textAnchor="end" className="chart-tick">{number(t*maxY,0)}</text><text x={75+t*590} y="345" textAnchor="middle" className="chart-tick">{number(t*maxX,0)}</text></g>)}{appliedCohorts.map(group=><path key={group.cohort} d={front.filter(r=>r.cohort===group.cohort).sort((a,b)=>a.latency-b.latency).map((r,i)=>`${i?'L':'M'}${75+r.latency/maxX*590},${320-r.throughput/maxY*270}`).join(' ')} fill="none" stroke="var(--accent)" strokeDasharray="5 5"/>)}{plotted.map(r=><g key={r.id} tabIndex="0" role="button" className="sample-point" aria-label={`Inspect ${r.id}: ${number(r.throughput)} tokens per second, ${number(r.latency)} milliseconds p95`} onClick={()=>setSelected(r.id)} onKeyDown={e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();setSelected(r.id);}}}><circle cx={75+r.latency/maxX*590} cy={320-r.throughput/maxY*270} r={selected===r.id?8:frontIds.has(r.id)?5:3.5} fill={colors[r.engine]||'#fff'} opacity={frontIds.has(r.id)?1:.5}/><title>{r.id} · {r.engine}</title></g>)}<g className="chart-axis"><text x="365" y="379" textAnchor="middle">p95 E2E latency · ms →</text><text transform="translate(17 185) rotate(-90)" textAnchor="middle">Output throughput · tok/s →</text></g></svg>:<p className="empty-message">No completed measurements with valid throughput and p95 latency in this selection. Recorded trials remain available below.</p>}
    <div className="chart-legend">{Object.entries(colors).map(([name,color])=><span key={name}><i style={{background:color}}/>{name}</span>)}</div><p className="chart-footnote">Each Pareto frontier maximizes output throughput and minimizes p95 E2E latency within one recorded data group. Points from different groups share the plot for browsing; they are not treated as comparable. These are observed configurations, not a general engine ranking.</p></section>
    <aside className="panel result-inspector"><h3>{inspected?'Selected trial':'Inspect a measured trial.'}</h3>{inspected?<><p>{inspected.id} · {inspected.state}</p><p>{inspected.context.model} · {inspected.context.variant||'Variant unreported'}</p><p className="small muted">{label(inspected)} · group {inspected.cohort}</p><dl className="metric-grid"><div><dt>Output throughput</dt><dd>{metric(inspected.throughput)} <small>tok/s</small></dd></div><div><dt>p95 E2E latency</dt><dd>{metric(inspected.latency)} <small>ms</small></dd></div><div><dt>p95 TTFT</dt><dd>{metric(inspected.ttft)} <small>ms</small></dd></div><div><dt>SLO feasible</dt><dd>{show(inspected.sloFeasible)}</dd></div></dl><p className="small">Measurement source: {show(inspected.measurementSource)}. {inspected.plotEligible?'Completed measured trial.':'Retained diagnostic trial; excluded from the performance frontier.'}</p><details><summary>Trial measurements, timings and logs</summary><p>{inspected.diagnosticReasons?.join('; ')}</p><pre className="measured-json">{JSON.stringify({parameters:inspected.parameters,metrics:inspected.metrics,metadata:inspected.trialMetadata,objectives:inspected.objectives,requestAccounting:inspected.requestAccounting,failureReason:inspected.failureReason,conflicts:inspected.metadataConflicts},null,2)}</pre></details>{inspectedStudy&&<StudyRecord study={inspectedStudy}/>}<details open><summary>Serving configuration</summary><pre className="measured-json">{JSON.stringify(inspected.configuration,null,2)}</pre></details><button className="secondary-button" disabled={!comparison.includes(inspected.id)&&(comparison.length>=3||!canCompare(inspected))} onClick={()=>toggle(inspected.id)}>{comparison.includes(inspected.id)?'Remove from comparison':'Compare trial'}</button>{!canCompare(inspected)&&<p className="small muted">Comparisons require valid measurements from the same data group.</p>}</>:<p>Select a point or trial row to inspect throughput, latency, TTFT, and serving settings.</p>}<a href="/methodology">Measurement methodology →</a></aside></div>
    <section className="comparison panel"><h3>Compare measured configurations <span>{chosen.length} / 3</span></h3>{chosen.length?<div className="table-scroll"><table><thead><tr><th>Metric / configuration</th>{chosen.map(r=><th key={r.id}>{r.id}<button className="remove-button" aria-label={`Remove ${r.id}`} onClick={()=>toggle(r.id)}>×</button></th>)}</tr></thead><tbody>{[['Engine','engine'],['Trial state','state'],['Output throughput · tok/s','throughput'],['p95 E2E latency · ms','latency'],['p95 TTFT · ms','ttft']].map(([title,key])=><tr key={key}><th>{title}</th>{chosen.map(r=><td key={r.id}>{['throughput','latency','ttft'].includes(key)?metric(r[key]):show(r[key])}</td>)}</tr>)}{[...new Set(chosen.flatMap(r=>Object.keys(r.configuration)))].map(key=><tr key={key}><th>{key}</th>{chosen.map(r=><td key={r.id}>{show(r.configuration[key])}</td>)}</tr>)}</tbody></table></div>:<p>Select up to three trials from the same data group. Applying filters clears the comparison.</p>}</section>
    <details className="technical-details" open><summary>Recorded trial table · {tableCount} trials · valid measurements by throughput, then diagnostics</summary><div className="table-scroll"><table><thead><tr><th>Compare</th><th>Trial</th><th>Engine / variant</th><th>Data group</th><th>State</th><th>Output · tok/s</th><th>p95 E2E · ms</th><th>p95 TTFT · ms</th></tr></thead><tbody>{tableRows.map(r=><tr key={r.id}><td><input type="checkbox" aria-label={`Compare measured ${r.id}`} checked={comparison.includes(r.id)} disabled={!comparison.includes(r.id)&&(comparison.length>=3||!canCompare(r))} onChange={()=>toggle(r.id)}/></td><td><button className="table-link" onClick={()=>setSelected(r.id)}>{r.id} · #{r.number}</button></td><td>{r.engine} · {r.context.variant||'Unreported'}</td><td>{r.cohort.slice(0,6)}</td><td>{r.state}{!r.plotEligible?' · diagnostic':''}</td><td>{metric(r.throughput)}</td><td>{metric(r.latency)}</td><td>{metric(r.ttft)}</td></tr>)}</tbody></table></div><div className="table-pagination"><button className="secondary-button" disabled={!currentPage} onClick={()=>setPage(currentPage-1)}>← Previous</button><span>Page {currentPage+1} of {maxPage+1}</span><button className="secondary-button" disabled={currentPage>=maxPage} onClick={()=>setPage(currentPage+1)}>Next →</button></div></details>
    <details className="technical-details"><summary>Provenance and comparison context</summary><p>{data.note.replace(/\bstudies\b/gi,'runs').replace(/\bstudy\b/gi,'run')}</p><p>Model, recorded draft model, seed and workload define the comparison context alongside recorded hardware and revision. Unknown values remain unknown. Different engine versions, search budgets, constraints or incomplete context can limit interpretation. A shared seed alone does not prove identical requests.</p><p>Some trials report MEASURED_ALLOCATION_WALL_PROXY as their measurement source. Preserve this source label; do not equate a proxy measurement with an independently validated engine benchmark.</p><p>Source SHA-256: <code className="measured-hash">{data.sourceSha256}</code></p><details><summary>Import sources and checksums</summary><pre className="measured-json">{JSON.stringify(data.sources,null,2)}</pre></details><a href="/data/publication.json" download>Download sanitized runs and trials ↓</a></details>
  </div>;
}


function StudyRecord({study}) {
  return <details className="technical-details"><summary>Run metadata · {study.id}</summary>
    <p>Duration: {study.durationSeconds == null ? 'Not reported' : `${number(study.durationSeconds,2)} seconds (${study.durationSource})`}. Termination reason: {show(study.terminationReason)}. Final state: {show(study.state)}. Interrupted: {show(study.interrupted)}.</p>
    {study.sourceUrl&&<p><a href={study.sourceUrl} target="_blank" rel="noreferrer">Original run record ↗</a></p>}
    <p>Source SHA-256: <code className="measured-hash">{study.sourceSha256}</code></p>
    {!!study.metadataConflicts?.length&&<div role="note"><strong>Recorded metadata conflicts</strong>{study.metadataConflicts.map((c,i)=><p key={i}>{c.field}: {show(c.recorded)}. Resolved to {show(c.resolved)}. {c.reason}</p>)}</div>}
    <pre className="measured-json">{JSON.stringify({context:study.context,search:study.search,timings:study.timings,missingMetadata:study.missingMetadata},null,2)}</pre>
  </details>;
}
