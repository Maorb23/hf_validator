import React, { useRef, useState } from 'react';

const steps = [
  {name:'Define the workload',summary:'Task, demand and objectives',title:'Start with the decision.',body:'Describe the inference task, input/output lengths, request concurrency, hardware constraints, and what metric you want to optimize (e.g. Latency, Throughput, ttft).',artifact:'workload context',fields:[['input_tokens','128'],['output_tokens','32'],['concurrency','8']]},
  {name:'Research candidates',summary:'Model metadata and evidence',title:'Separate evidence from assumptions.',body:'Use model metadata and sourced research to assess candidates. Preserve verified, inferred, and unknown claims rather than turning a web mention into runtime support.',artifact:'research.json',fields:[['claim_status','unknown'],['source_urls','attached'],['runtime_support','requires verification']]},
  {name:'Pin the model',summary:'A precise model identity',title:'Make the tested identity explicit.',body:'Selection records an immutable revision. Validation and optimization must refer to the same selected model identity.',artifact:'selected model reference',fields:[['model_id','selected repository'],['revision','immutable commit'],['selection','persisted']],note:'No fabricated model revision is used in this walkthrough.'},
  {name:'Prepare the study',summary:'Search space and objectives',title:'Fix the workload. Define the search.',body:'Build a validated request with framework-specific search domains, objectives, GPU settings, timeout, and artifact configuration.',artifact:'intent.json + request.json',fields:[['search_strategy','optuna'],['objectives','throughput + E2E latency'],['workload','fixed']],note:'The real execution workflow requires explicit approval and cost checks.'},
  {name:'Execute remotely',summary:'Jobs, trials and artifacts',title:'Clear orchestration ownership.',body:'Validation uses a dedicated worker path. Nebius jobs, trials, checkpoints, and collection are delegated to LLM Optimizer.',artifact:'job references + manifests',fields:[['validation','worker payload'],['optimization','optimizer controller'],['artifacts','bucket']]},
  {name:'Compare and validate',summary:'Measurements into decisions',title:'Inspect the tradeoff, then validate.',body:'Compare only matching measurement contexts. Inspect the Pareto front and selected configuration. A search winner still needs independent inference validation.',artifact:'analysis + configuration',fields:[['comparison','context matched'],['pareto_front','nondominated results'],['next_step','independent inference validation']]},
];

function ArtifactIcon(){
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true"><path d="M14 3H6v18h12V7l-4-4Z"/><path d="M14 3v5h4M9 12h6M9 16h4"/></svg>;
}

export default function Journey(){
  const [index,setIndex]=useState(0);
  const buttons=useRef([]);
  const step=steps[index];
  function navigate(event,current){
    const offsets={ArrowDown:1,ArrowRight:1,ArrowUp:-1,ArrowLeft:-1};
    let next;
    if(event.key==='Home')next=0;
    else if(event.key==='End')next=steps.length-1;
    else if(event.key in offsets)next=(current+offsets[event.key]+steps.length)%steps.length;
    else return;
    event.preventDefault();setIndex(next);buttons.current[next]?.focus();
  }
  return <div className="journey journey-redesign">
    <div className="journey-progress"><span>Step <strong>{String(index+1).padStart(2,'0')}</strong> of {String(steps.length).padStart(2,'0')}</span><progress value={index+1} max={steps.length} aria-label="Model journey progress"/><span>{step.name}</span></div>
    <div className="journey-layout">
      <nav className="journey-steps" aria-label="Model journey steps">{steps.map((item,i)=><button key={item.name} ref={element=>{buttons.current[i]=element;}} aria-current={index===i?'step':undefined} aria-controls="journey-detail" onClick={()=>setIndex(i)} onKeyDown={event=>navigate(event,i)}><span className="journey-step-number">{String(i+1).padStart(2,'0')}</span><span className="journey-step-copy"><strong>{item.name}</strong><small>{item.summary}</small></span><svg className="journey-step-arrow" width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m7 5 5 5-5 5"/></svg></button>)}</nav>
      <section className="panel journey-detail" id="journey-detail" aria-label="Journey step details">
        <div className="journey-content" aria-live="polite" aria-atomic="true"><div className="journey-scene" key={index}>
          <div className="journey-story"><h2>{step.title}</h2><p>{step.body}</p>{step.note&&<p className="journey-note">{step.note}</p>}</div>
          <div className="journey-artifact"><div className="journey-artifact-heading"><ArtifactIcon/><h3>{step.artifact}</h3></div><p className="journey-artifact-caption">Illustrative structure</p><dl>{step.fields.map(([key,value])=><div key={key}><dt>{key}</dt><dd>{value}</dd></div>)}</dl></div>
        </div></div>
        <div className="journey-actions"><button className="secondary-button" disabled={index===0} onClick={()=>setIndex(index-1)}>← Previous</button><span aria-label={`Step ${index+1} of ${steps.length}`}>{String(index+1).padStart(2,'0')} / {String(steps.length).padStart(2,'0')}</span><button className="primary-button" onClick={()=>setIndex((index+1)%steps.length)}>{index===steps.length-1?'Start again':'Next step'} →</button></div>
      </section>
    </div>
  </div>;
}
