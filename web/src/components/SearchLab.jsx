import React, { useEffect, useState } from 'react';
import { useDataset, DataState, SimulationNotice, saveQuery } from './data';
import { number } from './analysis.mjs';

const names = { tpe: 'TPE', gp: 'Gaussian process', random: 'Random search' };
const colors = { tpe: '#a7f3cf', gp: '#81b9ff', random: '#d5adf6' };

export default function SearchLab() {
  const { data, error, retry } = useDataset('simulations.json');
  const [objective, setObjective] = useState('branin');
  const [seed, setSeed] = useState(11);
  const [strategy, setStrategy] = useState('tpe');
  const [count, setCount] = useState(12);
  const [playing, setPlaying] = useState(false);
  const [inspected, setInspected] = useState(null);
  useEffect(() => {
    const q = new URLSearchParams(location.search);
    if (['branin','hills'].includes(q.get('objective'))) setObjective(q.get('objective'));
    if ([11,29].includes(Number(q.get('seed')))) setSeed(Number(q.get('seed')));
    if (Object.keys(names).includes(q.get('strategy'))) setStrategy(q.get('strategy'));
  }, []);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => { if (document.hidden) return; setCount(n => { if (n >= 24) { setPlaying(false); return n; } return n + 1; }); }, 650);
    return () => clearInterval(id);
  }, [playing]);
  function reset(next = {}) {
    setCount(0); setPlaying(false); setInspected(null);
    if (next.objective) setObjective(next.objective);
    if (next.seed) setSeed(next.seed);
    if (next.strategy) setStrategy(next.strategy);
    saveQuery({ objective, seed, strategy, ...next });
  }
  if (!data || error) return <DataState error={error} retry={retry}/>;
  const experiment = data.experiments.find(e => e.objective === objective && e.seed === seed);
  const trace = experiment.traces[strategy];
  const visible = trace.slice(0, count);
  const trial = visible[Math.min(inspected ?? count - 1, count - 1)];
  const min = Math.min(...experiment.grid.map(p => p.value));
  const max = Math.max(...experiment.grid.map(p => p.value));
  const historyMax = Math.max(...Object.values(experiment.traces).flat().map(t => t.best));
  return <div className="lab-root">
    <SimulationNotice>Actual Optuna samplers on synthetic objectives. Recorded offline; no model inference or GPU jobs.</SimulationNotice>
    <div className="control-bar"><label>Objective<select aria-label="Objective" value={objective} onChange={e => reset({objective:e.target.value})}><option value="branin">Branin · three basins</option><option value="hills">Two basins · smooth landscape</option></select></label><label>Seed<select aria-label="Seed" value={seed} onChange={e => reset({seed:Number(e.target.value)})}><option value="11">11</option><option value="29">29</option></select></label><div className="strategy-tabs" aria-label="Search strategy">{Object.entries(names).map(([key,label])=><button key={key} aria-pressed={strategy===key} onClick={()=>{setStrategy(key);setInspected(null);saveQuery({objective,seed,strategy:key});}} style={{'--strategy-color':colors[key]}}><i/>{label}</button>)}</div></div>
    <div className="lab-charts"><section className="panel chart-panel"><div className="chart-heading"><h2>Where the sampler looks.</h2><span>{names[strategy]}</span></div><svg viewBox="0 0 500 385" className="science-chart heatmap" role="group" aria-label="Synthetic objective heatmap with selectable sampled points"><rect x="55" y="25" width="400" height="300" fill="#172721"/>{experiment.grid.map((p,i)=><rect key={i} x={55+p.x*387} y={25+(1-p.y)*290} width="13" height="10" fill={`hsl(${160-(p.value-min)/(max-min)*50} 35% ${12+(1-(p.value-min)/(max-min))*23}%)`}/>)}{visible.map((p,i)=><g key={i} role="button" tabIndex="0" aria-label={`Inspect trial ${i+1}, loss ${number(p.value,3)}`} onClick={()=>setInspected(i)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setInspected(i);}}} className={`sample-point ${trial?.number===p.number?'selected-sample':''}`}><circle cx={55+p.x*400} cy={325-p.y*300} r={trial?.number===p.number?8:5} fill={colors[strategy]} stroke="#101316" strokeWidth="2"/><title>Trial {i+1}: {number(p.value,3)}</title></g>)}<g className="chart-axis"><text x="55" y="349">0</text><text x="452" y="349">1</text><text x="32" y="328">0</text><text x="32" y="35">1</text><text x="250" y="377" textAnchor="middle">Parameter x · normalized</text><text transform="translate(17 176) rotate(-90)" textAnchor="middle">Parameter y · normalized</text></g></svg><div className="heatmap-legend"><span>Lower objective loss</span><i/><span>Higher objective loss</span></div></section>
    <section className="panel chart-panel"><div className="chart-heading"><h2>What it has found.</h2><span>Best observed loss ↓</span></div><svg viewBox="0 0 500 385" className="science-chart" role="img" aria-label="Best-so-far objective loss by trial for all three strategies. Lower is better.">{[0,.25,.5,.75,1].map(t=><g key={t}><line x1="65" y1={315-t*265} x2="460" y2={315-t*265} className="plot-grid"/><text x="55" y={319-t*265} textAnchor="end" className="chart-tick">{number(t*historyMax)}</text></g>)}{Object.entries(experiment.traces).map(([key,rows])=><path key={key} d={rows.slice(0,count).map((p,i)=>`${i?'L':'M'}${65+i/23*395},${315-p.best/historyMax*265}`).join(' ')} fill="none" stroke={colors[key]} strokeWidth={key===strategy?3:1.8} strokeOpacity={key===strategy?1:.65}/>)}<g className="chart-axis"><text x="65" y="342">1</text><text x="450" y="342">24</text><text x="260" y="377" textAnchor="middle">Trial number</text></g></svg><div className="chart-legend">{Object.entries(names).map(([key,label])=><span key={key}><i style={{background:colors[key]}}/>{label}</span>)}</div></section></div>
    <div className="playback panel"><button className="primary-button" onClick={()=>{if(count===24)setCount(0);setPlaying(!playing);}}>{playing?'Pause':'Play'} <span aria-hidden="true">{playing?'Ⅱ':'▷'}</span></button><button className="secondary-button" onClick={()=>{setPlaying(false);setCount(n=>Math.min(24,n+1));}}>Step →</button><button className="text-button" onClick={()=>reset()}>Reset</button><label className="timeline">Observed trials <input aria-label="Observed trials" type="range" min="0" max="24" value={count} onChange={e=>{setPlaying(false);setCount(Number(e.target.value));setInspected(null);}}/></label><output aria-live="polite">{count} / 24</output></div>
    <div className="two-cards"><section className="panel trial-inspector"><h3>{trial?`Trial ${trial.number+1} · ${names[strategy]}`:'No observations yet.'}</h3>{trial?<dl className="metric-grid"><div><dt>Parameter x</dt><dd>{number(trial.x,4)}</dd></div><div><dt>Parameter y</dt><dd>{number(trial.y,4)}</dd></div><div><dt>Objective loss</dt><dd>{number(trial.value,4)}</dd></div><div><dt>Best so far</dt><dd>{number(trial.best,4)}</dd></div></dl>:<p>Step or play to reveal the recorded observations.</p>}<p className="small muted">{count<=6?'The first six trials are the sampler’s startup observations.':'The model-based samplers can now use observations to guide proposals.'}</p></section><section className="panel"><h3>Read the experiment correctly.</h3><p>Every strategy gets the same bounds, deterministic objective, seed, and trial budget. Different seeds can change the outcome. These functions are educational examples, not serving workloads.</p><p className="small muted">Seeded, single-threaded generation. Optuna {data.versions.optuna}. No posterior uncertainty or acquisition function is visualized.</p><a href="/data/simulations.json" download>Download observations & provenance ↓</a></section></div>
    <details className="technical-details"><summary>Objective definitions and reproducibility</summary><p>Branin maps x to [−5,10] and y to [0,15], then evaluates the standard Branin function. The two-basin objective is defined in the downloadable evaluation guide. Both are minimized.</p><p>GPSampler uses deterministic_objective=true. TPE and GP use six startup trials. The browser does not run an optimizer; it replays generated JSON. Reproduce with the versioned Python script in web/scripts.</p><code>Dataset SHA-256: {data.sha256}</code></details>
    <details className="technical-details"><summary>Accessible observation table · {names[strategy]}</summary><div className="table-scroll"><table><thead><tr><th>Trial</th><th>x</th><th>y</th><th>Loss</th><th>Best so far</th></tr></thead><tbody>{visible.map(t=><tr key={t.number}><td>{t.number+1}</td><td>{number(t.x,4)}</td><td>{number(t.y,4)}</td><td>{number(t.value,4)}</td><td>{number(t.best,4)}</td></tr>)}</tbody></table></div></details>
  </div>;
}
