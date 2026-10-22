import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource/inter/latin-400.css';
import '@fontsource/inter/latin-500.css';
import '@fontsource/inter/latin-600.css';
import '@fontsource/jetbrains-mono/latin-400.css';
import './styles.css';

function Icon({ name = 'arrow', size = 20, ...props }) {
  const paths = {
    arrow: <><path d="M5 12h14M13 6l6 6-6 6" /></>,
    diagonal: <><path d="M6 18 18 6M6 6h12v12" /></>,
    research: <><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" /><path d="m4 7.5 8 4.5 8-4.5M12 12v9M8 5.3l8 4.5" /></>,
    optimize: <><path d="M5 4v16h15M8 15l4-5 4 3 4-7" /><circle cx="12" cy="10" r="1" /><circle cx="16" cy="13" r="1" /></>,
    server: <><rect x="4" y="4" width="16" height="6" rx="2" /><rect x="4" y="14" width="16" height="6" rx="2" /><path d="M8 7h.01M8 17h.01M16 7h1M16 17h1" /></>,
    pin: <><path d="M9 3h6l1 6 3 3v2h-6v7l-2-3v-4H5v-2l3-3 1-6Z" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    file: <><path d="M14 3H6v18h12V7l-4-4Z M14 3v5h4M9 12h6M9 16h6" /></>,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5" /></>,
    moon: <path d="M20 14A8 8 0 0 1 10 4a8 8 0 1 0 10 10Z" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>;
}

function Wordmark() {
  return <a className="wordmark" href="#top" aria-label="Infiro home"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true"><path d="M4 25V7h6v12h6V7h6v12h6v6H4Z" fill="currentColor" /><path d="M22 3h6v6h-6z" fill="var(--accent)" /></svg><span>infiro<span className="wordmark-dot">.</span></span></a>;
}

const frontier = [[94, 263], [132, 222], [181, 188], [231, 165], [288, 146], [350, 134], [412, 126], [482, 120]];
const cloud = Array.from({ length: 49 }, (_, i) => {
  const x = 95 + ((i * 73 + 21) % 382);
  const floor = 108 + 110 * Math.exp(-(x - 90) / 100);
  return [x, floor + 18 + (((i * 47) % 91) / 91) * (278 - floor - 18)];
});

function TradeoffPlot({ choice, landscape }) {
  const selected = choice === 'latency' ? 1 : choice === 'throughput' ? 6 : 3;
  return <svg className="tradeoff-svg" viewBox="0 0 560 360" role="img" aria-label={landscape ? 'Illustrative search landscape with contours and candidate configurations. Not measured performance.' : 'Conceptual Pareto frontier. Horizontal axis is relative latency; vertical axis is relative throughput. Not measured performance.'}>
    <defs>
      <linearGradient id="plot-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--accent)" stopOpacity=".17" /><stop offset="100%" stopColor="var(--accent)" stopOpacity="0" /></linearGradient>
      <radialGradient id="plot-aura"><stop offset="0%" stopColor="var(--accent)" stopOpacity=".10" /><stop offset="100%" stopColor="var(--accent)" stopOpacity="0" /></radialGradient>
    </defs>
    <ellipse cx="315" cy="160" rx="240" ry="155" fill="url(#plot-aura)" />
    {[90, 140, 190, 240, 290].map(y => <line key={`h${y}`} x1="65" y1={y} x2="512" y2={y} className="plot-grid" />)}
    {[90, 170, 250, 330, 410, 490].map(x => <line key={`v${x}`} x1={x} y1="52" x2={x} y2="300" className="plot-grid" />)}
    <path d="M65 52v248h447" className="plot-axis" fill="none" />
    {landscape ? <g className="contour-lines">{Array.from({ length: 13 }, (_, i) => <ellipse key={i} cx={310 + i * 2} cy={174 - i * 1.5} rx={24 + i * 15} ry={13 + i * 7.2} transform="rotate(-23 320 170)" fill="none" stroke="var(--accent)" strokeOpacity={.8 - i * .045} />)}{cloud.slice(0, 20).map(([x, y], i) => <circle key={i} cx={x} cy={y} r="2.5" fill="var(--plot-point)" />)}<circle cx="310" cy="174" r="6" className="selected-point" /><path d="m310 160 0-12m0 38v12m-14-24h-12m40 0h12" stroke="var(--accent)" /></g> : <>
      <path d="M94 263C150 189 199 175 231 165S350 128 482 120L482 300H94Z" fill="url(#plot-fill)" />
      {cloud.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r={i % 4 === 0 ? 3.4 : 2.7} fill="var(--plot-point)" opacity={.35 + (i % 4) * .12} /></g>)}
      <path d="M94 263C150 189 199 175 231 165S350 128 482 120" fill="none" stroke="var(--accent)" strokeWidth="2" />
      {frontier.map(([x, y], i) => <g key={i}><circle cx={x} cy={y} r={i === selected ? 13 : 5} fill={i === selected ? 'var(--accent-haze)' : 'var(--surface)'} stroke={i === selected ? 'none' : 'var(--accent)'} strokeWidth="1.5" /><circle cx={x} cy={y} r={i === selected ? 5 : 2} fill="var(--accent)" /></g>)}
      <path d={`M${frontier[selected][0]} ${frontier[selected][1] + 18}v${284 - frontier[selected][1]}`} className="plot-guide" />
      <text x="401" y="101" className="plot-annotation">Pareto frontier</text>
    </>}
    <text x="287" y="340" textAnchor="middle" className="plot-label">LATENCY →</text>
    <text transform="translate(22 182) rotate(-90)" textAnchor="middle" className="plot-label">THROUGHPUT →</text>
    <text x="75" y="73" className="plot-small">relative scale</text>
  </svg>;
}

function HeroGraphic() {
  const [choice, setChoice] = useState('balanced');
  const [landscape, setLandscape] = useState(false);
  return <div className="hero-graphic">
    <div className="graphic-top"><span className="graphic-title"><span className="tiny-cross">+</span> The inference tradeoff</span><span className="data-label">CONCEPTUAL</span></div>
    <div className="graphic-tabs" aria-label="Visualization"><button aria-pressed={!landscape} onClick={() => setLandscape(false)}>Pareto frontier</button><button aria-pressed={landscape} onClick={() => setLandscape(true)}>Search landscape</button></div>
    <TradeoffPlot choice={choice} landscape={landscape} />
    <div className="plot-controls"><span>{landscape ? 'Explore the search space' : 'What matters to your workload?'}</span><div className="segmented" aria-label="Objective preference">{['latency', 'balanced', 'throughput'].map(value => <button key={value} aria-pressed={choice === value} onClick={() => {setChoice(value); setLandscape(false);}}>{value}</button>)}</div></div>
    <div className="graphic-foot"><span className="legend-dot" />Illustrative data · no measured performance<span className="corner-mark">↗</span></div>
  </div>;
}

const nodes = [
  { id: 'research', name: 'Nemotron', subtitle: 'Model research', icon: 'research', role: 'Research with context.', description: 'Inspect model candidates using Hugging Face metadata, sourced research, and Nemotron-assisted assessment.', input: 'Requirements + source evidence', output: 'Assessed model candidates', note: 'Research confidence is distinct from runtime validation.' },
  { id: 'revision', name: 'Pinned revision', subtitle: 'Selected model', icon: 'pin', role: 'Keep the model precise.', description: 'Carry the selected model and immutable revision into validation and optimization preparation.', input: 'Selected model candidate', output: 'Model ID + immutable revision', note: 'A revision identifies what is tested; it does not establish compatibility.' },
  { id: 'optimizer', name: 'LLM Optimizer', subtitle: 'Configuration search', icon: 'optimize', role: 'Search the configuration.', description: 'Prepare a fixed-workload study. The existing optimizer controller owns search execution and its GPU jobs.', input: 'Revision + workload + search space', output: 'Trial configurations + objectives', note: 'Engine and sampler availability depend on the configured runtime.' },
  { id: 'nebius', name: 'Nebius Jobs', subtitle: 'Remote execution', icon: 'server', role: 'Run the workload remotely.', description: 'Run validation and optimization workloads as remote GPU jobs through their existing orchestration paths.', input: 'Validation or optimization request', output: 'Measurements + worker artifacts', note: 'Validation jobs and optimizer jobs have separate orchestration owners.' },
  { id: 'results', name: 'Inference evidence', subtitle: 'Comparable results', icon: 'file', role: 'Compare what was measured.', description: 'Analyze throughput and latency with their model, workload, hardware, and serving configuration attached.', input: 'Measurements + workload context', output: 'Comparable configurations + Pareto analysis', note: 'A search winner still requires independent validation.' },
];

function Architecture() {
  const [active, setActive] = useState('research');
  const selected = nodes.find(node => node.id === active);
  const index = nodes.findIndex(node => node.id === active);
  return <section className="architecture section" id="architecture">
    <div className="section-heading"><div><h2>One connected workflow.<br /><span className="muted-heading">Every decision, traceable.</span></h2><p>From a model candidate to a configuration you can inspect.<br className="desktop-break" /> Explore how the pieces work together.</p></div><button className="text-button walkthrough" onClick={() => setActive(nodes[(index + 1) % nodes.length].id)}>Follow the workflow <Icon /></button></div>
    <div className="architecture-workspace">
      <div className="architecture-map">
        <div className="map-caption"><span>Architecture walkthrough</span><span>SELECT A COMPONENT <span aria-hidden="true">↙</span></span></div>
        <div className="graph">
          <svg className="graph-connectors" viewBox="0 0 800 330" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="edge-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="m1 1 7 4-7 4" fill="none" stroke="var(--edge)" strokeWidth="1.5" /></marker></defs><g fill="none" stroke="var(--edge)" strokeWidth="1.2" markerEnd="url(#edge-arrow)"><path d="M207 89H295" /><path d="M475 89H568" /><path d="M660 132V218" /><path d="M385 132v42q0 18 18 18h239q18 0 18 18v8" /><path d="M568 261H475" /></g><g className="graph-flow" fill="none" stroke="var(--accent)" strokeWidth="1.5"><path d={active === 'research' ? 'M207 89H295' : active === 'revision' ? 'M475 89H568' : active === 'optimizer' ? 'M660 132V218' : 'M568 261H475'} /></g><text x="503" y="73">prepare</text><text x="460" y="180">validate</text><text x="502" y="246">artifacts</text></svg>
          {nodes.map((node, i) => <button key={node.id} className={`graph-node node-${node.id} ${active === node.id ? 'selected' : ''}`} onClick={() => setActive(node.id)} aria-pressed={active === node.id}><span className="node-icon"><Icon name={node.icon} size={22} /></span><span className="node-copy"><strong>{node.name}</strong><small>{node.subtitle}</small></span><span className="node-mobile-step">0{i + 1}</span></button>)}
        </div>
        <div className="map-legend"><span><i /> Research & preparation</span><span><i /> Execution & measurements</span></div>
      </div>
      <aside className="node-details" aria-live="polite" aria-atomic="true">
        <div className="details-symbol"><Icon name={selected.icon} size={26} /><span>0{index + 1} / 05</span></div>
        <h3>{selected.role}</h3><p>{selected.description}</p>
        <dl><div><dt>INPUT</dt><dd>{selected.input}</dd></div><div><dt>OUTPUT</dt><dd>{selected.output}</dd></div></dl>
        <div className="detail-note"><Icon name="file" size={16} /><span>{selected.note}</span></div>
      </aside>
    </div>
  </section>;
}

const engineSettings = {
  vllm: [['max_num_seqs', 'Serving capacity'], ['max_num_batched_tokens', 'Token budget'], ['gpu_memory_utilization', 'Memory allocation']],
  sglang: [['max_running_requests', 'Serving capacity'], ['chunked_prefill_size', 'Prefill budget'], ['mem_fraction_static', 'Memory allocation']],
  trtllm: [['max_batch_size', 'Serving capacity'], ['max_num_tokens', 'Token budget'], ['kv_cache_config', 'KV cache configuration']],
};
function Capability() {
  const [engine, setEngine] = useState('vllm');
  return <section className="capability section" id="optimization"><div className="capability-copy"><h2>A better configuration<br />starts with <em>your workload.</em></h2><p>Keep the benchmark fixed. Explore serving parameters. Compare throughput and latency with the context that makes the measurements meaningful.</p><ul className="benefit-list"><li><Icon name="check" /> Fixed input lengths, concurrency, and seed</li><li><Icon name="check" /> Framework-specific parameter search</li><li><Icon name="check" /> Explicit objectives and Pareto analysis</li></ul><a href="#architecture" className="text-button">See where optimization fits <Icon /></a><p className="compatibility-note">Preparation profiles cover vLLM, SGLang, and TensorRT-LLM. Model and runtime compatibility must be checked.</p></div>
    <div className="config-card"><div className="config-header"><span><Icon name="optimize" size={17} />Serving configuration</span><span className="data-label">EXAMPLE DIMENSIONS</span></div><div className="engine-tabs" aria-label="Inference engine">{[['vllm','vLLM'],['sglang','SGLang'],['trtllm','TensorRT-LLM']].map(([value,label]) => <button key={value} onClick={() => setEngine(value)} aria-pressed={engine === value}>{label}<span aria-hidden="true">↗</span></button>)}</div><div className="configuration-body"><div className="fixed-workload"><span className="fixed-icon"><Icon name="pin" size={16}/></span><div><strong>Workload stays fixed</strong><span>Model revision · request concurrency · tokens</span></div><span className="fixed-tag">FIXED</span></div><div className="parameter-table"><div className="parameter-table-head"><span>SEARCH DIMENSION</span><span>WHAT IT CONTROLS</span></div>{engineSettings[engine].map(([name,label], i) => <div className="parameter-row" key={name}><code><span className="parameter-number">0{i + 1}</span>{name}</code><span>{label}</span></div>)}</div><div className="config-objectives"><span>MEASURE</span><span>Throughput <Icon name="diagonal" size={13}/></span><span>Latency <span aria-hidden="true">↘</span></span></div></div><div className="config-footer"><span className="legend-dot" />Parameter names are illustrative; domains come from the prepared request.</div></div>
  </section>;
}

function Methodology({ close }) {
  return <div className="dialog-backdrop" onClick={close}><section className="methodology-dialog" role="dialog" aria-modal="true" aria-labelledby="method-title" onClick={e => e.stopPropagation()}><button className="icon-button dialog-close" aria-label="Close methodology" onClick={close} autoFocus><Icon name="close" /></button><h2 id="method-title">Context before comparison.</h2><p>The visuals in this design prototype are conceptual. They are not inference benchmark results.</p><p>Real comparisons need the same model revision, GPU platform and count, workload, request concurrency, objectives, and measurement constraints.</p><div className="method-list"><span>01 / Pin the revision</span><span>02 / Fix the workload</span><span>03 / Preserve the configuration</span><span>04 / Compare the measurements</span></div><button className="primary-button" onClick={close}>Back to the prototype <Icon /></button></section></div>;
}

function App() {
  const theme = window.location.pathname.startsWith('/light') ? 'light' : 'dark';
  const [menuOpen, setMenuOpen] = useState(false);
  const [methodOpen, setMethodOpen] = useState(false);
  const previousFocus = React.useRef(null);
  const openMethod = () => { previousFocus.current = document.activeElement; setMethodOpen(true); };
  const closeMethod = () => {
    setMethodOpen(false);
    requestAnimationFrame(() => {
      const target = previousFocus.current?.isConnected ? previousFocus.current : document.querySelector('.mobile-menu');
      target?.focus();
    });
  };
  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#101316' : '#f5f5f0';
    document.title = `Infiro — ${theme === 'dark' ? 'Dark' : 'Light'} technical prototype`;
  }, [theme]);
  React.useEffect(() => {
    if (!methodOpen && !menuOpen) return;
    const old = document.body.style.overflow;
    if (methodOpen) document.body.style.overflow = 'hidden';
    const handler = e => {
      if (e.key === 'Escape') { if (methodOpen) closeMethod(); setMenuOpen(false); }
      if (e.key === 'Tab' && methodOpen) {
        const buttons = [...document.querySelectorAll('.methodology-dialog button')];
        if (e.shiftKey && document.activeElement === buttons[0]) { e.preventDefault(); buttons.at(-1).focus(); }
        else if (!e.shiftKey && document.activeElement === buttons.at(-1)) { e.preventDefault(); buttons[0].focus(); }
      }
    };
    window.addEventListener('keydown', handler);
    return () => {document.body.style.overflow = old; window.removeEventListener('keydown', handler);};
  }, [methodOpen, menuOpen]);
  return <div className={`site theme-${theme}`} id="top"><a className="skip-link" href="#main">Skip to content</a>
    <header className="header"><div className="header-inner"><Wordmark /><nav className={menuOpen ? 'navigation open' : 'navigation'} aria-label="Main navigation"><a href="#architecture" onClick={() => setMenuOpen(false)}>Technology</a><a href="#optimization" onClick={() => setMenuOpen(false)}>Optimization</a><button onClick={() => {openMethod(); setMenuOpen(false);}}>Methodology <Icon name="diagonal" size={13}/></button></nav><a className="header-cta" href={theme === 'dark' ? '/light' : '/dark'}><Icon name={theme === 'dark' ? 'sun' : 'moon'} size={16}/><span>View {theme === 'dark' ? 'light' : 'dark'} design</span></a><button className="mobile-menu icon-button" aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen} aria-controls="mobile-links" onClick={() => setMenuOpen(!menuOpen)}><Icon name={menuOpen ? 'close' : 'menu'} /></button></div>{menuOpen && <div id="mobile-links" className="mobile-links"><a href="#architecture" onClick={() => setMenuOpen(false)}>Technology <Icon /></a><a href="#optimization" onClick={() => setMenuOpen(false)}>Optimization <Icon /></a><button onClick={() => {openMethod();setMenuOpen(false);}}>Methodology <Icon name="diagonal" /></button></div>}</header>
    <main id="main"><section className="hero section"><div className="hero-copy"><h1>Better inference.<br />Backed by<br /><em>evidence.</em></h1><p>Choose models with context. Run workloads remotely. Find the serving configurations that fit your throughput and latency needs.</p><div className="hero-actions"><a className="primary-button" href="#architecture">Explore the technology <Icon /></a><button className="hero-secondary" onClick={openMethod}>How we compare <Icon name="diagonal" size={16}/></button></div><div className="hero-proof"><span className="proof-icon"><Icon name="check" size={13}/></span>Model research → execution → inference optimization</div></div><HeroGraphic /></section>
      <div className="integration-strip section"><p>Connected by design.</p><div>{[
        ['Nemotron', 'nvidia.svg'],
        ['SGLang', 'sglang.png'],
        ['vLLM', 'vllm.svg'],
        ['TensorRT-LLM', 'nvidia.svg'],
        ['Nebius Serverless AI', `nebius-${theme}.svg`],
        ['LLM Optimizer', null],
        ['Dynamo', 'nvidia.svg'],
      ].map(([name, logo]) => <span key={name} className={`integration-brand ${name.startsWith('Nebius') ? 'integration-nebius' : ''}`}>{logo ? <img src={`/logos/${logo}`} alt={name.startsWith('Nebius') ? 'Nebius' : ''} width={name.startsWith('Nebius') ? 58 : 25} height="25" /> : <span className="optimizer-symbol" title="Custom LLM Optimizer symbol"><Icon name="optimize" size={22} /></span>}<span>{name.startsWith('Nebius') ? 'Serverless AI' : name}</span></span>)}</div></div>
      <Architecture /><Capability />
      <section className="closing section"><div><h2>Know what you’re comparing.</h2><p>The model. The workload. The configuration. The evidence.</p></div><button className="primary-button" onClick={openMethod}>Explore the methodology <Icon /></button></section>
    </main><footer className="footer section"><Wordmark /><p>Inference, with evidence.</p><span>Design exploration · provisional brand</span><a href={theme === 'dark' ? '/light' : '/dark'}>Compare {theme === 'dark' ? 'light' : 'dark'} direction <Icon name="arrow" size={15}/></a></footer>
    <div className="design-switcher" aria-label="Design direction"><span>DESIGN STUDY</span><a href="/dark" aria-current={theme === 'dark' ? 'page' : undefined}><span className="swatch dark-swatch"/>Dark</a><a href="/light" aria-current={theme === 'light' ? 'page' : undefined}><span className="swatch light-swatch"/>Light</a></div>
    {methodOpen && <Methodology close={closeMethod} />}
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);
