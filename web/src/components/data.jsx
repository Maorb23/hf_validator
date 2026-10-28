import { useEffect, useState } from 'react';

export function useDataset(file) {
  const [state, setState] = useState({ data: null, error: '', retry: 0 });
  useEffect(() => {
    const controller = new AbortController();
    setState(s => ({ ...s, error: '' }));
    fetch(`/data/${file}`, { signal: controller.signal }).then(r => {
      if (!r.ok) throw new Error(`Dataset unavailable (${r.status})`);
      return r.json();
    }).then(data => {
      if (data.schemaVersion !== 1 && !(file === 'publication.json' && data.schemaVersion === 2)) throw new Error('Unsupported dataset version');
      setState(s => ({ ...s, data }));
    }).catch(error => { if (error.name !== 'AbortError') setState(s => ({ ...s, error: error.message })); });
    return () => controller.abort();
  }, [file, state.retry]);
  return { ...state, retry: () => setState(s => ({ ...s, retry: s.retry + 1 })) };
}

export function DataState({ error, retry }) {
  return <div className="panel data-state" role="status">{error ? <><h3>The dataset could not be loaded.</h3><p>{error}</p><button className="primary-button" onClick={retry}>Try again</button></> : <p>Loading the bundled dataset…</p>}</div>;
}

export function SimulationNotice({ children }) {
  return <div className="provenance-notice"><span className="provenance-tag">SIMULATION</span><p>{children || 'Synthetic data for learning. These are not inference measurements or evidence of engine performance.'}</p></div>;
}

export function saveQuery(values) {
  const url = new URL(window.location.href);
  for (const [key,value] of Object.entries(values)) url.searchParams.set(key, String(value));
  window.history.replaceState({}, '', url);
}
