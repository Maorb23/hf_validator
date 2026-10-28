export const metricLabels = {p95_e2e_latency_ms:'95th-percentile end-to-end latency (ms)',p95_ttft_ms:'95th-percentile time to first token (ms)',output_throughput:'Output throughput (tok/s)',mean_e2e_latency_ms:'Mean E2E latency (ms)',mean_ttft_ms:'Mean TTFT (ms)'};
export const frameworkColors = {vllm:'#a7f3cf',sglang:'#81b9ff',trtllm:'#d5adf6'};
export const displayMetric = value => Number.isFinite(value) ? value.toLocaleString(undefined,{maximumFractionDigits:2}) : 'Unavailable';
export const trialKey = trial => `${trial.framework}:${trial.trial_id??trial.configuration_id}`;
export const samplerLabel = settings => settings?.sampler==='TPESampler'&&(settings.tpe_multivariate||settings.sampler_kwargs?.multivariate)?'TPE (multivariate)':settings?.sampler||'TPESampler';
