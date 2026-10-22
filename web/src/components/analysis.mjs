// Educational visualization math only. Real result publication uses Python analysis.
export function pareto(rows) {
  return rows.filter(p => !rows.some(q => q.latency <= p.latency && q.throughput >= p.throughput && (q.latency < p.latency || q.throughput > p.throughput)));
}
export function comparable(a, b) { return a.cohort === b.cohort; }
export const number = (n, digits = 1) => Number(n).toLocaleString('en-US', { maximumFractionDigits: digits });
