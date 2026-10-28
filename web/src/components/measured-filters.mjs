import { pareto } from './analysis.mjs';

// Browsing can include several recorded contexts; comparisons still use one.
export function measuredCohorts(rows, model, engine = 'all', sampler = 'all', variant = 'all') {
  return [...new Map(rows.filter(row =>
    (row.context.model || '') === model &&
    (engine === 'all' || row.engine === engine) &&
    (sampler === 'all' || row.sampler === sampler)
    && (variant === 'all' || (row.context.variant || 'Unreported') === variant)
  ).map(row => [row.cohort, row])).values()];
}

export function matchingMeasuredRows(rows, filters) {
  return rows.filter(row =>
    (row.context.model || '') === filters.model &&
    (!filters.variant || filters.variant === 'all' || (row.context.variant || 'Unreported') === filters.variant) &&
    (filters.engine === 'all' || row.engine === filters.engine) &&
    (filters.sampler === 'all' || row.sampler === filters.sampler) &&
    (filters.cohort === 'all' || row.cohort === filters.cohort) &&
    (!filters.runId || filters.runId === 'all' || row.runId === filters.runId)
  );
}

export function resolveMeasuredFilters(rows, current, changes) {
  const next = { ...current, ...changes };
  if ('model' in changes) {
    next.engine = 'all';
    next.sampler = 'all';
    if ('variant' in current) next.variant = 'all';
  }
  if ('engine' in changes && !measuredCohorts(rows, next.model, next.engine, next.sampler).length) {
    next.sampler = 'all';
  }
  // Never silently select the first data group and hide the rest of an engine.
  if (['model', 'engine', 'sampler', 'variant'].some(key => key in changes)) next.cohort = 'all';
  const cohorts = measuredCohorts(rows, next.model, next.engine, next.sampler);
  if (next.cohort !== 'all' && !cohorts.some(row => row.cohort === next.cohort)) next.cohort = 'all';
  if ('runId' in current) next.runId = 'all';
  return next;
}

export function measuredPareto(rows) {
  const groups = new Map();
  for (const row of rows) {
    if (row.plotEligible === false) continue;
    if (!groups.has(row.cohort)) groups.set(row.cohort, []);
    groups.get(row.cohort).push(row);
  }
  return [...groups.values()].flatMap(group => pareto(group));
}
