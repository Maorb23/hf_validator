"""CPU-only tests for public run-import retention and inference metric semantics."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('public_import', Path(__file__).parents[1] / 'scripts/import_public_optimizer_runs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ImportTests(unittest.TestCase):
    def test_missing_context_retains_runs_and_trial_states(self):
        trials = [{'trial/state': state, 'benchmark/source': 'MEASURED_ALLOCATION_WALL_PROXY',
                   'benchmark/output_throughput': 123, 'benchmark/p95_e2e_latency_ms': 456,
                   'benchmark/p95_ttft_ms': 78} for state in ('COMPLETE', 'FAIL', 'PRUNED')]
        payload = {'runs': [{'id': 'a', 'state': 'crashed', 'params': {'target_model': 'Org/Model',
                    'environment': {'model_revision': 'REPLACE_WITH_COMMIT'}, 'workload': {'seed': 7}}, 'trial_history': trials},
                           {'id': 'b', 'params': {}, 'trial_history': []}]}
        result = module.convert(payload, 'abc')
        self.assertEqual(len(result['studies']), 2)
        self.assertEqual(len(result['rows']), 3)
        self.assertEqual([r['plotEligible'] for r in result['rows']], [True, False, False])
        self.assertIsNone(result['studies'][0]['context']['revision'])
        self.assertEqual(result['rows'][0]['latency'], 456)
        self.assertEqual(result['rows'][0]['throughput'], 123)
        self.assertIsNone(result['studies'][0]['context']['draftModel'])

    def test_draft_model_and_workload_separate_cohorts(self):
        def run(draft, seed):
            return {'params': {'target_model': 'Org/Model', 'draft_model': draft, 'workload': {'seed': seed}}, 'trial_history': [{}]}
        result = module.convert({'runs': [run('Org/Draft', 1), run(None, 1), run('Org/Draft', 2)]}, 'hash')
        self.assertEqual(len({r['cohort'] for r in result['rows']}), 3)
        self.assertEqual(result['rows'][0]['context']['draftModel'], 'Org/Draft')

    def test_sanitize_and_reject_invalid_plot_values_without_dropping_trial(self):
        run = {'params': {}, 'trial_history': [{'trial/state': 'COMPLETE', 'benchmark/source': 'MEASURED',
               'benchmark/output_throughput': float('nan'), 'benchmark/p95_e2e_latency_ms': 4,
               'trial/config.model_path': '/home/private/model', 'trial/config.draft_model': 'Org/Draft'}]}
        result = module.convert({'runs': [run]}, 'hash')
        self.assertEqual(len(result['rows']), 1)
        self.assertFalse(result['rows'][0]['plotEligible'])
        self.assertIsNone(result['rows'][0]['throughput'])
        self.assertNotIn('model_path', result['rows'][0]['configuration'])


if __name__ == '__main__':
    unittest.main()
