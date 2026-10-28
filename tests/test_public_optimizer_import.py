"""CPU-only tests for public run-import retention and inference metric semantics."""
import importlib.util
from pathlib import Path
import unittest
import copy
import json

spec = importlib.util.spec_from_file_location('public_import', Path(__file__).parents[1] / 'scripts/import_public_optimizer_runs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ImportTests(unittest.TestCase):
    def config_run(self, **changes):
        run = {'id': 'study', 'run_state': 'crashed', 'summary': {'_runtime': 150, 'interrupted': True},
               'url': 'https://wandb.ai/team/project/runs/study',
               'config': {'target_model': 'Qwen/Qwen3.5-4B', 'concurrency': 1, 'framework': 'vllm',
                          'sampler': 'tpe', 'sampler_settings': {'multivariate': True, 'group': True, 'seed': 11},
                          'pruner_class': 'MedianPruner', 'pruner_settings': {'n_startup_trials': 10},
                          'metrics': ['p95_e2e_latency_ms'], 'directions': ['MINIMIZE'],
                          'workload': {'num_prompts': 1000, 'max_concurrency': 128, 'seed': 2026},
                          'environment': {'hf_target_model_id': 'Org/Actual', 'model_revision': 'b' * 40,
                                          'model': '/home/private/models--Org--Actual/snapshots/' + 'a' * 40}},
               'trials': [{'number': 3, 'state': 'COMPLETE', 'params': {'batch': 16},
                           'metrics': {'source': 'MEASURED', 'output_throughput': 500,
                                       'p95_e2e_latency_ms': 200, 'mean_e2e_latency_ms': 100,
                                       'p95_ttft_ms': 40, 'mean_ttft_ms': 20, 'completed': 1000, 'failed': 0},
                           'trial_metadata': {'config.batch': 16}}]}
        run.update(changes)
        return run

    def test_short_interrupted_study_keeps_valid_trials_and_resolves_identity(self):
        result = module.convert({'runs': [self.config_run()]}, 'sha')
        study, row = result['studies'][0], result['rows'][0]
        self.assertEqual(study['durationSeconds'], 150)
        self.assertIsNone(study['terminationReason'])
        self.assertTrue(row['plotEligible'])
        self.assertEqual(row['context']['model'], 'Org/Actual')
        self.assertEqual(row['context']['revision'], 'a' * 40)
        self.assertEqual(row['context']['workload']['max_concurrency'], 128)
        self.assertEqual({c['field'] for c in study['metadataConflicts']}, {'model', 'revision', 'concurrency'})
        self.assertEqual(study['sampler'], 'tpe_multivariate')
        self.assertTrue(study['search']['sampler_settings']['group'])
        self.assertEqual(study['search']['pruner_class'], 'MedianPruner')
        self.assertEqual((row['latency'], row['meanLatency'], row['ttft'], row['meanTtft']), (200, 100, 40, 20))

    def test_unknown_and_zero_duration_remain_distinct(self):
        runs = [self.config_run(id='unknown', summary={}), self.config_run(id='zero', summary={'_runtime': 0})]
        result = module.convert({'runs': runs}, 'sha')
        self.assertEqual([s['durationSeconds'] for s in result['studies']], [None, 0])
        self.assertTrue(all(r['plotEligible'] for r in result['rows']))

    def test_diagnostic_trials_are_retained_and_never_eligible(self):
        run = self.config_run()
        trials = []
        for i, state in enumerate(['COMPLETE', 'FAIL', 'PRUNED', 'RUNNING', 'COMPLETE', 'COMPLETE']):
            t = copy.deepcopy(run['trials'][0]); t.update(number=i, state=state); trials.append(t)
        trials[4]['metrics']['completed'] = 400
        trials[5]['metrics']['failed'] = 1
        run['trials'] = trials
        result = module.convert({'runs': [run]}, 'sha')
        self.assertEqual([r['plotEligible'] for r in result['rows']], [True, False, False, False, False, False])
        self.assertTrue(all(r['diagnosticReasons'] for r in result['rows'][1:]))

    def test_dspark_native_framework_settings_override_dflash_description(self):
        for fixed in [
            {'speculative_config': {'method': 'dspark', 'model': '/root/models--Org--Draft/snapshots/' + 'c'*40}},
            {'speculative_algorithm': 'DSPARK', 'speculative_draft_model_path': '/root/models--Org--Draft/snapshots/' + 'c'*40},
            {'speculative_config': {'decoding_type': 'DSpark', 'speculative_model': '/root/models--Org--Draft/snapshots/' + 'c'*40}},
        ]:
            run = self.config_run()
            run['config']['environment'].update(fixed_server=fixed, speculative_decoding={'algorithm': 'DFLASH'})
            study = module.convert({'runs': [run]}, 'sha')['studies'][0]
            self.assertEqual(study['context']['variant'], 'DSpark')
            self.assertEqual(study['context']['draftModel'], 'Org/Draft')
            self.assertEqual(study['context']['draftRevision'], 'c'*40)
            self.assertIn('speculativeMethod', [c['field'] for c in study['metadataConflicts']])

    def test_overlaps_match_trial_number_and_preserve_legacy_row_ids(self):
        original = module.convert({'runs': [self.config_run()]}, 'old')
        original['rows'][0]['id'] = 'legacy-position-0'
        addition = module.convert({'runs': [self.config_run()]}, 'new')
        merged = module.merge_results(original, [addition, addition])
        self.assertEqual((len(merged['studies']), len(merged['rows'])), (1, 1))
        self.assertEqual(merged['rows'][0]['id'], 'legacy-position-0')
        self.assertEqual(merged['studies'][0]['sourceHashes'], ['new', 'old'])
        addition['rows'][0]['throughput'] = 999
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate trial'):
            module.merge_results(original, [addition])

    def test_duplicate_trial_numbers_are_deduplicated_or_explicitly_rejected(self):
        run = self.config_run()
        run['trials'].append(copy.deepcopy(run['trials'][0]))
        self.assertEqual(len(module.convert({'runs': [run]}, 'sha')['rows']), 1)
        run['trials'][1]['metrics']['output_throughput'] = 999
        with self.assertRaises(ValueError):
            module.convert({'runs': [run]}, 'sha')

    def test_cohorts_separate_runtime_sources_drafts_and_unknown_context(self):
        context = {'model': 'Org/M', 'revision': 'a'*40, 'gpu': 'H100', 'gpuCount': 1,
                   'engineVersion': '1', 'requestsSha256': 'r', 'draftRevision': 'x'}
        cohort = module.cohort_for(context, 'vLLM', 'MEASURED', 'a')
        self.assertEqual(cohort, module.cohort_for(context, 'vLLM', 'MEASURED', 'b'))
        for key, value in [('engineVersion', '2'), ('draftRevision', 'y')]:
            self.assertNotEqual(cohort, module.cohort_for({**context, key: value}, 'vLLM', 'MEASURED', 'a'))
        self.assertNotEqual(cohort, module.cohort_for(context, 'SGLang', 'MEASURED', 'a'))
        self.assertNotEqual(cohort, module.cohort_for(context, 'vLLM', 'MEASURED_PROXY', 'a'))
        self.assertNotEqual(module.cohort_for({}, 'vLLM', 'MEASURED', 'a'), module.cohort_for({}, 'vLLM', 'MEASURED', 'b'))

    def test_logs_are_sanitized_without_dropping_timings_or_source_link(self):
        run = self.config_run()
        run['trials'][0]['metrics'].update({'execution.server_log': 'read /home/private/log token=badsecret https://private.example/a?token=secret hf_abcdefghijk',
                                          'execution.timings_s.benchmark': 10, 'execution.command': 'private'})
        result = module.convert({'runs': [run]}, 'sha')
        text = json.dumps(result)
        for forbidden in ['/home/private', 'badsecret', 'private.example', 'hf_abcdefghijk', 'execution.command']:
            self.assertNotIn(forbidden, text)
        self.assertEqual(result['rows'][0]['metrics']['execution.timings_s.benchmark'], 10)
        self.assertEqual(result['studies'][0]['sourceUrl'], run['url'])
        self.assertIsNone(module.public_url('https://wandb.ai/team?token=secret'))

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
