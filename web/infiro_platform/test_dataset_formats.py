"""Format adapters share the existing frozen sampler and execution contract."""
from types import SimpleNamespace

from django.test import SimpleTestCase

from .contracts import DatasetConfig
from .demo import FixtureTokenizer, fixture_manifest
from . import test_dataset_safety
from .workload_runtime import verify_frozen_workload
from .workloads import WorkloadLimitError, adapt_record, materialize


class DatasetFormatTests(SimpleTestCase):
    def freeze(self, source_format, rows):
        config, _ = fixture_manifest()
        if source_format is None:
            config.pop('format', None)
        else:
            config['format'] = source_format
        api = SimpleNamespace(
            dataset_info=lambda *a, **k: SimpleNamespace(
                sha='a' * 40, siblings=[SimpleNamespace(size=4096, rfilename='data.json')]),
            model_info=lambda *a, **k: SimpleNamespace(sha='a' * 40, siblings=[]),
        )
        return materialize(config, api=api, loader=lambda *a, **k: rows,
                           tokenizer=FixtureTokenizer())

    def test_normalization(self):
        for extra, expected in [('', 'Task'), ('Context', 'Task\n\nContext')]:
            self.assertEqual(adapt_record({'instruction': 'Task', 'input': extra},
                                         'alpaca', 0)['conversations'][0]['value'], expected)
        row = {'messages': [{'role': 'system', 'content': 'Rules'},
                            {'role': 'user', 'content': 'First'},
                            {'role': 'user', 'content': 'Second'}]}
        self.assertEqual(adapt_record(row, 'messages', 0)['conversations'][0]['value'], 'First')

    def test_invalid_records_are_clear(self):
        for fmt, row in [('alpaca', {}), ('alpaca', {'instruction': 'x', 'input': None}),
                         ('messages', {}), ('messages', {'messages': []}),
                         ('messages', {'messages': [{'role': 'user', 'content': []}]}),
                         ('messages', {'messages': [{'role': 'assistant', 'content': 'x'}]})]:
            with self.subTest(fmt=fmt, row=row):
                with self.assertRaisesRegex(WorkloadLimitError, f'Malformed {fmt} record at row 1'):
                    self.freeze(fmt, [row])

    def test_legacy_identity_and_default(self):
        config, expected = fixture_manifest()
        self.assertEqual(DatasetConfig(**config).format, 'sharegpt')
        rows = [{'conversations': [{'from': 'human', 'value': f'Local fixture prompt {i}'},
                                  {'from': 'gpt', 'value': 'Fixture response'}]} for i in range(12)]
        self.assertEqual(self.freeze('sharegpt', rows), expected)
        self.assertEqual(self.freeze(None, rows), expected)
        config.pop('format')
        self.assertEqual(DatasetConfig(**config).format, 'sharegpt')
        self.assertNotIn('source_format', expected['dataset'])
        with self.assertRaises(ValueError):
            DatasetConfig(**(config | {'format': 'csv'}))

    def test_deterministic_identity_and_frozen_execution(self):
        manifests = []
        for fmt in ('alpaca', 'messages'):
            rows = [({'instruction': f'Prompt {i}'} if fmt == 'alpaca' else
                     {'messages': [{'role': 'user', 'content': f'Prompt {i}'}]}) for i in range(12)]
            manifest = self.freeze(fmt, rows)
            self.assertEqual(manifest, self.freeze(fmt, rows))
            self.assertEqual(manifest['dataset']['source_format'], fmt)
            self.assertEqual(manifest['dataset']['adapter_version'], 1)
            for framework in ('vllm', 'sglang', 'trtllm'):
                payload = test_dataset_safety.FrozenRuntimeTests().payload(framework)
                payload['request']['workload_manifest'] = manifest
                result = verify_frozen_workload(payload)
                self.assertEqual(result['sha256'], manifest['sha256'])
            manifests.append(manifest)
        self.assertEqual(manifests[0]['requests'], manifests[1]['requests'])
        self.assertNotEqual(manifests[0]['sha256'], manifests[1]['sha256'])
