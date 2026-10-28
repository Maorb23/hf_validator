"""History deletion must persist, respect ownership and never interrupt live work."""
import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone

from .models import Job, Prepared, Recommendation
from .contracts import Choices


@override_settings(ALLOWED_HOSTS=['testserver'])
class HistoryTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user('history-owner')
        self.other = get_user_model().objects.create_user('another-owner')
        self.client.force_login(self.user)
        self.rec = Recommendation.objects.create(owner=self.user, requirements={'request': 'test'})
        self.prepared = Prepared.objects.create(owner=self.user, recommendation=self.rec,
            recommendation_version=1, choices=Choices().model_dump(), status='prepared')
        self.job = Job.objects.create(owner=self.user, kind='study', target_id=self.prepared.id,
            available_at=timezone.now(), state='completed', dedupe_key='history-study')

    def delete(self, route, payload=None):
        return self.client.delete('/app/api/'+route, json.dumps(payload or {}), content_type='application/json')

    def test_recommendation_deletes_related_history_and_disappears_after_reload(self):
        discovery = Job.objects.create(owner=self.user, kind='discover', target_id=self.rec.id,
            available_at=timezone.now(), state='failed', dedupe_key='history-discovery')
        response = self.delete('recommendations/'+str(self.rec.id))
        self.assertEqual(response.status_code, 200)
        for item in [self.rec, self.prepared, self.job, discovery]:
            item.refresh_from_db(); self.assertIsNotNone(item.deleted_at)
        data = self.client.get('/app/api/bootstrap').json()
        self.assertEqual(data['recommendations'], [])
        self.assertEqual(data['prepared'], [])
        self.assertEqual(data['jobs'], [])
        for route in ['recommendations/'+str(self.rec.id), 'jobs/'+str(self.job.id), 'prepared/'+str(self.prepared.id)]:
            self.assertEqual(self.client.get('/app/api/'+route).status_code, 404)
        self.assertTrue(Job.objects.filter(pk=self.job.pk).exists(), 'Original execution evidence remains')

    def test_delete_study_preserves_recommendation_and_other_users(self):
        foreign = Recommendation.objects.create(owner=self.other, requirements={})
        self.assertEqual(self.delete('recommendations/'+str(foreign.id)).status_code, 404)
        self.assertEqual(self.delete('jobs/'+str(self.job.id)).status_code, 200)
        self.rec.refresh_from_db(); self.assertIsNone(self.rec.deleted_at)
        self.assertIsNotNone(Prepared.objects.get(pk=self.prepared.id).deleted_at)
        self.assertIsNone(Recommendation.objects.get(pk=foreign.id).deleted_at)

    def test_active_or_leased_operations_cannot_be_deleted(self):
        for state in ['queued', 'running', 'attention_required']:
            Job.objects.filter(pk=self.job.pk).update(state=state)
            self.assertEqual(self.delete('jobs/'+str(self.job.id)).status_code, 400)
            self.assertEqual(self.delete('recommendations/'+str(self.rec.id)).status_code, 400)
            self.assertEqual(self.delete('prepared/'+str(self.prepared.id)).status_code, 400)
        Job.objects.filter(pk=self.job.pk).update(state='completed', lease_until=timezone.now()+timedelta(minutes=1))
        self.assertEqual(self.delete('jobs/'+str(self.job.id)).status_code, 400)
        self.rec.refresh_from_db(); self.assertIsNone(self.rec.deleted_at)

    def test_terminal_background_operations_can_be_deleted_but_not_retried_afterward(self):
        self.job.kind = 'discover'; self.job.target_id = self.rec.id; self.job.state = 'cancelled'; self.job.save()
        self.assertEqual(self.delete('jobs/'+str(self.job.id)).status_code, 200)
        self.assertEqual(self.client.post('/app/api/jobs/'+str(self.job.id)+'/retry', '{}', content_type='application/json').status_code, 404)
        self.rec.refresh_from_db(); self.assertIsNone(self.rec.deleted_at)

    def test_trial_deletion_is_owned_terminal_and_preserves_original_result(self):
        self.job.result = {'summary': {'trials': [{'framework': 'vllm', 'trial_id': '1', 'status': 'COMPLETE'}], 'candidates': []}}
        self.job.save()
        route = 'jobs/'+str(self.job.id)+'/trials'
        Job.objects.filter(pk=self.job.pk).update(state='running')
        self.assertEqual(self.delete(route, {'trial_key': 'vllm:1'}).status_code, 400)
        Job.objects.filter(pk=self.job.pk).update(state='completed')
        self.client.force_login(self.other)
        self.assertEqual(self.delete(route, {'trial_key': 'vllm:1'}).status_code, 404)
        self.client.force_login(self.user)
        self.assertEqual(self.delete(route, {'trial_key': 'vllm:99'}).status_code, 400)
        self.assertEqual(self.delete(route, {'trial_key': 'vllm:1'}).status_code, 200)
        self.job.refresh_from_db()
        self.assertEqual(len(self.job.result['summary']['trials']), 1)
        result = self.client.get('/app/api/jobs/'+str(self.job.id)).json()['result']
        self.assertEqual(result['summary']['trials'], [])
        self.assertEqual(self.job.deleted_trials, ['vllm:1'])

    def test_removing_dominant_trial_recomputes_frontier_and_selection(self):
        from hf_model_watch.optimization.analysis import analyze
        from hf_model_watch.optimization.contracts import Candidate, SelectionPolicy, StudyContext, StudySummary
        context = StudyContext(model_id='org/model', revision='a'*40, gpu_type='gpu-h100-sxm',
            workload={'input_tokens': 16, 'output_tokens': 8}, concurrency=8)
        candidates = [Candidate(framework='vllm', configuration_id=str(i), trial_id=str(i),
            configuration={'batch': i+1}, metrics={'output_throughput': throughput, 'mean_e2e_latency_ms': latency})
            for i, throughput, latency in [(0,100,10),(1,50,20)]]
        summary = StudySummary(study_reference='history', context=context, state='completed', candidates=candidates,
            trials=[{'framework':'vllm','trial_id':str(i),'status':'COMPLETE'} for i in range(2)])
        policy = SelectionPolicy(name='throughput_first')
        self.job.result = {'summary': summary.model_dump(mode='json'), 'analysis': analyze(summary,policy).model_dump(mode='json')}
        self.job.save()
        self.assertEqual(self.delete('jobs/'+str(self.job.id)+'/trials', {'trial_key':'vllm:0'}).status_code,200)
        result = self.client.get('/app/api/jobs/'+str(self.job.id)).json()['result']
        self.assertEqual([c['trial_id'] for c in result['analysis']['pareto_front']], ['1'])
        self.assertEqual(result['analysis']['selected']['trial_id'], '1')
        self.assertEqual(result['analysis']['best_per_objective']['output_throughput']['trial_id'], '1')
        self.assertEqual(result['summary']['trial_state_counts'], {'vllm':{'COMPLETE':1}})
        self.job.refresh_from_db()
        self.assertEqual(len(self.job.result['summary']['candidates']),2)

    def test_deletion_requires_authenticated_csrf_protected_request(self):
        from django.test import Client
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        self.assertEqual(client.delete('/app/api/jobs/'+str(self.job.id)).status_code,403)
        self.client.logout()
        self.assertEqual(self.delete('jobs/'+str(self.job.id)).status_code,401)
