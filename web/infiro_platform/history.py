"""Owner-scoped removal from workspace history; execution evidence stays immutable."""
from copy import deepcopy
from collections import Counter

from django.db import transaction
from django.utils import timezone

from .models import Job, Prepared, Recommendation
from .services import TERMINAL, owned


def require_finished(jobs):
    if jobs.exclude(state__in=TERMINAL).exists():
        raise ValueError('Finish or cancel active operations before deleting this history.')
    if jobs.filter(lease_until__gt=timezone.now()).exists():
        raise ValueError('The worker is still saving this operation. Try again shortly.')


@transaction.atomic
def delete_history(user, model, identifier):
    item = owned(model, user, identifier)
    now = timezone.now()
    if model is Recommendation:
        prepared = Prepared.objects.filter(owner=user, recommendation=item)
        jobs = Job.objects.filter(owner=user, target_id__in=[item.id, *prepared.values_list('id', flat=True)])
        require_finished(jobs)
        prepared.update(deleted_at=now)
        Recommendation.objects.filter(pk=item.pk, owner=user).update(deleted_at=now)
    elif model is Prepared or (model is Job and item.kind == 'study'):
        target = item.pk if model is Prepared else item.target_id
        jobs = Job.objects.filter(owner=user, target_id=target)
        require_finished(jobs)
        Prepared.objects.filter(pk=target, owner=user).update(deleted_at=now)
    else:
        jobs = Job.objects.filter(pk=item.pk, owner=user)
        require_finished(jobs)
    jobs.update(deleted_at=now)


def trial_key(trial):
    value = trial.get('trial_id')
    return f"{trial.get('framework')}:{value if value is not None else trial.get('configuration_id')}"


@transaction.atomic
def delete_trial(user, identifier, key):
    job = owned(Job, user, identifier)
    if job.kind != 'study':
        raise ValueError('Trial results belong to studies.')
    require_finished(Job.objects.filter(pk=job.pk, owner=user))
    summary = job.result.get('summary', {})
    trials = summary.get('trials') or summary.get('candidates', [])
    if not isinstance(key, str) or key not in {trial_key(t) for t in trials}:
        raise ValueError('Trial not found in this study.')
    job.deleted_trials = sorted(set(job.deleted_trials + [key]))
    job.save(update_fields=['deleted_trials', 'updated_at'])


def visible_result(job):
    if not job.deleted_trials:
        return job.result
    result = deepcopy(job.result)
    summary = result.get('summary', {})
    hidden = set(job.deleted_trials)
    for field in ('trials', 'candidates'):
        summary[field] = [t for t in summary.get(field, []) if trial_key(t) not in hidden]
    if 'trials' in job.result.get('summary', {}):
        counts = {}
        for trial in summary['trials']:
            counts.setdefault(trial['framework'], Counter())[trial['status']] += 1
        summary['trial_state_counts'] = {k: dict(v) for k, v in counts.items()}
    for framework, trials in summary.get('failed_trials', {}).items():
        summary['failed_trials'][framework] = [t for t in trials if trial_key({'framework': framework, **t}) not in hidden]
    # Reuse the optimizer analysis so removing a dominant trial reveals the new frontier.
    from hf_model_watch.optimization.analysis import analyze
    from hf_model_watch.optimization.contracts import SelectionPolicy, StudySummary
    from pydantic import ValidationError
    try:
        parsed = StudySummary.model_validate(summary)
        policy = SelectionPolicy.model_validate(result.get('analysis', {}).get('policy', {}))
        parsed.importance_per_objective = {}
        result['analysis'] = analyze(parsed, policy).model_dump(mode='json')
    except (ValidationError, ImportError):
        # Older summaries may not carry enough context to establish a new selection.
        result['analysis'] = {'pareto_front': [], 'best_per_objective': {}, 'selected': None,
                              'reason': 'Remaining history lacks the context or analysis dependencies needed to recompute selection.'}
    summary['importance_per_objective'] = {}
    result['history_note'] = 'Deleted trials are excluded from this view. Original artifacts and resource usage remain unchanged.'
    return result
