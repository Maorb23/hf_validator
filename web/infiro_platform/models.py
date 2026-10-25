"""Private resources. Ownership is explicit even on child resources."""

import uuid

from django.conf import settings
from django.db import models


class Owned(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Connection(Owned):
    kind = models.CharField(max_length=24)
    name = models.CharField(max_length=80)
    metadata = models.JSONField(default=dict)
    ciphertext = models.TextField()
    version = models.PositiveIntegerField(default=1)
    validation = models.CharField(max_length=24, default="unvalidated")
    validated_at = models.DateTimeField(null=True)


class Dataset(Owned):
    name = models.CharField(max_length=100)
    config = models.JSONField()
    connection_refs = models.JSONField(default=dict)
    status = models.CharField(max_length=24, default="draft")
    identity = models.JSONField(default=dict)
    preview = models.JSONField(default=list)
    error = models.TextField(blank=True)


class Recommendation(Owned):
    requirements = models.JSONField()
    connection_refs = models.JSONField(default=dict)
    status = models.CharField(max_length=24, default="queued")
    report = models.JSONField(default=dict)
    selected = models.JSONField(default=dict)
    version = models.PositiveIntegerField(default=1)


class Prepared(Owned):
    name = models.CharField(max_length=160, blank=True, default="")
    recommendation = models.ForeignKey(Recommendation, on_delete=models.PROTECT)
    dataset = models.ForeignKey(Dataset, null=True, on_delete=models.PROTECT)
    choices = models.JSONField()
    connection_refs = models.JSONField(default=dict)
    connection_versions = models.JSONField(default=dict)
    recommendation_version = models.PositiveIntegerField()
    status = models.CharField(max_length=24, default="queued")
    payload = models.JSONField(default=dict)
    intent = models.JSONField(default=dict)
    summary = models.JSONField(default=dict)
    digest = models.CharField(max_length=64, blank=True)
    approved_at = models.DateTimeField(null=True)
    invalidated = models.BooleanField(default=False)


class Job(Owned):
    kind = models.CharField(max_length=24)
    target_id = models.UUIDField()
    connection_refs = models.JSONField(default=dict)
    connection_versions = models.JSONField(default=dict)
    state = models.CharField(max_length=24, default="queued", db_index=True)
    attempts = models.PositiveIntegerField(default=0)
    available_at = models.DateTimeField()
    lease_until = models.DateTimeField(null=True)
    cancel_requested = models.BooleanField(default=False)
    error = models.TextField(blank=True)
    progress = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    controller_run = models.CharField(max_length=64, blank=True)
    dedupe_key = models.CharField(max_length=160)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["owner", "dedupe_key"], name="infiro_unique_submission")
        ]


class Artifact(Owned):
    job = models.ForeignKey(Job, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    relative_path = models.CharField(max_length=200)
    sha256 = models.CharField(max_length=64)
    size = models.PositiveIntegerField()


class AuthThrottle(models.Model):
    key = models.CharField(max_length=64, primary_key=True)
    window = models.DateTimeField()
    attempts = models.PositiveIntegerField(default=0)
