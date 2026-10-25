"""Print the existing optimizer's stored execution logs for a web study."""

import boto3
from botocore.config import Config
from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from llm_optimizer.remote.adapters import S3Artifacts

from infiro_platform.models import Job, Prepared
from infiro_platform.security import unseal
from infiro_platform.services import connections


def provider_jobs(job):
    table = "infiro_ctl_" + job.id.hex + "_jobs"
    with connection.cursor() as cursor:
        cursor.execute(
            f'SELECT framework, nebius_id FROM "{table}" WHERE run_id = %s',
            [job.controller_run],
        )
        return dict(cursor.fetchall())


class Command(BaseCommand):
    help = "Print saved benchmark log tails for every framework in the latest web study."

    def add_arguments(self, parser):
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--latest", action="store_true", help="Use the latest launched study.")
        group.add_argument("--study", help="Use this web study UUID instead.")

    def handle(self, *args, **options):
        studies = Job.objects.filter(kind="study").exclude(controller_run="")
        if options["study"]:
            studies = studies.filter(pk=options["study"])
        job = studies.order_by("-created_at").first()
        if job is None:
            raise CommandError("No launched study found.")
        prepared = Prepared.objects.get(pk=job.target_id, owner=job.owner)
        found = connections(job.owner, job.connection_refs, required=("storage",))
        storage = found["storage"]
        values = unseal(storage)
        client = boto3.client(
            "s3", aws_access_key_id=values["access_key_id"],
            aws_secret_access_key=values["secret_access_key"],
            aws_session_token=values.get("session_token"),
            region_name=storage.metadata["region"],
            endpoint_url=storage.metadata.get("endpoint") or None,
            config=Config(connect_timeout=10, read_timeout=20, retries={"max_attempts": 1}),
        )
        store = S3Artifacts(prepared.payload["bucket"], prepared.payload["prefix"], client=client)
        identities = provider_jobs(job)
        self.stdout.write(f"Study {job.id} ({job.state}); last 16 KiB per framework")
        try:
            for framework in prepared.choices["frameworks"]:
                identity = identities.get(framework) or "no provider ID"
                self.stdout.write(f"\n=== {framework}: {identity} ===")
                text = store.execution_log_tail(job.controller_run, framework)
                self.stdout.write(text or "No execution.log uploaded yet.")
        finally:
            client.close()
