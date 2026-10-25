"""Read-only W&B SecretStash check; never create runs or GPU jobs."""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from infiro_platform.models import Job, Prepared
from infiro_platform.nebius_provider import NebiusSDKProvider, pin_secret_references
from infiro_platform.security import unseal
from infiro_platform.services import connections


class Command(BaseCommand):
    help = "Verify W&B secret access and print pinned config, without starting a run."

    def add_arguments(self, parser):
        parser.add_argument("--prepared", required=True, help="Prepared configuration UUID")

    def handle(self, *args, **options):
        tracking = dict(settings.INFIRO_WANDB)
        if not tracking.get("project") or not tracking.get("secret_version_id"):
            raise CommandError("Set INFIRO_WANDB with project and secret_version_id first.")
        try:
            prepared = Prepared.objects.get(pk=options["prepared"])
            found = connections(prepared.owner, prepared.connection_refs, required=("nebius_jobs",))
            values = unseal(found["nebius_jobs"])
            variables = {
                "NEBIUS_PRIVATE_KEY": values.get("private_key", ""),
                "NEBIUS_IAM_TOKEN": values.get("iam_token", ""),
            }
            with patch.dict(os.environ, variables):
                pinned = pin_secret_references(
                    {
                        "nebius_jobs": found["nebius_jobs"],
                        "wandb": SimpleNamespace(metadata=tracking),
                    }
                )["wandb"]
                provider = NebiusSDKProvider(Job(owner=prepared.owner), prepared, found)
                try:
                    provider._verify_secret(
                        SimpleNamespace(metadata=pinned), {}, required_keys=("WANDB_API_KEY",)
                    )
                finally:
                    provider.sdk.sync_close()
        except Exception as exc:
            # Provider exception strings may contain credentials. Preserve only type.
            raise CommandError(
                f"W&B secret verification failed ({type(exc).__name__}). "
                "Check the project, secret reference, and payload-read permission."
            ) from None
        self.stdout.write("W&B secret version and WANDB_API_KEY payload-read verified.")
        self.stdout.write("Use this INFIRO_WANDB for both Django and platform_worker:")
        self.stdout.write(json.dumps(pinned))
        self.stdout.write("W&B API authentication is confirmed only when a remote run connects.")
