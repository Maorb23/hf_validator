import fcntl
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from infiro_platform.queue import run_once


class Command(BaseCommand):
    help = (
        "Run the single durable platform worker. No GPU jobs without exact approval "
        "and a secure live adapter."
    )

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--prepare-only", action="store_true",
                            help="Process preparation only; never claim or recover GPU studies.")

    def handle(self, *args, **options):
        lock_path = Path(str(settings.DATABASES["default"]["NAME"]) + ".platform-worker.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise CommandError("Another platform worker holds this database lease.") from None
            self.stdout.write("Infiro worker ready; durable queue, isolated child processes.")
            tracking = settings.INFIRO_WANDB
            self.stdout.write(
                "W&B: " + (f"{tracking.get('entity', '(default entity)')}/{tracking['project']} "
                           "(verified during preparation/submission)"
                           if tracking.get("project") else "not configured in this worker")
            )
            if options["prepare_only"]:
                self.stdout.write("Preparation only: approved GPU studies will not be processed.")
            while True:
                busy = run_once(kinds={"prepare"}) if options["prepare_only"] else run_once()
                if options["once"]:
                    return
                if not busy:
                    time.sleep(1)
