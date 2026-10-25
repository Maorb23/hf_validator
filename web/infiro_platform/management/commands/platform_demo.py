from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from infiro_platform.demo import seed_prepared


class Command(BaseCommand):
    help = (
        "Seed an explicitly labeled offline dry-run fixture for an existing local user. "
        "Does not approve or launch it."
    )

    def add_arguments(self, parser):
        parser.add_argument("--username", required=True)

    def handle(self, *args, **options):
        try:
            user = get_user_model().objects.get(username=options["username"])
        except get_user_model().DoesNotExist:
            raise CommandError("Create this user at /app/register first.") from None
        prepared = seed_prepared(user)
        self.stdout.write("Offline fixture created. No provider calls or GPU allocation.")
        self.stdout.write(
            "Review and explicitly approve at http://127.0.0.1:8000/app/approval/"
            + str(prepared.id)
        )
