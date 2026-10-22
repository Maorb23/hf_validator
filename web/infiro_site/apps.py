from django.apps import AppConfig


class InfiroSiteConfig(AppConfig):
    name = "infiro_site"

    def ready(self):
        from . import checks  # noqa: F401
