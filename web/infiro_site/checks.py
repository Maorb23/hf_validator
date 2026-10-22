from html.parser import HTMLParser

from django.conf import settings
from django.core.checks import Error, register

from .frontend import resolve_built_path


class LocalReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"src", "href", "component-url", "renderer-url"} and value and value.startswith("/") and not value.startswith("//"):
                self.paths.add(value)


@register()
def frontend_build_check(app_configs, **kwargs):
    root = settings.FRONTEND_DIST
    if not (root / "index.html").is_file():
        return [Error("Infiro frontend build is missing.", hint="Run npm ci and npm run build from web/, or set INFIRO_FRONTEND_DIST to an existing build.", id="infiro.E001")]
    expected = ["/404.html", "/data/simulations.json", "/data/inference-examples.json", "/data/publication.json"]
    references = set(expected)
    for html in root.rglob("*.html"):
        parser = LocalReferences()
        parser.feed(html.read_text(encoding="utf-8"))
        references.update(parser.paths)
    missing = sorted(path for path in references if resolve_built_path(root, path) is None)
    if missing:
        return [Error(f"Frontend build has missing local references: {', '.join(missing[:10])}", hint="Rebuild the complete Astro artifact before starting Django.", id="infiro.E002")]
    return []
