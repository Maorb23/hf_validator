import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from django.core.checks import run_checks
from django.test import Client, SimpleTestCase, override_settings

from .frontend import resolve_built_path


@override_settings(ALLOWED_HOSTS=["localhost", "testserver"])
class FrontendServingTests(SimpleTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        files = {
            "index.html": b"<h1>Infiro home</h1>",
            "technology/index.html": b"<h1>Technology</h1>",
            "demo/search/index.html": b"<h1>Search playground</h1>",
            "404.html": b"<h1>This path ends here.</h1>",
            "_astro/app.ABCDEF12.js": b"console.log('island');",
            "_astro/app.ABCDEF12.css": b"body{color:white}",
            "_astro/inter.ABCDEF12.woff2": b"test-font-bytes",
            "data/simulations.json": b'{"schemaVersion":1}',
            "data/inference-examples.json": b'{"kind":"simulation"}',
            "data/publication.json": b'{"rows":[]}',
            "inference-evaluation-guide.md": b"# Evaluation guide\n",
            "sitemap.xml": b"<urlset/>",
        }
        for name, data in files.items():
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        patch = override_settings(FRONTEND_DIST=self.root, WHITENOISE_ROOT=self.root)
        patch.enable()
        self.addCleanup(patch.disable)

    def body(self, response):
        if response.streaming:
            data = b"".join(response.streaming_content)
        else:
            data = response.content
        response.close()
        return data

    def test_pages_preserve_bytes_and_both_slash_forms(self):
        for route, file in [("/", "index.html"), ("/technology", "technology/index.html"), ("/technology/", "technology/index.html"), ("/demo/search?seed=29", "demo/search/index.html")]:
            with self.subTest(route=route):
                response = self.client.get(route)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response["Content-Type"], "text/html; charset=utf-8")
                self.assertEqual(response["Cache-Control"], "no-cache")
                self.assertEqual(self.body(response), (self.root / file).read_bytes())

    def test_custom_404_and_missing_asset_never_return_homepage(self):
        for route in ["/unknown", "/_astro/missing.js", "/data/missing.json"]:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(self.body(response), (self.root / "404.html").read_bytes())

    def test_assets_downloads_and_mime_types_work_without_debug(self):
        for path, content_type in [("/_astro/app.ABCDEF12.js", "text/javascript"), ("/_astro/app.ABCDEF12.css", "text/css"), ("/_astro/inter.ABCDEF12.woff2", "font/woff2"), ("/data/simulations.json", "application/json"), ("/inference-evaluation-guide.md", "text/markdown"), ("/sitemap.xml", "application/xml")]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response["Content-Type"].startswith(content_type))
            self.assertEqual(response["X-Content-Type-Options"], "nosniff")
            self.assertEqual(response["X-Frame-Options"], "DENY")
            self.assertEqual(self.body(response), (self.root / path.lstrip("/")).read_bytes())

    def test_cache_policy_etags_head_and_range_requests(self):
        response = self.client.get("/_astro/app.ABCDEF12.js")
        self.assertIn("immutable", response["Cache-Control"])
        etag = response["ETag"]
        self.body(response)
        self.assertEqual(self.client.get("/_astro/app.ABCDEF12.js", HTTP_IF_NONE_MATCH=etag).status_code, 304)
        response = self.client.head("/_astro/app.ABCDEF12.js")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.body(response), b"")
        response = self.client.get("/_astro/inter.ABCDEF12.woff2", HTTP_RANGE="bytes=0-3")
        self.assertEqual(response.status_code, 206)
        self.assertEqual(self.body(response), b"test")
        response = self.client.get("/data/simulations.json")
        self.assertNotIn("immutable", response["Cache-Control"])
        self.assertEqual(json.loads(self.body(response)), {"schemaVersion": 1})

    def test_bad_hosts_are_rejected_for_assets_and_pages(self):
        for route in ["/", "/_astro/app.ABCDEF12.js", "/data/simulations.json"]:
            self.assertEqual(self.client.get(route, HTTP_HOST="untrusted.example").status_code, 400)

    def test_traversal_and_symlink_escape_are_rejected(self):
        for path in ["/../index.html", "/%2e%2e/index.html", "/foo\\bar", "/.env", "/%00"]:
            self.assertIsNone(resolve_built_path(self.root, path))
        with TemporaryDirectory() as external:
            target = Path(external) / "index.html"
            target.write_text("private")
            (self.root / "escape").symlink_to(external, target_is_directory=True)
            self.assertIsNone(resolve_built_path(self.root, "/escape"))

    def test_missing_build_is_actionable_and_readiness_fails(self):
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        (self.root / "index.html").unlink()
        response = self.client.get("/")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"npm run build", self.body(response))
        self.assertEqual(self.client.get("/healthz").status_code, 503)
        self.assertIn("infiro.E001", [issue.id for issue in run_checks()])

    def test_build_check_detects_missing_references(self):
        self.assertEqual(run_checks(), [])
        (self.root / "index.html").write_text('<script src="/_astro/missing.js"></script>')
        self.assertIn("infiro.E002", [issue.id for issue in run_checks()])

    def test_writes_are_not_supported(self):
        self.assertEqual(self.client.post("/").status_code, 405)
        self.assertEqual(self.client.post("/_astro/app.ABCDEF12.js").status_code, 405)


@override_settings(ALLOWED_HOSTS=["testserver"])
class ExistingBuildParityTests(SimpleTestCase):
    def test_all_built_pages_are_served_unchanged(self):
        from django.conf import settings

        for file in settings.FRONTEND_DIST.rglob("index.html"):
            if file.parent.name == "app":
                continue  # The private app shell now requires a session.
            route = "/" + file.parent.relative_to(settings.FRONTEND_DIST).as_posix().strip(".")
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, route)
            self.assertEqual(b"".join(response.streaming_content), file.read_bytes(), route)
            response.close()


class ProductionSettingsTests(SimpleTestCase):
    def load_settings(self, **overrides):
        env = os.environ.copy()
        for key in ["DJANGO_SECRET_KEY", "DJANGO_ALLOWED_HOSTS", "DJANGO_SECURE_SSL_REDIRECT", "DJANGO_TRUST_PROXY_SSL_HEADER"]:
            env.pop(key, None)
        env.update(INFIRO_ENV="production", **overrides)
        return subprocess.run(
            [sys.executable, "-c", "import json; import infiro_site.settings as s; print(json.dumps({'debug':s.DEBUG,'ssl':s.SECURE_SSL_REDIRECT,'hsts':s.SECURE_HSTS_SECONDS,'databases':s.DATABASES}))"],
            cwd=Path(__file__).resolve().parent.parent,
            env=env, text=True, capture_output=True, check=False,
        )

    def test_production_requires_explicit_secret(self):
        result = self.load_settings(DJANGO_ALLOWED_HOSTS="example.invalid")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires DJANGO_SECRET_KEY", result.stderr)

    def test_production_requires_explicit_nonwildcard_hosts(self):
        for hosts in ["", "*"]:
            result = self.load_settings(DJANGO_SECRET_KEY="test-only-" * 8, DJANGO_ALLOWED_HOSTS=hosts)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("explicit DJANGO_ALLOWED_HOSTS", result.stderr)

    def test_production_defaults_to_https_with_private_database(self):
        result = self.load_settings(DJANGO_SECRET_KEY="test-only-" * 8, DJANGO_ALLOWED_HOSTS="example.invalid")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = json.loads(result.stdout)
        self.assertFalse(result["debug"])
        self.assertTrue(result["ssl"])
        self.assertEqual(result["hsts"], 31536000)
        self.assertEqual(result["databases"]["default"]["ENGINE"], "django.db.backends.sqlite3")
