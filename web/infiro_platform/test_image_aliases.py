import hashlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from .image_aliases import verified_alias


class ImageAliasTests(SimpleTestCase):
    def setUp(self):
        self.raw = b'{"schemaVersion":2}'
        self.pinned = "registry/image@sha256:" + hashlib.sha256(self.raw).hexdigest()
        self.alias = "cr.eu-north1.nebius.cloud/example/vllm:test"
        self.sdk = Mock()
        self.sdk.get_token_sync.return_value = SimpleNamespace(token="private-token")
        self.client = Mock()
        self.client.get.return_value = SimpleNamespace(status_code=200, content=self.raw)
        patcher = patch("infiro_platform.image_aliases.httpx.Client")
        self.addCleanup(patcher.stop)
        patcher.start().return_value.__enter__.return_value = self.client

    def test_matching_manifest_is_accepted_and_changed_tag_is_rejected(self):
        aliases = {self.pinned: self.alias}
        self.assertEqual(verified_alias(self.pinned, aliases, self.sdk), self.alias)
        self.client.get.return_value.content = b"changed manifest"
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            verified_alias(self.pinned, aliases, self.sdk)

    def test_missing_alias_and_untrusted_registry_fail_before_credentials(self):
        for aliases in ({}, {self.pinned: "evil.example/image:tag"}):
            with self.assertRaises(ValueError):
                verified_alias(self.pinned, aliases, self.sdk)
        self.sdk.get_token_sync.assert_not_called()

    def test_bearer_challenge_and_untrusted_auth_endpoint(self):
        challenge = SimpleNamespace(
            status_code=401,
            headers={
                "www-authenticate": 'Bearer realm="https://cr.eu-north1.nebius.cloud/token",service="registry"'
            },
        )
        self.client.get.side_effect = [
            challenge,
            SimpleNamespace(status_code=200, json=lambda: {"token": "scoped-token"}),
            SimpleNamespace(status_code=200, content=self.raw),
        ]
        self.assertEqual(
            verified_alias(self.pinned, {self.pinned: self.alias}, self.sdk), self.alias
        )
        challenge.headers["www-authenticate"] = 'Bearer realm="https://evil.example/token"'
        self.client.get.side_effect = [challenge]
        with self.assertRaisesRegex(ValueError, "unexpected authentication"):
            verified_alias(self.pinned, {self.pinned: self.alias}, self.sdk)
