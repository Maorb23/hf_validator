"""Worker shutdown must stop isolated preparation children before lease release."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from .queue import run_once


class WorkerShutdownTests(SimpleTestCase):
    def test_shutdown_kills_and_reaps_isolated_child(self):
        job = SimpleNamespace(kind="model", cancel_requested=False, owner_id=1,
                              id="offline", progress={}, refresh_from_db=Mock())
        child = Mock(pid=12345)
        child.poll.return_value = None
        with (
            patch("infiro_platform.queue.recover"),
            patch("infiro_platform.queue.claim", return_value=job),
            patch("infiro_platform.queue.child_environment", return_value={}),
            patch("infiro_platform.queue.private_dir"),
            patch("infiro_platform.queue.directory_bytes", return_value=0),
            patch("infiro_platform.queue.subprocess.Popen", return_value=child),
            patch("infiro_platform.queue.time.sleep", side_effect=SystemExit(0)),
            patch("infiro_platform.queue.os.killpg") as kill,
            patch("infiro_platform.queue.fail") as fail,
        ):
            with self.assertRaises(SystemExit):
                run_once()
        kill.assert_called_once_with(12345, 9)
        child.wait.assert_called_once()
        fail.assert_not_called()
