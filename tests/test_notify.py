"""Tests for macOS notification utility."""

import subprocess
from unittest.mock import patch

from src.notify import notify


def test_notify_calls_osascript():
    """notify() should call osascript with the right AppleScript command."""
    with patch("src.notify.subprocess.run") as mock_run:
        notify("Test Title", "Test message body")

        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "osascript"
        assert "Test Title" in cmd[2]
        assert "Test message body" in cmd[2]


def test_notify_does_not_raise_on_failure():
    """notify() should swallow errors (best-effort)."""
    with patch("src.notify.subprocess.run", side_effect=FileNotFoundError("no osascript")):
        # Should not raise
        notify("Title", "Body")
