"""macOS notification utility."""

import subprocess


def notify(title: str, message: str):
    """Send a macOS notification via osascript.

    Best-effort: swallows errors silently so it never breaks the pipeline.
    """
    try:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:
        pass
