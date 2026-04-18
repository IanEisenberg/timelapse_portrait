"""macOS notification utility."""

import subprocess


def notify(title: str, message: str):
    """Send a macOS notification via osascript.

    Best-effort: swallows errors silently so it never breaks the pipeline.
    """
    try:
        safe_title = title.replace('"', '\\"')
        safe_message = message.replace('"', '\\"')
        script = f'display notification "{safe_message}" with title "{safe_title}"'
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:
        pass
