# Autonomous Monthly Sync & Deploy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate the monthly Google Photos sync, face processing, video generation, and website deployment pipeline with macOS notifications on failures.

**Architecture:** A new `GooglePhotosDownloader` module handles incremental photo sync from Google Photos. A new `cmd_auto` orchestrator in `align_faces.py` chains sync → process → video → resize → push. A `notify()` utility sends macOS notifications. Launchd runs the pipeline monthly via a wrapper script.

**Tech Stack:** Python 3.11, google-api-python-client, google-auth-oauthlib, ffmpeg (subprocess), launchd, osascript

**Spec:** `docs/superpowers/specs/2026-04-08-autonomous-monthly-sync-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/notify.py` | Create | `notify(title, message)` — macOS notification via osascript |
| `src/google_photos.py` | Create | `GooglePhotosDownloader` — OAuth auth, album listing, incremental download |
| `align_faces.py` | Modify | Add `sync`, `auto` subcommands; `resize_video()` helper |
| `config.yaml` | Modify | Add `google_photos` and `auto` config sections |
| `pyproject.toml` | Modify | Add google-auth dependencies |
| `run_monthly.sh` | Create | Wrapper script for launchd (sets PATH, logs output) |
| `com.ian.timelapse-portrait.plist` | Create | Launchd plist for monthly scheduling |
| `CLAUDE.md` | Modify | Document new commands and automation |
| `tests/test_google_photos.py` | Create | Test sync filtering logic |
| `tests/test_notify.py` | Create | Test notification function |

---

### Task 1: Add dependencies and config sections

**Files:**
- Modify: `pyproject.toml:9-20`
- Modify: `config.yaml`

- [ ] **Step 1: Add Google API dependencies to pyproject.toml**

Add to `[tool.poetry.dependencies]` in `pyproject.toml`:

```toml
google-api-python-client = "*"
google-auth-oauthlib = "*"
```

- [ ] **Step 2: Add config sections to config.yaml**

Append to end of `config.yaml`:

```yaml

# Google Photos sync settings
google_photos:
  album_name: "Ma Face, Straight"
  credentials_path: "credentials.json"
  token_path: "token.json"

# Autonomous pipeline settings
auto:
  website_repo: "/Users/ian/Projects/IanEisenberg.github.io"
  website_video_path: "img/face-timelapse-small.mp4"
  resize_width: 720
  resize_height: 720
  resize_crf: 28
```

- [ ] **Step 3: Install dependencies**

Run: `poetry install`
Expected: installs `google-api-python-client` and `google-auth-oauthlib` (plus transitive deps like `google-auth`)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml poetry.lock config.yaml
git commit -m "feat: add Google Photos and auto pipeline config and dependencies"
```

---

### Task 2: Notification utility

**Files:**
- Create: `src/notify.py`
- Create: `tests/test_notify.py`

- [ ] **Step 1: Write the test**

Create `tests/test_notify.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_notify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.notify'`

- [ ] **Step 3: Write the implementation**

Create `src/notify.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_notify.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/notify.py tests/test_notify.py
git commit -m "feat: add macOS notification utility"
```

---

### Task 3: Google Photos sync module

**Files:**
- Create: `src/google_photos.py`
- Create: `tests/test_google_photos.py`

- [ ] **Step 1: Write tests for filtering logic**

Create `tests/test_google_photos.py`:

```python
"""Tests for Google Photos sync module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

from src.google_photos import GooglePhotosDownloader


def test_get_new_items_filters_existing(tmp_path):
    """Items whose filename already exists in output_dir should be excluded."""
    # Create some "existing" files
    (tmp_path / "IMG_001.jpg").touch()
    (tmp_path / "IMG_002.HEIC").touch()

    # Simulate API response with 3 items, 2 of which already exist
    api_items = [
        {"filename": "IMG_001.jpg", "baseUrl": "https://example.com/1", "id": "1"},
        {"filename": "IMG_002.HEIC", "baseUrl": "https://example.com/2", "id": "2"},
        {"filename": "IMG_003.jpg", "baseUrl": "https://example.com/3", "id": "3"},
    ]

    downloader = GooglePhotosDownloader.__new__(GooglePhotosDownloader)
    downloader.output_dir = str(tmp_path)

    new_items = downloader._filter_new_items(api_items)

    assert len(new_items) == 1
    assert new_items[0]["filename"] == "IMG_003.jpg"


def test_get_new_items_case_insensitive_extension(tmp_path):
    """Filtering should match regardless of extension case."""
    (tmp_path / "IMG_001.JPG").touch()

    api_items = [
        {"filename": "IMG_001.jpg", "baseUrl": "https://example.com/1", "id": "1"},
    ]

    downloader = GooglePhotosDownloader.__new__(GooglePhotosDownloader)
    downloader.output_dir = str(tmp_path)

    new_items = downloader._filter_new_items(api_items)

    # IMG_001.JPG exists locally, IMG_001.jpg from API should be filtered
    assert len(new_items) == 0


def test_get_new_items_empty_dir(tmp_path):
    """All items are new if output_dir is empty."""
    api_items = [
        {"filename": "IMG_001.jpg", "baseUrl": "https://example.com/1", "id": "1"},
        {"filename": "IMG_002.jpg", "baseUrl": "https://example.com/2", "id": "2"},
    ]

    downloader = GooglePhotosDownloader.__new__(GooglePhotosDownloader)
    downloader.output_dir = str(tmp_path)

    new_items = downloader._filter_new_items(api_items)

    assert len(new_items) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_google_photos.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.google_photos'`

- [ ] **Step 3: Write the implementation**

Create `src/google_photos.py`:

```python
"""Google Photos album sync module."""

import os
from typing import List, Dict, Optional

import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]


class GooglePhotosDownloader:
    """Downloads new photos from a Google Photos album."""

    def __init__(
        self,
        credentials_path: str,
        token_path: str,
        album_name: str,
        output_dir: str,
        verbose: bool = True
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.album_name = album_name
        self.output_dir = output_dir
        self.verbose = verbose
        self.service = None

    def authenticate(self):
        """Authenticate with Google Photos API.

        Uses existing token.json if valid, otherwise runs interactive OAuth flow.
        """
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_path}\n"
                        "Download it from Google Cloud Console (OAuth 2.0 Client ID, Desktop type)."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save token for next run
            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        self.service = build("photoslibrary", "v1", credentials=creds, static_discovery=False)

    def _find_album_id(self) -> str:
        """Find album ID by name."""
        page_token = None
        while True:
            results = self.service.albums().list(
                pageSize=50, pageToken=page_token
            ).execute()

            for album in results.get("albums", []):
                if album["title"] == self.album_name:
                    return album["id"]

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        raise ValueError(f"Album not found: '{self.album_name}'")

    def _list_album_items(self, album_id: str) -> List[Dict]:
        """List all media items in an album (handles pagination)."""
        items = []
        page_token = None

        while True:
            body = {"albumId": album_id, "pageSize": 100}
            if page_token:
                body["pageToken"] = page_token

            results = self.service.mediaItems().search(body=body).execute()

            items.extend(results.get("mediaItems", []))

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        return items

    def _filter_new_items(self, items: List[Dict]) -> List[Dict]:
        """Filter out items that already exist locally (case-insensitive on stem)."""
        existing_stems = set()
        for f in os.listdir(self.output_dir):
            stem = os.path.splitext(f)[0].lower()
            existing_stems.add(stem)

        new_items = []
        for item in items:
            stem = os.path.splitext(item["filename"])[0].lower()
            if stem not in existing_stems:
                new_items.append(item)

        return new_items

    def _download_item(self, item: Dict) -> bool:
        """Download a single media item at original quality."""
        filename = item["filename"]
        # Append =d for original quality download
        url = item["baseUrl"] + "=d"

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, "wb") as f:
                f.write(response.content)

            if self.verbose:
                print(f"  Downloaded: {filename}")
            return True

        except Exception as e:
            print(f"  Failed to download {filename}: {e}")
            return False

    def sync(self) -> int:
        """Sync new photos from the album. Returns count of new photos downloaded."""
        if self.service is None:
            self.authenticate()

        os.makedirs(self.output_dir, exist_ok=True)

        if self.verbose:
            print(f"Looking for album: '{self.album_name}'...")

        album_id = self._find_album_id()

        if self.verbose:
            print(f"Found album. Listing items...")

        all_items = self._list_album_items(album_id)

        if self.verbose:
            print(f"Album contains {len(all_items)} items total")

        new_items = self._filter_new_items(all_items)

        if not new_items:
            if self.verbose:
                print("No new photos to download")
            return 0

        if self.verbose:
            print(f"Downloading {len(new_items)} new photos...")

        downloaded = 0
        for item in new_items:
            if self._download_item(item):
                downloaded += 1

        if self.verbose:
            print(f"Downloaded {downloaded}/{len(new_items)} photos")

        return downloaded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_google_photos.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/google_photos.py tests/test_google_photos.py
git commit -m "feat: add Google Photos sync module"
```

---

### Task 4: Add `sync` CLI command

**Files:**
- Modify: `align_faces.py` — add import + `cmd_sync` + subparser

- [ ] **Step 1: Add import at top of align_faces.py**

After the existing imports (line 6), add:

```python
from src.google_photos import GooglePhotosDownloader
from src.notify import notify
```

- [ ] **Step 2: Add cmd_sync function**

Add after `cmd_retry` (around line 338):

```python
def cmd_sync(args, config: dict):
    """Sync new photos from Google Photos album."""
    gp_config = config.get('google_photos', {})
    album_name = gp_config.get('album_name', 'Ma Face, Straight')
    credentials_path = gp_config.get('credentials_path', 'credentials.json')
    token_path = gp_config.get('token_path', 'token.json')
    output_dir = config['paths']['original_faces']

    print(f"Syncing from Google Photos album: '{album_name}'...")

    downloader = GooglePhotosDownloader(
        credentials_path=credentials_path,
        token_path=token_path,
        album_name=album_name,
        output_dir=output_dir,
        verbose=config['processing']['verbose']
    )

    try:
        downloaded = downloader.sync()
        print(f"\nSync complete! Downloaded {downloaded} new photos.")
    except Exception as e:
        print(f"\nSync failed: {e}")
        raise
```

- [ ] **Step 3: Register `sync` subparser and command**

Add subparser after the `retry` subparser registration:

```python
    # Sync command
    subparsers.add_parser('sync', help='Sync new photos from Google Photos album')
```

Add to the `commands` dict:

```python
        'sync': cmd_sync,
```

- [ ] **Step 4: Smoke test**

Run: `poetry run timelapse sync --help`
Expected: shows help without error. (Actual sync will fail until credentials.json exists.)

- [ ] **Step 5: Commit**

```bash
git add align_faces.py
git commit -m "feat: add sync CLI command"
```

---

### Task 5: Add `auto` CLI command

**Files:**
- Modify: `align_faces.py` — add `resize_video()` helper, `cmd_auto`, and subparser

- [ ] **Step 1: Add resize_video helper function**

Add after the imports in `align_faces.py`:

```python
import subprocess as sp
import re
from datetime import date


def resize_video(input_path: str, output_path: str, width: int, height: int, crf: int):
    """Resize a video using ffmpeg.

    Args:
        input_path: Path to source video
        output_path: Path to write resized video
        width: Target width
        height: Target height
        crf: Constant Rate Factor (higher = smaller file, lower quality)
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-an",
        output_path
    ]
    result = sp.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def update_last_updated_date(file_path: str):
    """Update the _Last updated_ date in a markdown file to today."""
    with open(file_path, 'r') as f:
        content = f.read()

    today = date.today().strftime("%Y-%m-%d")
    updated = re.sub(
        r'_Last updated: \d{4}-\d{2}-\d{2}_',
        f'_Last updated: {today}_',
        content
    )

    with open(file_path, 'w') as f:
        f.write(updated)
```

- [ ] **Step 2: Add cmd_auto function**

Add after `cmd_sync`:

```python
def cmd_auto(args, config: dict):
    """Run autonomous pipeline: sync -> process -> video -> resize -> push."""
    print("Running autonomous pipeline...")
    print("=" * 50)

    # Step 1: Sync
    print("\n1. SYNC")
    print("-" * 50)
    gp_config = config.get('google_photos', {})
    downloader = GooglePhotosDownloader(
        credentials_path=gp_config.get('credentials_path', 'credentials.json'),
        token_path=gp_config.get('token_path', 'token.json'),
        album_name=gp_config.get('album_name', 'Ma Face, Straight'),
        output_dir=config['paths']['original_faces'],
        verbose=config['processing']['verbose']
    )

    try:
        downloaded = downloader.sync()
    except Exception as e:
        notify("Timelapse Sync Failed", str(e))
        raise

    if downloaded == 0:
        print("\nNo new photos. Pipeline complete (nothing to do).")
        return

    print(f"\nDownloaded {downloaded} new photos")

    # Step 2: Process
    print("\n2. PROCESS")
    print("-" * 50)
    cmd_process(args, config)

    # Check for failures that need annotation
    metadata = init_metadata(config)
    failed_images = metadata.data.get("failed_images", {})
    # Count only images that don't have manual landmarks and aren't permanently failed
    needs_annotation = metadata.get_images_needing_annotation()
    if needs_annotation:
        notify(
            "Timelapse: Annotation Needed",
            f"{len(needs_annotation)} images need manual annotation. "
            f"Run: poetry run timelapse annotate"
        )

    # Step 3: Video
    print("\n3. VIDEO")
    print("-" * 50)
    try:
        cmd_video(args, config)
    except Exception as e:
        notify("Timelapse Video Failed", str(e))
        raise

    # Step 4: Resize
    print("\n4. RESIZE")
    print("-" * 50)
    auto_config = config.get('auto', {})
    source_video = os.path.join(config['paths']['videos'], 'timelapse.mp4')
    website_repo = auto_config.get('website_repo', '')
    website_video_rel = auto_config.get('website_video_path', 'img/face-timelapse-small.mp4')
    website_video_path = os.path.join(website_repo, website_video_rel)

    try:
        resize_video(
            source_video,
            website_video_path,
            auto_config.get('resize_width', 720),
            auto_config.get('resize_height', 720),
            auto_config.get('resize_crf', 28)
        )
        print(f"Resized video written to {website_video_path}")
    except Exception as e:
        notify("Timelapse Resize Failed", str(e))
        raise

    # Step 5: Push website
    print("\n5. PUSH WEBSITE")
    print("-" * 50)
    try:
        sp.run(["git", "add", website_video_rel], cwd=website_repo, check=True)
        sp.run(
            ["git", "commit", "-m", "Update face timelapse"],
            cwd=website_repo, check=True
        )
        sp.run(["git", "push"], cwd=website_repo, check=True)
        print("Website repo pushed successfully")
    except sp.CalledProcessError as e:
        notify("Timelapse Website Push Failed", str(e))
        raise

    # Step 6: Update dates in this repo
    print("\n6. UPDATE DATES")
    print("-" * 50)
    update_last_updated_date('CLAUDE.md')
    update_last_updated_date('README.md')
    sp.run(["git", "add", "CLAUDE.md", "README.md", "metadata.json"], check=True)
    sp.run(["git", "commit", "-m", "Update last-updated dates after sync"], check=True)
    sp.run(["git", "push"], check=True)
    print("Timelapse repo dates updated and pushed")

    print("\n" + "=" * 50)
    print("Autonomous pipeline complete!")
```

- [ ] **Step 3: Register `auto` subparser and command**

Add subparser:

```python
    # Auto command
    subparsers.add_parser('auto', help='Run autonomous pipeline: sync, process, video, resize, push')
```

Add to the `commands` dict:

```python
        'auto': cmd_auto,
```

- [ ] **Step 4: Smoke test**

Run: `poetry run timelapse auto --help`
Expected: shows help without error

- [ ] **Step 5: Commit**

```bash
git add align_faces.py
git commit -m "feat: add auto CLI command for autonomous pipeline"
```

---

### Task 6: Launchd scheduling

**Files:**
- Create: `run_monthly.sh`
- Create: `com.ian.timelapse-portrait.plist`

- [ ] **Step 1: Create the wrapper script**

Create `run_monthly.sh`:

```bash
#!/bin/bash
# Monthly timelapse portrait pipeline runner.
# Called by launchd. Can also be run manually: bash run_monthly.sh

cd /Users/ian/Projects/timelapse_portrait
export PATH="/Users/ian/.local/bin:$PATH"

echo "=== Timelapse auto run: $(date) ==="
poetry run timelapse auto
echo "=== Finished: $(date) ==="
```

Make it executable:

```bash
chmod +x run_monthly.sh
```

- [ ] **Step 2: Create the launchd plist**

Create `com.ian.timelapse-portrait.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ian.timelapse-portrait</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/ian/Projects/timelapse_portrait/run_monthly.sh</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Day</key>
        <integer>1</integer>
        <key>Hour</key>
        <integer>10</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/ian/Library/Logs/timelapse-portrait.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ian/Library/Logs/timelapse-portrait.log</string>

    <key>WorkingDirectory</key>
    <string>/Users/ian/Projects/timelapse_portrait</string>
</dict>
</plist>
```

- [ ] **Step 3: Commit**

```bash
git add run_monthly.sh com.ian.timelapse-portrait.plist
git commit -m "feat: add launchd plist and wrapper script for monthly scheduling"
```

- [ ] **Step 4: Install the launchd agent**

This step requires user action. Print instructions:

```bash
cp com.ian.timelapse-portrait.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ian.timelapse-portrait.plist
```

Verify it's loaded:

```bash
launchctl list | grep timelapse
```

Expected: one line showing `com.ian.timelapse-portrait`

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

Add to the Modules list:

```markdown
- **`src/google_photos.py`**: GooglePhotosDownloader for syncing albums via Google Photos API
- **`src/notify.py`**: macOS notification utility for pipeline error reporting
```

Add to Running Commands section, after the existing commands:

```markdown
# Sync new photos from Google Photos
poetry run python align_faces.py sync

# Run full autonomous pipeline (sync -> process -> video -> resize -> push)
poetry run python align_faces.py auto
```

Update the "Photos are brought in manually" sentence to:

```markdown
Photos can be synced from Google Photos (`sync` command) or added manually to `original_faces/`.
```

Add a new section after Development Notes:

```markdown
## Automation

- **Monthly schedule**: Launchd runs `auto` on the 1st of each month at 10am
- **Launchd plist**: `com.ian.timelapse-portrait.plist` (install to `~/Library/LaunchAgents/`)
- **Logs**: `~/Library/Logs/timelapse-portrait.log`
- **Notifications**: macOS notifications on sync/video/push failures and when images need annotation
- **Google Photos setup**: Requires `credentials.json` from Google Cloud Console (OAuth 2.0 Desktop client). Run `poetry run timelapse sync` once interactively to authorize.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with sync and auto commands"
```

---

### Task 8: Google Cloud setup and end-to-end test

This task requires user interaction for OAuth consent.

- [ ] **Step 1: Find or create Google Cloud project**

Help the user locate their existing project or create a new one:

1. Go to https://console.cloud.google.com/
2. Check existing projects for one with Photos Library API enabled
3. If none found, create a new project
4. Enable the "Photos Library API" in the API library

- [ ] **Step 2: Create OAuth credentials**

1. Go to APIs & Services > Credentials
2. Create Credentials > OAuth 2.0 Client ID
3. Application type: Desktop app
4. Download the JSON and save as `credentials.json` in the project root

- [ ] **Step 3: Run sync interactively to authorize**

Run: `poetry run timelapse sync`

Expected: Opens browser for Google account consent. After approval, prints download count and creates `token.json`.

- [ ] **Step 4: Run a full auto test**

Run: `poetry run timelapse auto`

Expected:
1. Sync finds new photos (or 0 if just synced)
2. Process runs on any new photos
3. Video regenerates
4. Resized video written to website repo
5. Website repo committed and pushed
6. Dates updated in CLAUDE.md/README.md

- [ ] **Step 5: Verify notification works**

Test by temporarily breaking something, or run:

```python
poetry run python -c "from src.notify import notify; notify('Test', 'Hello from timelapse')"
```

Expected: macOS notification appears
