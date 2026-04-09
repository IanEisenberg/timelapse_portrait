# Autonomous Monthly Sync & Deploy

## Summary

Automate the full timelapse portrait pipeline on a monthly schedule: sync new photos from Google Photos, process faces, regenerate the timelapse video, produce a compressed copy for the personal website, and push it live. Notify via macOS notification on failures or when images need manual annotation.

## Approach

All-Python orchestrator (Approach B). A new `auto` CLI command chains sync, process, video, resize, copy, and push as one entry point. Launchd handles monthly scheduling.

## Components

### 1. Google Photos Sync (`src/google_photos.py`)

New module: `GooglePhotosDownloader`.

- Uses Google Photos Library API (`mediaItems.search`) filtered by album name
- Album: "Ma Face, Straight" (configurable in `config.yaml`)
- Compares API results against filenames already in `original_faces/` — downloads only new items
- Preserves original filenames and EXIF data
- Auth: OAuth2 with `credentials.json` → `token.json`. First run requires interactive browser consent. Subsequent runs use the refresh token automatically.
- CLI exposure: `poetry run timelapse sync`

Dependencies to add: `google-auth`, `google-auth-oauthlib`, `google-api-python-client`

### 2. Auto Command (`cmd_auto` in `align_faces.py`)

New CLI command: `poetry run timelapse auto`

Pipeline steps, in order:

1. **Sync** — download new photos from Google Photos
2. **Process** — detect/align faces (skips already-processed via `skip_existing`)
3. **Video** — regenerate `timelapse.mp4` and yearly videos
4. **Resize** — ffmpeg subprocess converts `timelapse.mp4` to 720x720, CRF ~28
5. **Copy** — write resized video to website repo path
6. **Push website** — `git add`, `git commit -m "Update face timelapse"`, `git push` in the website repo
7. **Update dates** — bump `_Last updated_` in `CLAUDE.md` and `README.md`, commit and push timelapse repo

**Early exit:** If sync downloaded zero new photos, stop (no point regenerating identical videos).

### 3. Configuration Additions (`config.yaml`)

```yaml
google_photos:
  album_name: "Ma Face, Straight"
  credentials_path: "credentials.json"
  token_path: "token.json"

auto:
  website_repo: "/Users/ian/Projects/IanEisenberg.github.io"
  website_video_path: "img/face-timelapse-small.mp4"
  resize_width: 720
  resize_height: 720
  resize_crf: 28
```

### 4. Scheduling (launchd)

Plist: `~/Library/LaunchAgents/com.ian.timelapse-portrait.plist`

- `StartCalendarInterval`: day 1, hour 10 (10am on the 1st of each month)
- Runs as user (access to local repos, keychain, macOS notifications)
- If Mac was asleep/off on the 1st, launchd fires the job on next wake
- Logs stdout/stderr to `~/Library/Logs/timelapse-portrait.log`
- Working directory set to the timelapse_portrait project root
- Command: `poetry run timelapse auto`
- Install: `launchctl load ~/Library/LaunchAgents/com.ian.timelapse-portrait.plist`

### 5. Error Handling & Notifications

Notification mechanism: `osascript -e 'display notification ...'` via subprocess. One utility function: `notify(title, message)`.

| Scenario | Behavior |
|---|---|
| Sync failure (auth expired, network) | Notify with error reason, stop pipeline |
| Processing failures (faces not detected) | Continue pipeline. After processing, notify with count ("N images need annotation"). Videos regenerate with whatever succeeded. |
| Video/resize/push failure | Notify with error. If video OK but push failed, local video is still updated. |
| No new photos | Log "no new photos found", exit cleanly. No notification. |

## Files Changed

- **New:** `src/google_photos.py` — Google Photos sync module
- **Modified:** `align_faces.py` — add `sync` and `auto` subcommands
- **Modified:** `config.yaml` — add `google_photos` and `auto` sections
- **Modified:** `pyproject.toml` — add google-auth dependencies
- **New:** `com.ian.timelapse-portrait.plist` — launchd plist (committed to repo for reference, installed to `~/Library/LaunchAgents/`)
- **Modified:** `CLAUDE.md` — document new commands and automation

## Google Cloud Setup

During implementation, we'll need to:

1. Find or create a Google Cloud project with Photos Library API enabled
2. Create OAuth2 credentials (Desktop app type)
3. Download `credentials.json` to the project root
4. Run `poetry run timelapse sync` once interactively to complete browser auth and generate `token.json`

## Out of Scope

- Email notifications (deferred — macOS notifications for now)
- Average image regeneration in the auto pipeline (can be added later)
- Handling annotation within the auto flow (manual step after notification)
