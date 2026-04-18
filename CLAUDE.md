# CLAUDE.md

_Last updated: 2025-10-26_

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Maintenance note**: Whenever new photos are added to `original_faces/`, update the `_Last updated_` date at the top of this file (and `README.md`) to reflect that date.

## Project Overview

This is a face timelapse portrait project that processes a collection of portrait photos to create aligned face videos and average composite images over time. The pipeline detects faces, aligns them based on eye positions, and generates time-series visualizations.

**Note**: This project has been refactored from a single Jupyter notebook into a modular CLI application with Poetry for dependency management.

## Core Architecture

The project is now organized as a modular Python application with the following structure:

### Modules

- **`src/face_aligner.py`**: FaceAligner class for aligning faces based on eye positions
- **`src/image_processor.py`**: ImageProcessor for face detection, alignment pipeline, EXIF handling
- **`src/video_generator.py`**: VideoGenerator for creating timelapse videos with date overlays
- **`src/average_image.py`**: AverageImageGenerator for composite image creation
- **`src/metadata_manager.py`**: MetadataManager for unified tracking (replaces legacy .txt files)
- **`src/annotator.py`**: LandmarkAnnotator for manual eye-landmark annotation of failed images
- **`src/google_photos.py`**: GooglePhotosDownloader for syncing albums via Google Photos API
- **`src/notify.py`**: macOS notification utility for pipeline error reporting
- **`align_faces.py`**: Main CLI entry point with argparse commands

### Processing Pipeline

1. **Face Detection & Alignment**:
   - Uses dlib's face detector with 68-point facial landmarks
   - Automatically selects largest face when multiple detected
   - Aligns to 1024x1024 with eyes at (0.42, 0.5) relative coordinates
   - Saves to `aligned_faces/` as JPG

2. **Metadata Tracking** (`metadata.json`):
   - Failed images with reasons and categories
   - Face selection overrides (manual corrections)
   - Manual eye-landmark annotations
   - Successfully processed images

3. **Output Generation**:
   - **Videos**: H.264 MP4 timelapses at 9fps with optional date overlays
   - **Averages**: Composite images by overall, year, quarter, month
   - Minimum image thresholds configurable

## Dependencies & Environment

**Managed via Poetry** (`pyproject.toml`):

```bash
# Install all dependencies
poetry install

# Install without dev dependencies
poetry install --only main

# Add new dependency
poetry add package-name
```

Key dependencies:
- opencv-python, dlib, imutils: Face processing
- pillow-heif: HEIC format support
- pyyaml: Configuration management

**External requirement**: ffmpeg for H.264 video encoding

## Running Commands

All commands via CLI using Poetry:

```bash
# Process faces
poetry run python align_faces.py process

# Generate videos
poetry run python align_faces.py video

# Generate averages
poetry run python align_faces.py average

# Full pipeline
poetry run python align_faces.py all

# Manually annotate images where detection failed
poetry run python align_faces.py annotate

# Retry processing images with manual landmarks
poetry run python align_faces.py retry

# Sync new photos from Google Photos
poetry run python align_faces.py sync

# Run full autonomous pipeline (sync -> process -> video -> resize -> push)
poetry run python align_faces.py auto

# Or use shortcut
poetry run timelapse <command>
```

Photos can be synced from Google Photos (`sync` command) or added manually to `original_faces/`.

## Configuration

All settings in `config.yaml`:
- Paths (input/output directories)
- Alignment parameters (face size, eye position)
- Detection settings (upsample times, auto-select)
- Video settings (fps, codec, quality)
- Average image settings (time periods, minimums)
- Processing settings (parallelism, skip-existing, verbosity)
- `google_photos`: album name, credentials/token paths
- `auto`: website repo path, resize dimensions/CRF

## Development Notes

- **Face Selection**: Automatically uses largest face; manual overrides via metadata.json
- **Skip Existing**: Configured in config.yaml to skip already-processed images
- **Migration**: Legacy .txt files and rectangle_mapping.json auto-migrate to metadata.json on first run
- **Jupyter Notebook**: `align_faces_refactored.ipynb` demonstrates programmatic usage
- **Testing**: Run `poetry run pytest` (dev dependencies required)
- **Formatting**: Use `poetry run black src/ align_faces.py`

## Automation

- **Monthly schedule**: Launchd runs `auto` on the 1st of each month at 10am
- **Launchd plist**: `com.ian.timelapse-portrait.plist` (install to `~/Library/LaunchAgents/`)
- **Logs**: `~/Library/Logs/timelapse-portrait.log`
- **Notifications**: macOS notifications on sync/video/push failures and when images need annotation
- **Google Photos setup**: Requires `credentials.json` from Google Cloud Console (OAuth 2.0 Desktop client). Run `poetry run timelapse sync` once interactively to authorize and generate `token.json`.
