# Face Timelapse Portrait Generator

Generate timelapse videos and composite images from portrait photos over time. This tool detects faces, aligns them based on eye positions, and creates visualizations showing changes over time.

## Quick Start

Get up and running in 3 steps:

### 1. Install

```bash
# Install Poetry (if not already installed)
brew install poetry  # macOS
# or: curl -sSL https://install.python-poetry.org | python3 -

# Install ffmpeg for video encoding
brew install ffmpeg  # macOS
# or: sudo apt-get install ffmpeg  # Ubuntu/Debian

# Clone and install dependencies
cd timelapse_portrait
poetry install
```

### 2. Add Photos

Place your portrait photos in the `original_faces/` directory.

### 3. Run

```bash
# One command to do everything:
# - Detect and align faces
# - Generate timelapse videos
# - Create average composite images
poetry run timelapse all
```

That's it! Your videos will be in `videos/` and composite images in `average_images/`.

---

## Features

- **Face Detection & Alignment**: Automatically detects and aligns faces using dlib facial landmarks
- **Video Generation**: Create timelapse videos with optional date overlays
- **Average Composites**: Generate "average face" images by time period (overall, yearly, quarterly, monthly)
- **Modular Architecture**: Clean, testable code structure with CLI and Jupyter notebook support
- **Smart Metadata**: Unified tracking of processing state and failures

## Installation

### Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- ffmpeg (for H.264 video encoding)
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Install Poetry

If you don't have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or on macOS/Linux:

```bash
brew install poetry
```

### Install Dependencies

```bash
# Install all dependencies
poetry install

# Or install without dev dependencies (jupyter, pytest, etc.)
poetry install --only main
```

This will create a virtual environment and install all required packages.

## Quick Start

### 1. Configure Settings

Edit `config.yaml` to set your preferences:

```yaml
paths:
  original_faces: "original_faces"
  aligned_faces: "aligned_faces"
  videos: "videos"
  average_images: "average_images"

alignment:
  face_width: 1024
  face_height: 1024

video:
  fps: 9
  codec: "libx264"
  show_date_overlay: true
```

### 2. Add Photos

Place your portrait photos in the `original_faces/` directory, or set up Google Photos sync (see below).

### 3. Run the Pipeline

```bash
# Process faces (detect and align)
poetry run python align_faces.py process

# Generate videos
poetry run python align_faces.py video

# Generate average images
poetry run python align_faces.py average

# Or run everything at once
poetry run python align_faces.py all

# Or use the shortcut command
poetry run timelapse all
```

Or activate the Poetry shell and run directly:

```bash
poetry shell
python align_faces.py all
```

## CLI Commands

All commands can be run with `poetry run python align_faces.py <command>` or `poetry run timelapse <command>`. For brevity, examples below show the direct command.

### Process Faces

```bash
poetry run python align_faces.py process
# or: poetry run timelapse process
```

Detects faces in images, aligns them, and saves to `aligned_faces/` directory.

### Generate Videos

```bash
poetry run python align_faces.py video
```

Creates timelapse videos from aligned faces. Generates:
- `videos/timelapse.mp4` - Overall timelapse
- `videos/YYYY_timelapse.mp4` - Per-year videos (if enabled)

### Generate Average Images

```bash
poetry run python align_faces.py average
```

Creates composite "average face" images:
- `average_images/overall_average_image.jpg`
- `average_images/YYYY_average_Nimages.jpg` - Yearly averages
- `average_images/YYYY_QN_average_Nimages.jpg` - Quarterly averages
- `average_images/YYYY_MM_average_Nimages.jpg` - Monthly averages

### Run Full Pipeline

```bash
poetry run python align_faces.py all
```

Runs the complete pipeline: process → video → average

## Configuration

See `config.yaml` for all available options:

- **Paths**: Directory locations for inputs/outputs
- **Alignment**: Face size and eye position settings
- **Detection**: Face detection parameters
- **Video**: FPS, codec, quality settings
- **Average Images**: Which time periods to generate
- **Processing**: Verbose logging, skip existing files, **parallel processing** (enabled by default)

## Project Structure

```
timelapse_portrait/
├── config.yaml              # Configuration file
├── align_faces.py           # Main CLI script
├── align_faces_refactored.ipynb  # Jupyter notebook
├── src/
│   ├── face_aligner.py      # Face alignment logic
│   ├── image_processor.py   # Image processing pipeline
│   ├── video_generator.py   # Video creation
│   ├── average_image.py     # Composite image generation
│   ├── google_photos.py     # Google Photos API integration
│   └── metadata_manager.py  # Metadata tracking
├── original_faces/          # Input photos
├── aligned_faces/           # Aligned face outputs
├── videos/                  # Generated videos
└── average_images/          # Generated composites
```

## Jupyter Notebook

For experimentation and visualization, use the provided notebook:

```bash
# Install dev dependencies (includes Jupyter)
poetry install

# Launch Jupyter in the Poetry environment
poetry run jupyter notebook align_faces_refactored.ipynb
```

The notebook demonstrates:
- Processing individual images
- Visualizing results
- Generating videos and averages
- Analyzing metadata

## Development

### Adding Dependencies

```bash
# Add a main dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Update all dependencies
poetry update
```

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black src/ align_faces.py
```

## Migration from Legacy

If you have existing data from the old notebook-based workflow, the tool will automatically migrate:
- `capture_issues.txt` → `metadata.json`
- `needs_to_be_renamed.txt` → `metadata.json`
- `script_failures.txt` → `metadata.json`
- `rectangle_mapping.json` → `metadata.json`

Old files are backed up with `.backup` extension.

## Troubleshooting

### No faces detected

- Increase `upsample_times` in config (default: 2)
- Check that images contain clear frontal faces
- Verify images aren't corrupted

### Wrong face selected

- The tool automatically selects the largest face
- For manual override, add entry to `metadata.json`:
  ```json
  "face_overrides": {
    "IMG_1234.jpg": 1
  }
  ```

### Manual Landmark Annotation

For images where automatic face detection fails (e.g., wearing mask/goggles), you can manually annotate eye landmarks:

```bash
# Run automatic processing first
poetry run python align_faces.py process

# Annotate failed images
poetry run python align_faces.py annotate

# Process annotated images
poetry run python align_faces.py retry
```

**Annotation workflow:**
1. Failed images are displayed one by one
2. Click **their left eye** center (your right side - marked green)
3. Click **their right eye** center (your left side - marked red)
4. Press ENTER to save, 'r' to reset, 'f' to mark permanently failed, ESC to skip

**Important:** Use anatomical left/right (the person's perspective), not viewer's left/right.

Manual landmarks are saved in `metadata.json` and used automatically during retry.

### Video encoding fails

- Ensure ffmpeg is installed
- Fallback to `codec: "mp4v"` in config if needed

### Google Photos authentication fails

- Check `credentials.json` is in project directory
- Delete `token.pickle` and re-authenticate
- Verify Photos Library API is enabled

## Advanced Usage

### Custom Configuration File

```bash
python align_faces.py -c custom_config.yaml process
```

### Programmatic Usage

```python
from src.image_processor import ImageProcessor
from src.metadata_manager import MetadataManager

metadata = MetadataManager('metadata.json')
processor = ImageProcessor('shape_predictor_68_face_landmarks.dat')

processor.process_directory(
    'original_faces',
    'aligned_faces',
    metadata=metadata
)
```

## Contributing

This is a personal project, but suggestions and improvements are welcome!

## License

MIT License

## Acknowledgments

- dlib for face detection and alignment
- imutils for face utilities
- Google Photos API for photo management
