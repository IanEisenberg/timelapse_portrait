#!/usr/bin/env python3
"""Face timelapse portrait generator CLI."""

import argparse
import os
import sys
import yaml
import requests

from src.metadata_manager import MetadataManager
from src.image_processor import ImageProcessor
from src.video_generator import VideoGenerator
from src.average_image import AverageImageGenerator
from src.annotator import LandmarkAnnotator


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_landmarks_model(model_path: str):
    """Download facial landmarks model if not present."""
    if os.path.exists(model_path):
        return

    print(f"Facial landmarks model not found at {model_path}")
    print("Downloading shape_predictor_68_face_landmarks.dat...")

    landmarks_url = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"

    try:
        response = requests.get(landmarks_url, timeout=300)
        response.raise_for_status()

        with open(model_path, 'wb') as f:
            f.write(response.content)

        print(f"Downloaded landmarks model to {model_path}")

    except Exception as e:
        print(f"Error downloading landmarks model: {e}")
        print("Please download manually from:")
        print(landmarks_url)
        sys.exit(1)


def init_metadata(config: dict) -> MetadataManager:
    """Initialize or migrate metadata."""
    metadata_path = config['paths']['metadata']

    # Check if legacy files exist and need migration
    legacy_files = ['capture_issues.txt', 'needs_to_be_renamed.txt', 'script_failures.txt', 'rectangle_mapping.json']
    needs_migration = any(os.path.exists(f) for f in legacy_files)

    if needs_migration and not os.path.exists(metadata_path):
        print("Detected legacy metadata files. Starting migration...")
        metadata = MetadataManager.migrate_from_legacy(metadata_path, '.')
        print("Migration complete!")
    else:
        metadata = MetadataManager(metadata_path)

    return metadata




def cmd_process(args, config: dict):
    """Process faces: detect, align, and save."""
    print("Processing faces...")

    # Initialize metadata
    metadata = init_metadata(config)

    # Ensure landmarks model exists
    landmarks_path = config['paths']['landmarks_model']
    ensure_landmarks_model(landmarks_path)

    # Initialize processor
    processor = ImageProcessor(
        landmarks_model_path=landmarks_path,
        face_width=config['alignment']['face_width'],
        face_height=config['alignment']['face_height'],
        left_eye_position=tuple(config['alignment']['left_eye_position']),
        resize_width=config['alignment']['resize_width'],
        upsample_times=config['detection']['upsample_times'],
        auto_select_largest=config['detection']['auto_select_largest'],
        verbose=config['processing']['verbose']
    )

    # Process directory
    input_dir = config['paths']['original_faces']
    output_dir = config['paths']['aligned_faces']
    skip_existing = config['processing']['skip_existing']
    parallel = config['processing'].get('parallel', True)
    num_workers = config['processing'].get('num_workers', None)

    successful, failed = processor.process_directory(
        input_dir,
        output_dir,
        metadata,
        skip_existing,
        parallel=parallel,
        num_workers=num_workers
    )

    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


def cmd_video(args, config: dict):
    """Generate timelapse videos."""
    print("Generating videos...")

    # Initialize metadata
    metadata = init_metadata(config)

    # Get images with dates
    aligned_dir = config['paths']['aligned_faces']
    original_dir = config['paths']['original_faces']
    date_images = ImageProcessor.get_images_with_dates(aligned_dir, metadata, original_dir)

    if not date_images:
        print("No images with EXIF dates found")
        return

    print(f"Found {len(date_images)} images with dates")

    # Initialize video generator
    generator = VideoGenerator(
        fps=config['video']['fps'],
        codec=config['video']['codec'],
        h264_preset=config['video']['h264_preset'],
        crf=config['video']['crf'],
        show_date_overlay=config['video']['show_date_overlay'],
        verbose=config['processing']['verbose']
    )

    video_dir = config['paths']['videos']
    os.makedirs(video_dir, exist_ok=True)

    # Generate overall video
    overall_path = os.path.join(video_dir, 'timelapse.mp4')
    generator.create_overall_video(overall_path, date_images)

    # Generate yearly videos
    if config['video']['generate_yearly_videos']:
        videos_created = generator.create_videos_by_year(video_dir, date_images)
        print(f"Created {videos_created} yearly videos")

    print("\nVideo generation complete!")


def cmd_average(args, config: dict):
    """Generate average composite images."""
    print("Generating average images...")

    # Initialize metadata
    metadata = init_metadata(config)

    # Get images with dates
    aligned_dir = config['paths']['aligned_faces']
    original_dir = config['paths']['original_faces']
    date_images = ImageProcessor.get_images_with_dates(aligned_dir, metadata, original_dir)

    if not date_images:
        print("No images with EXIF dates found")
        return

    print(f"Found {len(date_images)} images with dates")

    # Initialize generator
    generator = AverageImageGenerator(
        verbose=config['processing']['verbose']
    )

    output_dir = config['paths']['average_images']
    min_images = config['average_images']['min_images']

    # Generate all averages
    counts = generator.generate_all_averages(
        date_images,
        output_dir,
        generate_overall=config['average_images']['generate_overall'],
        generate_by_year=config['average_images']['generate_by_year'],
        generate_by_quarter=config['average_images']['generate_by_quarter'],
        generate_by_month=config['average_images']['generate_by_month'],
        min_images=min_images
    )

    print("\nAverage image generation complete!")
    print(f"Overall: {counts['overall']}")
    print(f"Yearly: {counts['year']}")
    print(f"Quarterly: {counts['quarter']}")
    print(f"Monthly: {counts['month']}")


def cmd_all(args, config: dict):
    """Run full pipeline: process -> video -> average."""
    print("Running full pipeline...")
    print("=" * 50)

    # Process
    print("\n1. PROCESS")
    print("-" * 50)
    cmd_process(args, config)

    # Video
    print("\n2. VIDEO")
    print("-" * 50)
    cmd_video(args, config)

    # Average
    print("\n3. AVERAGE")
    print("-" * 50)
    cmd_average(args, config)

    print("\n" + "=" * 50)
    print("Full pipeline complete!")


def cmd_annotate(args, config: dict):
    """Manually annotate failed images."""
    print("Starting manual annotation...")

    # Initialize metadata
    metadata = init_metadata(config)

    # Get images needing annotation
    images_needing_annotation = metadata.get_images_needing_annotation()

    if not images_needing_annotation:
        print("No images need annotation!")
        print("\nAll failed images either:")
        print("  - Already have manual landmarks")
        print("  - Are marked as permanently failed")
        return

    print(f"Found {len(images_needing_annotation)} images needing annotation")
    print("\nInstructions:")
    print("  1. Click THEIR LEFT EYE (your right side) - green")
    print("  2. Click THEIR RIGHT EYE (your left side) - red")
    print("  3. Press ENTER to save")
    print("  4. Press 'r' to reset clicks")
    print("  5. Press 'f' to mark as permanently failed")
    print("  6. Press ESC to skip")
    print()

    # Initialize annotator
    annotator = LandmarkAnnotator(
        config['paths']['original_faces'],
        metadata,
        resize_width=config['alignment']['resize_width']
    )

    # Run annotation
    annotated, skipped, perm_failed = annotator.annotate_all_failed()

    print("\n" + "=" * 50)
    print("Annotation complete!")
    print(f"  Annotated: {annotated}")
    print(f"  Permanently failed: {perm_failed}")
    print(f"  Skipped: {skipped}")

    if annotated > 0:
        print(f"\nRun 'poetry run python align_faces.py retry' to process annotated images")


def cmd_retry(args, config: dict):
    """Retry processing images with manual landmarks."""
    print("Retrying images with manual landmarks...")

    # Initialize metadata
    metadata = init_metadata(config)

    # Get images with manual landmarks
    images_with_landmarks = list(metadata.data.get("manual_landmarks", {}).keys())

    if not images_with_landmarks:
        print("No images have manual landmarks!")
        return

    print(f"Found {len(images_with_landmarks)} images with manual landmarks")

    # Ensure landmarks model exists
    landmarks_path = config['paths']['landmarks_model']
    ensure_landmarks_model(landmarks_path)

    # Initialize processor
    processor = ImageProcessor(
        landmarks_model_path=landmarks_path,
        face_width=config['alignment']['face_width'],
        face_height=config['alignment']['face_height'],
        left_eye_position=tuple(config['alignment']['left_eye_position']),
        resize_width=config['alignment']['resize_width'],
        upsample_times=config['detection']['upsample_times'],
        auto_select_largest=config['detection']['auto_select_largest'],
        verbose=config['processing']['verbose']
    )

    # Process each image
    input_dir = config['paths']['original_faces']
    output_dir = config['paths']['aligned_faces']

    successful = 0
    failed = 0

    for image_name in images_with_landmarks:
        # Find original image
        base_name = os.path.splitext(image_name)[0]
        image_path = None
        for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.HEIC', '.heic']:
            test_path = os.path.join(input_dir, base_name + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break

        if not image_path:
            print(f"Could not find original image for {image_name}")
            failed += 1
            continue

        # Process image (will use manual landmarks)
        output_path = os.path.join(output_dir, base_name + '.jpg')
        if processor.process_single_image(image_path, output_path, metadata, skip_existing=False):
            successful += 1
        else:
            failed += 1

    print(f"\nRetry complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")




def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Face timelapse portrait generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s process         Detect and align faces
  %(prog)s annotate        Manually annotate failed images
  %(prog)s retry           Process annotated images
  %(prog)s video           Generate timelapse videos
  %(prog)s average         Generate composite images
  %(prog)s all             Run full pipeline
        """
    )

    parser.add_argument(
        '-c', '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Process command
    subparsers.add_parser('process', help='Process faces: detect, align, and save')

    # Video command
    subparsers.add_parser('video', help='Generate timelapse videos')

    # Average command
    subparsers.add_parser('average', help='Generate average composite images')

    # All command
    subparsers.add_parser('all', help='Run full pipeline (process -> video -> average)')

    # Annotate command
    subparsers.add_parser('annotate', help='Manually annotate failed images with eye landmarks')

    # Retry command
    subparsers.add_parser('retry', help='Retry processing images with manual landmarks')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Route to command
    commands = {
        'process': cmd_process,
        'video': cmd_video,
        'average': cmd_average,
        'all': cmd_all,
        'annotate': cmd_annotate,
        'retry': cmd_retry
    }

    command_func = commands.get(args.command)
    if command_func:
        try:
            command_func(args, config)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
