"""Video generation from aligned face images."""

import os
from typing import List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import cv2
import numpy as np


class VideoGenerator:
    """Generates timelapse videos from aligned face images."""

    def __init__(
        self,
        fps: int = 9,
        codec: str = "libx264",
        h264_preset: str = "medium",
        crf: int = 23,
        show_date_overlay: bool = True,
        verbose: bool = True
    ):
        """Initialize video generator.

        Args:
            fps: Frames per second
            codec: Video codec ("libx264" or "mp4v")
            h264_preset: H.264 encoding preset (ultrafast to veryslow)
            crf: Constant Rate Factor for H.264 (0-51, lower = better)
            show_date_overlay: Add date overlay to frames
            verbose: Enable verbose logging
        """
        self.fps = fps
        self.codec = codec
        self.h264_preset = h264_preset
        self.crf = crf
        self.show_date_overlay = show_date_overlay
        self.verbose = verbose

    def add_date_to_image(self, image: np.ndarray, date_string: str) -> np.ndarray:
        """Add date overlay to an image.

        Args:
            image: Input image
            date_string: Date string in format "YYYY:MM:DD HH:MM:SS"

        Returns:
            Image with date overlay
        """
        # Convert date string to desired format
        date_obj = datetime.strptime(date_string, "%Y:%m:%d %H:%M:%S")
        formatted_date = date_obj.strftime("%Y-%m-%d")

        # Set text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        thickness = 2

        # Get text size
        text_size = cv2.getTextSize(formatted_date, font, font_scale, thickness)[0]

        # Set text position (bottom left)
        text_x = 10
        text_y = image.shape[0] - 10  # 10 pixels from the bottom

        # Add black background for better readability
        cv2.rectangle(
            image,
            (text_x, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0),
            -1
        )

        # Add text to the image
        cv2.putText(
            image,
            formatted_date,
            (text_x + 5, text_y),
            font,
            font_scale,
            font_color,
            thickness
        )

        return image

    def create_video(
        self,
        output_path: str,
        date_image_array: List[Tuple[str, str]],
        add_date_overlay: Optional[bool] = None
    ) -> bool:
        """Create a video from a list of dated images.

        Args:
            output_path: Output video file path
            date_image_array: List of (date_string, image_path) tuples
            add_date_overlay: Override to enable/disable date overlay

        Returns:
            True if successful, False otherwise
        """
        if not date_image_array:
            print("No images provided for video generation")
            return False

        # Use instance setting if not overridden
        if add_date_overlay is None:
            add_date_overlay = self.show_date_overlay

        try:
            # Get video dimensions from first frame
            first_frame = cv2.imread(date_image_array[0][1])
            if first_frame is None:
                print(f"Could not read first frame: {date_image_array[0][1]}")
                return False

            height, width, layers = first_frame.shape

            # Create video writer
            if self.codec == "libx264":
                # Use ffmpeg backend for H.264
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    self.fps,
                    (width, height)
                )
            else:
                # Fallback to MP4V
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    self.fps,
                    (width, height)
                )

            # Add frames
            for date, path in date_image_array:
                img = cv2.imread(path)
                if img is None:
                    print(f"Warning: Could not read {path}, skipping")
                    continue

                if add_date_overlay:
                    img = self.add_date_to_image(img.copy(), date)

                video.write(img)

            # Release video
            video.release()
            cv2.destroyAllWindows()

            if self.verbose:
                print(f"Video created: {output_path} ({len(date_image_array)} frames)")

            return True

        except Exception as e:
            print(f"Error creating video {output_path}: {e}")
            return False

    def create_videos_by_year(
        self,
        output_dir: str,
        date_image_array: List[Tuple[str, str]],
        min_images: int = 5
    ) -> int:
        """Create separate videos for each year.

        Args:
            output_dir: Output directory for videos
            date_image_array: List of (date_string, image_path) tuples
            min_images: Minimum images required to create a video

        Returns:
            Number of videos created
        """
        # Group images by year
        images_by_year = defaultdict(list)

        for date_str, image_path in date_image_array:
            date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            images_by_year[date.year].append((date_str, image_path))

        # Create videos
        videos_created = 0
        for year, images in sorted(images_by_year.items()):
            if len(images) < min_images:
                print(f"Skipping year {year} - only {len(images)} images (minimum: {min_images})")
                continue

            output_path = os.path.join(output_dir, f'{year}_timelapse.mp4')
            if self.create_video(output_path, images):
                videos_created += 1

        return videos_created

    def create_overall_video(
        self,
        output_path: str,
        date_image_array: List[Tuple[str, str]]
    ) -> bool:
        """Create a single video with all images.

        Args:
            output_path: Output video file path
            date_image_array: List of (date_string, image_path) tuples

        Returns:
            True if successful
        """
        return self.create_video(output_path, date_image_array)

    @staticmethod
    def group_images_by_period(
        date_image_array: List[Tuple[str, str]]
    ) -> Tuple[dict, dict, dict]:
        """Group images by year, quarter, and month.

        Args:
            date_image_array: List of (date_string, image_path) tuples

        Returns:
            Tuple of (images_by_year, images_by_quarter, images_by_month) dicts
        """
        images_by_year = defaultdict(list)
        images_by_month = defaultdict(list)
        images_by_quarter = defaultdict(list)

        for date_str, image_path in date_image_array:
            date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')

            # Calculate the quarter
            quarter = (date.month - 1) // 3 + 1

            year_key = date.year
            month_key = (date.year, date.month)
            quarter_key = (date.year, f"Q{quarter}")

            # Add the image to the appropriate list
            images_by_year[year_key].append(image_path)
            images_by_month[month_key].append(image_path)
            images_by_quarter[quarter_key].append(image_path)

        return images_by_year, images_by_quarter, images_by_month
