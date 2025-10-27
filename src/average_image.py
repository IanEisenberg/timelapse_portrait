"""Generate average/composite images from aligned faces."""

import os
from typing import List, Optional, Dict
from datetime import datetime
from collections import defaultdict
import cv2
import numpy as np


class AverageImageGenerator:
    """Creates composite 'average face' images from multiple aligned faces."""

    def __init__(self, verbose: bool = True):
        """Initialize average image generator.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

    def create_average_image(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """Create an average image from a list of image paths.

        Args:
            image_paths: List of paths to aligned face images

        Returns:
            Average image as numpy array, or None if failed
        """
        if not image_paths:
            print("No images provided for averaging")
            return None

        average_image = None
        total_images = 0

        for path in image_paths:
            try:
                # Read the image
                image = cv2.imread(path)
                if image is None:
                    if self.verbose:
                        print(f"Warning: Could not read {path}, skipping")
                    continue

                # Convert to float for averaging
                image = image.astype(np.float32)
                total_images += 1

                # Add to average
                if average_image is None:
                    average_image = image
                else:
                    average_image += image

            except Exception as e:
                if self.verbose:
                    print(f"Error processing {path}: {e}")
                continue

        if average_image is None or total_images == 0:
            print("No valid images to average")
            return None

        # Compute average
        average_image /= total_images

        # Convert back to 8-bit
        average_image = average_image.astype(np.uint8)

        if self.verbose:
            print(f"Created average from {total_images} images")

        return average_image

    def save_average_image(
        self,
        image_paths: List[str],
        output_path: str
    ) -> bool:
        """Create and save an average image.

        Args:
            image_paths: List of paths to aligned face images
            output_path: Path to save the average image

        Returns:
            True if successful, False otherwise
        """
        average_image = self.create_average_image(image_paths)

        if average_image is None:
            return False

        try:
            cv2.imwrite(output_path, average_image)
            if self.verbose:
                print(f"Saved average image: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving average image to {output_path}: {e}")
            return False

    def generate_overall_average(
        self,
        image_paths: List[str],
        output_dir: str
    ) -> bool:
        """Generate overall average image from all images.

        Args:
            image_paths: List of all image paths
            output_dir: Output directory

        Returns:
            True if successful
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'overall_average_image.jpg')
        return self.save_average_image(image_paths, output_path)

    def generate_averages_by_period(
        self,
        date_image_array: List[tuple],
        output_dir: str,
        generate_by_year: bool = True,
        generate_by_quarter: bool = True,
        generate_by_month: bool = True,
        min_images: int = 3
    ) -> Dict[str, int]:
        """Generate average images grouped by time periods.

        Args:
            date_image_array: List of (date_string, image_path) tuples
            output_dir: Output directory
            generate_by_year: Generate yearly averages
            generate_by_quarter: Generate quarterly averages
            generate_by_month: Generate monthly averages
            min_images: Minimum images required to generate an average

        Returns:
            Dictionary with counts of generated averages by period type
        """
        os.makedirs(output_dir, exist_ok=True)

        # Group images by period
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

            # Add the image to the appropriate lists
            images_by_year[year_key].append(image_path)
            images_by_month[month_key].append(image_path)
            images_by_quarter[quarter_key].append(image_path)

        counts = {
            'year': 0,
            'quarter': 0,
            'month': 0
        }

        # Generate yearly averages
        if generate_by_year:
            for year, images in sorted(images_by_year.items()):
                if len(images) < min_images:
                    if self.verbose:
                        print(f"Skipping year {year} - only {len(images)} images")
                    continue

                output_path = os.path.join(
                    output_dir,
                    f'{year}_average_{len(images)}images.jpg'
                )
                if self.save_average_image(images, output_path):
                    counts['year'] += 1

        # Generate quarterly averages
        if generate_by_quarter:
            for (year, quarter), images in sorted(images_by_quarter.items()):
                if len(images) < min_images:
                    if self.verbose:
                        print(f"Skipping {year} {quarter} - only {len(images)} images")
                    continue

                output_path = os.path.join(
                    output_dir,
                    f'{year}_{quarter}_average_{len(images)}images.jpg'
                )
                if self.save_average_image(images, output_path):
                    counts['quarter'] += 1

        # Generate monthly averages
        if generate_by_month:
            for (year, month), images in sorted(images_by_month.items()):
                if len(images) < min_images:
                    if self.verbose:
                        print(f"Skipping {year}-{month:02d} - only {len(images)} images")
                    continue

                output_path = os.path.join(
                    output_dir,
                    f'{year}_{month:02d}_average_{len(images)}images.jpg'
                )
                if self.save_average_image(images, output_path):
                    counts['month'] += 1

        if self.verbose:
            print(f"\nGenerated averages: {counts['year']} yearly, "
                  f"{counts['quarter']} quarterly, {counts['month']} monthly")

        return counts

    def generate_all_averages(
        self,
        date_image_array: List[tuple],
        output_dir: str,
        generate_overall: bool = True,
        generate_by_year: bool = True,
        generate_by_quarter: bool = True,
        generate_by_month: bool = True,
        min_images: int = 3
    ) -> Dict[str, int]:
        """Generate all average images (overall and by periods).

        Args:
            date_image_array: List of (date_string, image_path) tuples
            output_dir: Output directory
            generate_overall: Generate overall average
            generate_by_year: Generate yearly averages
            generate_by_quarter: Generate quarterly averages
            generate_by_month: Generate monthly averages
            min_images: Minimum images required for period averages

        Returns:
            Dictionary with counts of generated averages
        """
        counts = {'overall': 0, 'year': 0, 'quarter': 0, 'month': 0}

        # Generate overall average
        if generate_overall:
            image_paths = [path for _, path in date_image_array]
            if self.generate_overall_average(image_paths, output_dir):
                counts['overall'] = 1

        # Generate period averages
        period_counts = self.generate_averages_by_period(
            date_image_array,
            output_dir,
            generate_by_year,
            generate_by_quarter,
            generate_by_month,
            min_images
        )

        counts.update(period_counts)

        return counts
