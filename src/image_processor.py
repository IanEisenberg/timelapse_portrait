"""Image processing pipeline for face alignment."""

import os
from typing import List, Tuple, Optional
from glob import glob
from datetime import datetime
from multiprocessing import Pool, cpu_count
import cv2
import dlib
import imutils
import numpy as np
from imutils.face_utils import rect_to_bb
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm

from .face_aligner import FaceAligner
from .metadata_manager import MetadataManager

# Register HEIC support
register_heif_opener()


# Global variables for multiprocessing workers
_worker_detector = None
_worker_predictor = None
_worker_face_aligner = None


def _init_worker(landmarks_model_path: str, face_width: int, face_height: int,
                 left_eye_position: Tuple[float, float], resize_width: int,
                 upsample_times: int):
    """Initialize dlib models for worker process.

    This is called once per worker process to load models into global variables.
    """
    global _worker_detector, _worker_predictor, _worker_face_aligner

    _worker_detector = dlib.get_frontal_face_detector()
    _worker_predictor = dlib.shape_predictor(landmarks_model_path)
    _worker_face_aligner = FaceAligner(
        _worker_predictor,
        desiredLeftEye=left_eye_position,
        desiredFaceWidth=face_width,
        desiredFaceHeight=face_height
    )


def _process_image_worker(task_data: Tuple) -> Tuple[str, bool, Optional[str]]:
    """Worker function to process a single image.

    Uses globally initialized models to avoid reloading for each image.

    Args:
        task_data: Tuple of (image_path, output_path, skip_existing, resize_width,
                   upsample_times, auto_select_largest)

    Returns:
        Tuple of (image_name, success, error_message)
    """
    global _worker_detector, _worker_predictor, _worker_face_aligner

    image_path, output_path, skip_existing, resize_width, upsample_times, auto_select_largest = task_data
    image_name = os.path.basename(image_path)

    try:
        # Skip if already processed
        if skip_existing and os.path.exists(output_path):
            return (image_name, True, None)

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = imutils.resize(image, width=resize_width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        rects = _worker_detector(gray, upsample_times)
        if not rects:
            return (image_name, False, "No faces detected")

        # Select face (largest if auto_select_largest)
        if auto_select_largest and len(rects) > 1:
            areas = [(rect.width() * rect.height(), rect) for rect in rects]
            rect = max(areas, key=lambda x: x[0])[1]
        else:
            rect = rects[0]

        # Align face
        aligned_face = _worker_face_aligner.align(image, gray, rect)

        # Save
        cv2.imwrite(output_path, aligned_face)

        return (image_name, True, None)

    except Exception as e:
        return (image_name, False, str(e))


class ImageProcessor:
    """Handles face detection, alignment, and image processing pipeline."""

    def __init__(
        self,
        landmarks_model_path: str,
        face_width: int = 1024,
        face_height: int = 1024,
        left_eye_position: Tuple[float, float] = (0.42, 0.5),
        resize_width: int = 800,
        upsample_times: int = 2,
        auto_select_largest: bool = True,
        verbose: bool = True
    ):
        """Initialize image processor.

        Args:
            landmarks_model_path: Path to dlib facial landmarks model
            face_width: Output face width in pixels
            face_height: Output face height in pixels
            left_eye_position: Desired left eye position (x, y) as fraction
            resize_width: Width to resize input images to before processing
            upsample_times: Number of times to upsample image for face detection
            auto_select_largest: Automatically select largest face when multiple detected
            verbose: Enable verbose logging
        """
        self.landmarks_model_path = landmarks_model_path
        self.resize_width = resize_width
        self.upsample_times = upsample_times
        self.auto_select_largest = auto_select_largest
        self.verbose = verbose
        self.face_width = face_width
        self.face_height = face_height
        self.left_eye_position = left_eye_position

        # Initialize face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarks_model_path)

        # Initialize face aligner
        self.face_aligner = FaceAligner(
            self.predictor,
            desiredLeftEye=left_eye_position,
            desiredFaceWidth=face_width,
            desiredFaceHeight=face_height
        )

    def load_and_preprocess_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load image and apply preprocessing.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (color_image, grayscale_image) or (None, None) on error
        """
        try:
            # Load image and handle EXIF orientation
            image = Image.open(image_path).convert('RGB')
            image = ImageOps.exif_transpose(image)

            # Convert to OpenCV format
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Resize for faster processing
            image = imutils.resize(image, width=self.resize_width)

            # Create grayscale version
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.verbose:
                print(f"Successfully loaded {image_path}")

            return image, gray

        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, None

    def detect_faces(self, gray: np.ndarray) -> List[dlib.rectangle]:
        """Detect faces in grayscale image.

        Args:
            gray: Grayscale image

        Returns:
            List of face rectangles
        """
        return self.detector(gray, self.upsample_times)

    def select_face(
        self,
        rects: List[dlib.rectangle],
        image_name: str,
        metadata: Optional[MetadataManager] = None
    ) -> Optional[dlib.rectangle]:
        """Select which face to use from detected faces.

        Args:
            rects: List of detected face rectangles
            image_name: Base name of the image file
            metadata: Optional metadata manager for face overrides

        Returns:
            Selected face rectangle or None if no faces
        """
        if not rects:
            return None

        # Check for manual override
        if metadata:
            face_index = metadata.get_face_index(image_name, default=0)
            if 0 <= face_index < len(rects):
                return rects[face_index]

        # Auto-select largest face
        if self.auto_select_largest and len(rects) > 1:
            # Calculate area of each face
            areas = [(rect.width() * rect.height(), rect) for rect in rects]
            # Return largest face
            return max(areas, key=lambda x: x[0])[1]

        # Default to first face
        return rects[0]

    def align_face(self, image: np.ndarray, gray: np.ndarray, rect: dlib.rectangle) -> np.ndarray:
        """Align a single face in the image.

        Args:
            image: Color image
            gray: Grayscale image
            rect: Face rectangle

        Returns:
            Aligned face image
        """
        return self.face_aligner.align(image, gray, rect)

    def process_single_image(
        self,
        image_path: str,
        output_path: str,
        metadata: Optional[MetadataManager] = None,
        skip_existing: bool = True
    ) -> bool:
        """Process a single image: detect face, align, and save.

        Args:
            image_path: Path to input image
            output_path: Path to save aligned face
            metadata: Optional metadata manager
            skip_existing: Skip if output already exists

        Returns:
            True if successful, False otherwise
        """
        image_name = os.path.basename(image_path)

        # Skip if already processed
        if skip_existing and os.path.exists(output_path):
            if self.verbose:
                print(f"Skipping {image_name} - already processed")
            return True

        # Skip if marked as permanently failed
        if metadata and metadata.is_permanently_failed(image_name):
            if self.verbose:
                print(f"Skipping {image_name} - permanently failed")
            return False

        # Load and preprocess
        image, gray = self.load_and_preprocess_image(image_path)
        if image is None:
            if metadata:
                metadata.mark_failed(image_name, "Failed to load image", "processing")
            return False

        try:
            # Check for manual landmarks first
            if metadata and metadata.has_manual_landmarks(image_name):
                if self.verbose:
                    print(f"Using manual landmarks for {image_name}")
                landmarks = metadata.get_manual_landmarks(image_name)

                # Landmarks are stored in original image coordinates, but the image
                # has been resized to resize_width. We need to scale the landmarks.
                # Get the original image to calculate scale factor
                original_image = Image.open(image_path).convert('RGB')
                original_image = ImageOps.exif_transpose(original_image)
                original_width = original_image.width

                # Calculate scale factor
                scale_factor = self.resize_width / original_width if original_width > self.resize_width else 1.0

                # Scale landmarks to match resized image
                scaled_left_eye = (
                    int(landmarks['left_eye'][0] * scale_factor),
                    int(landmarks['left_eye'][1] * scale_factor)
                )
                scaled_right_eye = (
                    int(landmarks['right_eye'][0] * scale_factor),
                    int(landmarks['right_eye'][1] * scale_factor)
                )

                aligned_face = self.face_aligner.align_with_manual_landmarks(
                    image,
                    scaled_left_eye,
                    scaled_right_eye
                )
            else:
                # Automatic face detection and alignment
                # Detect faces
                rects = self.detect_faces(gray)
                if not rects:
                    print(f"No faces detected in {image_name}")
                    if metadata:
                        metadata.mark_failed(image_name, "No faces detected", "processing")
                    return False

                # Select face
                rect = self.select_face(rects, image_name, metadata)
                if rect is None:
                    print(f"Could not select face in {image_name}")
                    if metadata:
                        metadata.mark_failed(image_name, "Face selection failed", "processing")
                    return False

                # Align face
                aligned_face = self.align_face(image, gray, rect)

            # Save aligned face
            cv2.imwrite(output_path, aligned_face)

            if metadata:
                metadata.mark_processed(image_name)
                # Remove from failed images if it was there
                if image_name in metadata.data.get("failed_images", {}):
                    del metadata.data["failed_images"][image_name]
                    metadata.save()

            if self.verbose:
                print(f"Successfully processed {image_name}")

            return True

        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            if metadata:
                metadata.mark_failed(image_name, str(e), "processing")
            return False

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        metadata: Optional[MetadataManager] = None,
        skip_existing: bool = True,
        parallel: bool = True,
        num_workers: Optional[int] = None
    ) -> Tuple[int, int]:
        """Process all images in a directory.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            metadata: Optional metadata manager
            skip_existing: Skip already processed images
            parallel: Use parallel processing (default: True)
            num_workers: Number of parallel workers (default: CPU count - 1)

        Returns:
            Tuple of (successful_count, failed_count)
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get all image paths
        patterns = ['*.jpg', '*.jpeg', '*.png', '*.HEIC', '*.heic', '*.JPG', '*.JPEG', '*.PNG']
        paths = []
        for pattern in patterns:
            paths.extend(glob(os.path.join(input_dir, pattern)))

        # Filter out non-image files
        paths = [p for p in paths if not p.endswith('.MP4') and not p.endswith('.mp4')]

        if self.verbose:
            print(f"Found {len(paths)} images to process")

        # Prepare tasks with all required parameters
        tasks = []
        for image_path in paths:
            base_name = os.path.basename(image_path)
            output_name = os.path.splitext(base_name)[0] + '.jpg'
            output_path = os.path.join(output_dir, output_name)

            # Pack all parameters needed by worker
            task_data = (
                image_path,
                output_path,
                skip_existing,
                self.resize_width,
                self.upsample_times,
                self.auto_select_largest
            )
            tasks.append(task_data)

        # Process images
        if parallel and len(tasks) > 1:
            # Use parallel processing with proper initialization
            if num_workers is None:
                num_workers = max(1, cpu_count() - 1)  # Leave one CPU free

            if self.verbose:
                print(f"Using {num_workers} parallel workers")
                print(f"Processing {len(tasks)} images...")

            # Initialize pool with worker initialization function
            init_args = (
                self.landmarks_model_path,
                self.face_width,
                self.face_height,
                self.left_eye_position,
                self.resize_width,
                self.upsample_times
            )

            with Pool(processes=num_workers, initializer=_init_worker, initargs=init_args) as pool:
                # Use imap with tqdm for progress bar
                results = list(tqdm(
                    pool.imap(_process_image_worker, tasks),
                    total=len(tasks),
                    desc="Processing faces",
                    unit="image"
                ))

            # Process results and update metadata
            successful = 0
            failed = 0
            for image_name, success, error_msg in results:
                if success:
                    successful += 1
                    if metadata:
                        metadata.mark_processed(image_name)
                else:
                    failed += 1
                    if metadata and error_msg:
                        metadata.mark_failed(image_name, error_msg, "processing")
                    if self.verbose:
                        print(f"Failed: {image_name} - {error_msg}")

        else:
            # Sequential processing
            if self.verbose:
                print("Using sequential processing")

            successful = 0
            failed = 0
            for task_data in tasks:
                image_path = task_data[0]
                output_path = task_data[1]
                if self.process_single_image(image_path, output_path, metadata, skip_existing):
                    successful += 1
                else:
                    failed += 1

        print(f"\nProcessing complete: {successful} successful, {failed} failed")
        return successful, failed

    @staticmethod
    def get_date_taken(image_path: str) -> Optional[str]:
        """Extract date taken from EXIF data.

        Args:
            image_path: Path to image file

        Returns:
            Date string in format "YYYY:MM:DD HH:MM:SS" or None
        """
        try:
            exif = Image.open(image_path).getexif()
            if not exif:
                print(f'Image {image_path} does not have EXIF data.')
                return None
            return exif.get(306)  # DateTime tag
        except Exception as e:
            print(f"Error reading EXIF from {image_path}: {e}")
            return None

    @staticmethod
    def get_images_with_dates(
        image_dir: str,
        metadata: Optional[MetadataManager] = None,
        original_dir: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Get all images with their dates, sorted chronologically.

        Args:
            image_dir: Directory containing aligned images
            metadata: Optional metadata manager to filter failed images
            original_dir: Directory containing original images with EXIF data.
                         If provided, dates will be read from original images.

        Returns:
            List of (date_string, image_path) tuples sorted by date
        """
        # Get all image paths
        paths = glob(os.path.join(image_dir, '*.jpg'))

        dates = []
        for aligned_path in paths:
            image_name = os.path.basename(aligned_path)

            # Skip failed images
            if metadata and metadata.is_failed(image_name):
                continue

            # Try to read date from original image if available
            date = None
            if original_dir:
                # Try to find corresponding original image
                # Remove .jpg extension and try common image extensions
                base_name = os.path.splitext(image_name)[0]
                original_patterns = [
                    os.path.join(original_dir, f'{base_name}.jpg'),
                    os.path.join(original_dir, f'{base_name}.JPG'),
                    os.path.join(original_dir, f'{base_name}.jpeg'),
                    os.path.join(original_dir, f'{base_name}.JPEG'),
                    os.path.join(original_dir, f'{base_name}.png'),
                    os.path.join(original_dir, f'{base_name}.PNG'),
                    os.path.join(original_dir, f'{base_name}.HEIC'),
                    os.path.join(original_dir, f'{base_name}.heic'),
                ]

                for original_path in original_patterns:
                    if os.path.exists(original_path):
                        date = ImageProcessor.get_date_taken(original_path)
                        if date is not None:
                            break

            # Fallback: try reading from aligned image itself
            if date is None:
                date = ImageProcessor.get_date_taken(aligned_path)

            if date is not None:
                dates.append((date, aligned_path))

        # Sort by date
        dates.sort(key=lambda x: x[0])

        return dates
