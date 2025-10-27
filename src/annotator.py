"""Manual landmark annotation tool for failed face detections."""

import os
import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageOps

from .metadata_manager import MetadataManager


class LandmarkAnnotator:
    """Interactive tool for manually annotating eye landmarks."""

    def __init__(self, original_dir: str, metadata: MetadataManager, resize_width: int = 800):
        """Initialize the annotator.

        Args:
            original_dir: Directory containing original images
            metadata: MetadataManager instance
            resize_width: Width to resize images for annotation
        """
        self.original_dir = original_dir
        self.metadata = metadata
        self.resize_width = resize_width
        self.current_image = None
        self.current_image_name = None
        self.display_image = None
        self.clicks = []
        self.scale_factor = 1.0

    def find_original_image(self, image_name: str) -> Optional[str]:
        """Find the original image file for a given base name.

        Args:
            image_name: Base name of the image (e.g., IMG_1234.jpg)

        Returns:
            Full path to original image, or None if not found
        """
        base_name = os.path.splitext(image_name)[0]
        extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.HEIC', '.heic']

        for ext in extensions:
            path = os.path.join(self.original_dir, base_name + ext)
            if os.path.exists(path):
                return path

        return None

    def load_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare image for annotation.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (original_image, display_image)
        """
        # Load image and handle EXIF orientation
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.exif_transpose(image)

        # Convert to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create display version (resized for screen)
        if image.shape[1] > self.resize_width:
            self.scale_factor = self.resize_width / image.shape[1]
            display = cv2.resize(image, (self.resize_width, int(image.shape[0] * self.scale_factor)))
        else:
            self.scale_factor = 1.0
            display = image.copy()

        return image, display

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for landmark annotation.

        Args:
            event: OpenCV mouse event
            x, y: Mouse coordinates
            flags: Event flags
            param: User data
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.clicks) < 2:
                self.clicks.append((x, y))

                # Draw circle at click location
                display_copy = self.display_image.copy()
                for i, (cx, cy) in enumerate(self.clicks):
                    color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for left, red for right
                    cv2.circle(display_copy, (cx, cy), 5, color, -1)
                    label = "Their Left" if i == 0 else "Their Right"
                    cv2.putText(display_copy, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imshow('Annotate Landmarks', display_copy)

    def annotate_image(self, image_name: str) -> bool:
        """Interactively annotate an image.

        Args:
            image_name: Base name of the image to annotate

        Returns:
            True if landmarks were saved, False if skipped/failed
        """
        # Find original image
        image_path = self.find_original_image(image_name)
        if not image_path:
            print(f"Could not find original image for {image_name}")
            return False

        # Load image
        try:
            self.current_image, self.display_image = self.load_image(image_path)
            self.current_image_name = image_name
            self.clicks = []
        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            return False

        # Setup window and mouse callback
        window_name = 'Annotate Landmarks'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Display instructions
        display_copy = self.display_image.copy()
        instructions = [
            "Click THEIR LEFT EYE (your right) - green",
            "Then click THEIR RIGHT EYE (your left) - red",
            "",
            "Press ENTER to save",
            "Press 'r' to reset clicks",
            "Press 'f' to mark as permanently failed",
            "Press ESC to skip"
        ]

        y_pos = 30
        for line in instructions:
            cv2.putText(display_copy, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_copy, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_pos += 25

        cv2.imshow(window_name, display_copy)

        # Wait for user input
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter key
                if len(self.clicks) == 2:
                    # Convert click coordinates back to original image scale
                    left_eye = (
                        int(self.clicks[0][0] / self.scale_factor),
                        int(self.clicks[0][1] / self.scale_factor)
                    )
                    right_eye = (
                        int(self.clicks[1][0] / self.scale_factor),
                        int(self.clicks[1][1] / self.scale_factor)
                    )

                    # Save landmarks
                    self.metadata.set_manual_landmarks(image_name, left_eye, right_eye)
                    print(f"Saved landmarks for {image_name}")
                    cv2.destroyWindow(window_name)
                    return True
                else:
                    print("Please click both eyes before saving")

            elif key == ord('r'):  # Reset
                self.clicks = []
                display_copy = self.display_image.copy()
                # Redraw instructions
                y_pos = 30
                for line in instructions:
                    cv2.putText(display_copy, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_copy, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    y_pos += 25
                cv2.imshow(window_name, display_copy)
                print("Clicks reset")

            elif key == ord('f'):  # Mark as permanently failed
                self.metadata.mark_permanently_failed(image_name)
                print(f"Marked {image_name} as permanently failed")
                cv2.destroyWindow(window_name)
                return False

            elif key == 27:  # ESC key
                print(f"Skipped {image_name}")
                cv2.destroyWindow(window_name)
                return False

    def annotate_all_failed(self) -> Tuple[int, int, int]:
        """Annotate all images that need annotation.

        Returns:
            Tuple of (annotated_count, skipped_count, failed_count)
        """
        images_to_annotate = self.metadata.get_images_needing_annotation()

        if not images_to_annotate:
            print("No images need annotation!")
            return (0, 0, 0)

        print(f"Found {len(images_to_annotate)} images needing annotation")
        print()

        annotated = 0
        skipped = 0
        perm_failed = 0

        for i, image_name in enumerate(images_to_annotate, 1):
            print(f"\n[{i}/{len(images_to_annotate)}] {image_name}")

            # Show failure reason if available
            failed_info = self.metadata.data.get("failed_images", {}).get(image_name)
            if failed_info:
                print(f"  Reason: {failed_info.get('reason', 'Unknown')}")

            result = self.annotate_image(image_name)
            if result:
                annotated += 1
            elif self.metadata.is_permanently_failed(image_name):
                perm_failed += 1
            else:
                skipped += 1

        cv2.destroyAllWindows()

        print(f"\n\nAnnotation complete!")
        print(f"  Annotated: {annotated}")
        print(f"  Permanently failed: {perm_failed}")
        print(f"  Skipped: {skipped}")

        return (annotated, skipped, perm_failed)
