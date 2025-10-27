"""Metadata management for tracking image processing state."""

import json
import os
from typing import Dict, List, Optional, Set
from pathlib import Path


class MetadataManager:
    """Manages metadata for image processing, including failures and face selection."""

    def __init__(self, metadata_path: str):
        """Initialize metadata manager.

        Args:
            metadata_path: Path to metadata JSON file
        """
        self.metadata_path = metadata_path
        self.data = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata from file or create new structure."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "version": "2.0",
                "failed_images": {},
                "face_overrides": {},
                "downloaded_images": {},
                "processed_images": set(),
                "manual_landmarks": {},
                "permanently_failed": set()
            }

    def save(self):
        """Save metadata to file."""
        # Convert sets to lists for JSON serialization
        data_copy = self.data.copy()
        if isinstance(data_copy.get("processed_images"), set):
            data_copy["processed_images"] = list(data_copy["processed_images"])
        if isinstance(data_copy.get("permanently_failed"), set):
            data_copy["permanently_failed"] = list(data_copy["permanently_failed"])

        with open(self.metadata_path, 'w') as f:
            json.dump(data_copy, f, indent=2)

    def mark_failed(self, image_name: str, reason: str, category: str = "processing"):
        """Mark an image as failed.

        Args:
            image_name: Base name of the image file
            reason: Reason for failure
            category: Category of failure (capture_issue, needs_rename, processing)
        """
        if "failed_images" not in self.data:
            self.data["failed_images"] = {}

        self.data["failed_images"][image_name] = {
            "reason": reason,
            "category": category
        }
        self.save()

    def is_failed(self, image_name: str) -> bool:
        """Check if an image is marked as failed.

        Args:
            image_name: Base name of the image file

        Returns:
            True if image is marked as failed
        """
        return image_name in self.data.get("failed_images", {})

    def get_failed_images(self, category: Optional[str] = None) -> Dict[str, Dict]:
        """Get all failed images, optionally filtered by category.

        Args:
            category: Optional category filter

        Returns:
            Dictionary of failed images
        """
        failed = self.data.get("failed_images", {})
        if category:
            return {k: v for k, v in failed.items() if v.get("category") == category}
        return failed

    def set_face_override(self, image_name: str, face_index: int):
        """Set manual override for face selection.

        Args:
            image_name: Base name of the image file
            face_index: Index of the face to use (0-based)
        """
        if "face_overrides" not in self.data:
            self.data["face_overrides"] = {}

        self.data["face_overrides"][image_name] = face_index
        self.save()

    def get_face_index(self, image_name: str, default: int = 0) -> int:
        """Get face index for an image.

        Args:
            image_name: Base name of the image file
            default: Default index if no override exists

        Returns:
            Face index to use
        """
        return self.data.get("face_overrides", {}).get(image_name, default)

    def mark_processed(self, image_name: str):
        """Mark an image as successfully processed.

        Args:
            image_name: Base name of the image file
        """
        if "processed_images" not in self.data:
            self.data["processed_images"] = set()
        elif isinstance(self.data["processed_images"], list):
            self.data["processed_images"] = set(self.data["processed_images"])

        self.data["processed_images"].add(image_name)
        self.save()

    def is_processed(self, image_name: str) -> bool:
        """Check if an image has been processed.

        Args:
            image_name: Base name of the image file

        Returns:
            True if image has been processed
        """
        processed = self.data.get("processed_images", set())
        if isinstance(processed, list):
            processed = set(processed)
        return image_name in processed

    def mark_downloaded(self, image_name: str, photo_id: str, download_date: str):
        """Mark an image as downloaded from Google Photos.

        Args:
            image_name: Base name of the downloaded image file
            photo_id: Google Photos media item ID
            download_date: ISO format date string
        """
        if "downloaded_images" not in self.data:
            self.data["downloaded_images"] = {}

        self.data["downloaded_images"][image_name] = {
            "photo_id": photo_id,
            "download_date": download_date
        }
        self.save()

    def is_downloaded(self, photo_id: str) -> bool:
        """Check if a photo has already been downloaded.

        Args:
            photo_id: Google Photos media item ID

        Returns:
            True if photo has been downloaded
        """
        downloaded = self.data.get("downloaded_images", {})
        return any(v.get("photo_id") == photo_id for v in downloaded.values())

    @classmethod
    def migrate_from_legacy(cls, metadata_path: str, project_root: str) -> 'MetadataManager':
        """Migrate from legacy text files to new metadata format.

        Args:
            metadata_path: Path to new metadata JSON file
            project_root: Root directory of the project

        Returns:
            MetadataManager instance with migrated data
        """
        manager = cls(metadata_path)

        # Migrate capture_issues.txt
        capture_issues_path = os.path.join(project_root, "capture_issues.txt")
        if os.path.exists(capture_issues_path):
            with open(capture_issues_path, 'r') as f:
                for line in f:
                    image_name = line.strip()
                    if image_name:
                        manager.mark_failed(image_name, "Capture issue", "capture_issue")
            # Backup old file
            os.rename(capture_issues_path, capture_issues_path + ".backup")

        # Migrate needs_to_be_renamed.txt
        rename_path = os.path.join(project_root, "needs_to_be_renamed.txt")
        if os.path.exists(rename_path):
            with open(rename_path, 'r') as f:
                for line in f:
                    image_name = line.strip()
                    if image_name:
                        manager.mark_failed(image_name, "Needs renaming", "needs_rename")
            os.rename(rename_path, rename_path + ".backup")

        # Migrate script_failures.txt
        failures_path = os.path.join(project_root, "script_failures.txt")
        if os.path.exists(failures_path):
            with open(failures_path, 'r') as f:
                for line in f:
                    image_name = line.strip()
                    if image_name:
                        manager.mark_failed(image_name, "Processing failed", "processing")
            os.rename(failures_path, failures_path + ".backup")

        # Migrate rectangle_mapping.json
        rect_mapping_path = os.path.join(project_root, "rectangle_mapping.json")
        if os.path.exists(rect_mapping_path):
            with open(rect_mapping_path, 'r') as f:
                rect_mapping = json.load(f)
                for image_name, face_index in rect_mapping.items():
                    manager.set_face_override(image_name, face_index)
            os.rename(rect_mapping_path, rect_mapping_path + ".backup")

        manager.save()
        print(f"Migration complete. Legacy files backed up with .backup extension")
        return manager

    def set_manual_landmarks(self, image_name: str, left_eye: tuple, right_eye: tuple):
        """Set manual eye landmarks for an image.

        Args:
            image_name: Base name of the image file
            left_eye: (x, y) coordinates of left eye center
            right_eye: (x, y) coordinates of right eye center
        """
        if "manual_landmarks" not in self.data:
            self.data["manual_landmarks"] = {}

        self.data["manual_landmarks"][image_name] = {
            "left_eye": list(left_eye),
            "right_eye": list(right_eye)
        }
        self.save()

    def get_manual_landmarks(self, image_name: str) -> Optional[Dict]:
        """Get manual eye landmarks for an image.

        Args:
            image_name: Base name of the image file

        Returns:
            Dictionary with 'left_eye' and 'right_eye' keys, or None if not set
        """
        return self.data.get("manual_landmarks", {}).get(image_name)

    def has_manual_landmarks(self, image_name: str) -> bool:
        """Check if an image has manual landmarks.

        Args:
            image_name: Base name of the image file

        Returns:
            True if manual landmarks exist
        """
        return image_name in self.data.get("manual_landmarks", {})

    def mark_permanently_failed(self, image_name: str):
        """Mark an image as permanently failed (unusable, will not show in annotation).

        Args:
            image_name: Base name of the image file
        """
        if "permanently_failed" not in self.data:
            self.data["permanently_failed"] = set()
        elif isinstance(self.data["permanently_failed"], list):
            self.data["permanently_failed"] = set(self.data["permanently_failed"])

        self.data["permanently_failed"].add(image_name)
        self.save()

    def is_permanently_failed(self, image_name: str) -> bool:
        """Check if an image is marked as permanently failed.

        Args:
            image_name: Base name of the image file

        Returns:
            True if image is permanently failed
        """
        perm_failed = self.data.get("permanently_failed", set())
        if isinstance(perm_failed, list):
            perm_failed = set(perm_failed)
        return image_name in perm_failed

    def get_images_needing_annotation(self) -> List[str]:
        """Get list of images that need manual annotation.

        Returns images that:
        - Are marked as failed
        - Don't have manual landmarks yet
        - Are not permanently failed

        Returns:
            List of image names needing annotation
        """
        failed = set(self.data.get("failed_images", {}).keys())
        has_landmarks = set(self.data.get("manual_landmarks", {}).keys())
        perm_failed = self.data.get("permanently_failed", set())
        if isinstance(perm_failed, list):
            perm_failed = set(perm_failed)

        return list(failed - has_landmarks - perm_failed)
