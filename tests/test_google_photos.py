"""Tests for Google Photos sync module."""

import os

from src.google_photos import GooglePhotosDownloader


def test_get_new_items_filters_existing(tmp_path):
    """Items whose filename already exists in output_dir should be excluded."""
    (tmp_path / "IMG_001.jpg").touch()
    (tmp_path / "IMG_002.HEIC").touch()

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
