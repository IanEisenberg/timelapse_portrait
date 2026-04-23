"""Tests for Google Photos sync module (Picker API)."""

import os

from src.google_photos import GooglePhotosDownloader


def _picker_item(filename, url, item_id):
    """Helper: build a Picker API media item with the nested mediaFile structure."""
    return {"id": item_id, "mediaFile": {"filename": filename, "baseUrl": url}}


def test_get_new_items_filters_existing(tmp_path):
    """Items whose filename already exists in output_dir should be excluded."""
    (tmp_path / "IMG_001.jpg").touch()
    (tmp_path / "IMG_002.HEIC").touch()

    api_items = [
        _picker_item("IMG_001.jpg", "https://example.com/1", "1"),
        _picker_item("IMG_002.HEIC", "https://example.com/2", "2"),
        _picker_item("IMG_003.jpg", "https://example.com/3", "3"),
    ]

    downloader = GooglePhotosDownloader.__new__(GooglePhotosDownloader)
    downloader.output_dir = str(tmp_path)

    new_items = downloader._filter_new_items(api_items)

    assert len(new_items) == 1
    assert new_items[0]["mediaFile"]["filename"] == "IMG_003.jpg"


def test_get_new_items_case_insensitive_extension(tmp_path):
    """Filtering should match regardless of extension case."""
    (tmp_path / "IMG_001.JPG").touch()

    api_items = [
        _picker_item("IMG_001.jpg", "https://example.com/1", "1"),
    ]

    downloader = GooglePhotosDownloader.__new__(GooglePhotosDownloader)
    downloader.output_dir = str(tmp_path)

    new_items = downloader._filter_new_items(api_items)

    assert len(new_items) == 0


def test_get_new_items_empty_dir(tmp_path):
    """All items are new if output_dir is empty."""
    api_items = [
        _picker_item("IMG_001.jpg", "https://example.com/1", "1"),
        _picker_item("IMG_002.jpg", "https://example.com/2", "2"),
    ]

    downloader = GooglePhotosDownloader.__new__(GooglePhotosDownloader)
    downloader.output_dir = str(tmp_path)

    new_items = downloader._filter_new_items(api_items)

    assert len(new_items) == 2
