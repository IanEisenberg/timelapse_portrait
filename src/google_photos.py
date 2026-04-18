"""Google Photos album sync module."""

import os
from typing import List, Dict

import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/photoslibrary.readonly"]


class GooglePhotosDownloader:
    """Downloads new photos from a Google Photos album."""

    def __init__(
        self,
        credentials_path: str,
        token_path: str,
        album_name: str,
        output_dir: str,
        verbose: bool = True
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.album_name = album_name
        self.output_dir = output_dir
        self.verbose = verbose
        self.service = None

    def authenticate(self):
        """Authenticate with Google Photos API.

        Uses existing token.json if valid, otherwise runs interactive OAuth flow.
        """
        creds = None

        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_path}\n"
                        "Download it from Google Cloud Console (OAuth 2.0 Client ID, Desktop type)."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token_path, "w") as f:
                f.write(creds.to_json())

        self.service = build("photoslibrary", "v1", credentials=creds, static_discovery=False)

    def _find_album_id(self) -> str:
        """Find album ID by name."""
        page_token = None
        while True:
            results = self.service.albums().list(
                pageSize=50, pageToken=page_token
            ).execute()

            for album in results.get("albums", []):
                if album["title"] == self.album_name:
                    return album["id"]

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        raise ValueError(f"Album not found: '{self.album_name}'")

    def _list_album_items(self, album_id: str) -> List[Dict]:
        """List all media items in an album (handles pagination)."""
        items = []
        page_token = None

        while True:
            body = {"albumId": album_id, "pageSize": 100}
            if page_token:
                body["pageToken"] = page_token

            results = self.service.mediaItems().search(body=body).execute()

            items.extend(results.get("mediaItems", []))

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        return items

    def _filter_new_items(self, items: List[Dict]) -> List[Dict]:
        """Filter out items that already exist locally (case-insensitive on stem)."""
        existing_stems = set()
        for f in os.listdir(self.output_dir):
            stem = os.path.splitext(f)[0].lower()
            existing_stems.add(stem)

        new_items = []
        for item in items:
            stem = os.path.splitext(item["filename"])[0].lower()
            if stem not in existing_stems:
                new_items.append(item)

        return new_items

    def _download_item(self, item: Dict) -> bool:
        """Download a single media item at original quality."""
        filename = item["filename"]
        url = item["baseUrl"] + "=d"

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            output_path = os.path.join(self.output_dir, filename)
            with open(output_path, "wb") as f:
                f.write(response.content)

            if self.verbose:
                print(f"  Downloaded: {filename}")
            return True

        except Exception as e:
            print(f"  Failed to download {filename}: {e}")
            return False

    def sync(self) -> int:
        """Sync new photos from the album. Returns count of new photos downloaded."""
        if self.service is None:
            self.authenticate()

        os.makedirs(self.output_dir, exist_ok=True)

        if self.verbose:
            print(f"Looking for album: '{self.album_name}'...")

        album_id = self._find_album_id()

        if self.verbose:
            print(f"Found album. Listing items...")

        all_items = self._list_album_items(album_id)

        if self.verbose:
            print(f"Album contains {len(all_items)} items total")

        new_items = self._filter_new_items(all_items)

        if not new_items:
            if self.verbose:
                print("No new photos to download")
            return 0

        if self.verbose:
            print(f"Downloading {len(new_items)} new photos...")

        downloaded = 0
        for item in new_items:
            if self._download_item(item):
                downloaded += 1

        if self.verbose:
            print(f"Downloaded {downloaded}/{len(new_items)} photos")

        return downloaded
