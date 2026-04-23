"""Google Photos Picker-based sync module.

Uses the Google Photos Picker API (2025+) which requires interactive photo
selection in the browser. The Library API's photoslibrary.readonly scope was
deprecated March 2025 and no longer works for accessing existing albums.
"""

import os
import time
import webbrowser
from typing import List, Dict

import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


SCOPES = ["https://www.googleapis.com/auth/photospicker.mediaitems.readonly"]
PICKER_API = "https://photospicker.googleapis.com/v1"


class GooglePhotosDownloader:
    """Downloads new photos from Google Photos via the interactive Picker API."""

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
        self.album_name = album_name  # kept for display/reference only
        self.output_dir = output_dir
        self.verbose = verbose
        self.creds = None

    def authenticate(self):
        """Authenticate with Google Photos Picker API.

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

        self.creds = creds

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.creds.token}"}

    def _create_session(self) -> dict:
        """Create a new Picker session and return the session object."""
        resp = requests.post(f"{PICKER_API}/sessions", headers=self._auth_headers(), json={})
        resp.raise_for_status()
        return resp.json()

    def _wait_for_selection(self, session_id: str, timeout_seconds: int = 600) -> bool:
        """Poll until the user finishes selecting photos in the browser.

        Returns True if selection completed, False if timed out.
        """
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            resp = requests.get(
                f"{PICKER_API}/sessions/{session_id}",
                headers=self._auth_headers()
            )
            resp.raise_for_status()
            if resp.json().get("mediaItemsSet"):
                return True
            time.sleep(5)
        return False

    def _list_picker_items(self, session_id: str) -> List[Dict]:
        """List all media items selected in the Picker session (handles pagination)."""
        items = []
        page_token = None

        while True:
            params = {"sessionId": session_id, "pageSize": 100}
            if page_token:
                params["pageToken"] = page_token

            resp = requests.get(
                f"{PICKER_API}/mediaItems",
                headers=self._auth_headers(),
                params=params
            )
            resp.raise_for_status()
            data = resp.json()
            items.extend(data.get("mediaItems", []))

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        return items

    def _delete_session(self, session_id: str):
        """Clean up the Picker session."""
        requests.delete(f"{PICKER_API}/sessions/{session_id}", headers=self._auth_headers())

    def _filter_new_items(self, items: List[Dict]) -> List[Dict]:
        """Filter out items that already exist locally (case-insensitive stem match).

        Picker API nests filename and baseUrl under mediaFile.
        """
        existing_stems = set()
        for f in os.listdir(self.output_dir):
            if os.path.isfile(os.path.join(self.output_dir, f)):
                stem = os.path.splitext(f)[0].lower()
                existing_stems.add(stem)

        new_items = []
        for item in items:
            filename = item.get("mediaFile", {}).get("filename", "")
            stem = os.path.splitext(filename)[0].lower()
            if stem and stem not in existing_stems:
                new_items.append(item)

        return new_items

    def _download_item(self, item: Dict) -> bool:
        """Download a single media item at original quality."""
        media_file = item.get("mediaFile", {})
        filename = media_file.get("filename", "")
        base_url = media_file.get("baseUrl", "")

        if not filename or not base_url:
            print(f"  Skipping item with missing filename or URL: {item.get('id')}")
            return False

        url = base_url + "=d"

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
        """Sync photos via the interactive Picker. Returns count downloaded.

        Opens a browser window for the user to select photos, then downloads
        any selected photos not already present in output_dir.
        """
        self.authenticate()
        os.makedirs(self.output_dir, exist_ok=True)

        if self.verbose:
            print("Creating Google Photos Picker session...")

        session = self._create_session()
        session_id = session["id"]
        picker_uri = session["pickerUri"]

        print(f"\nOpening Google Photos Picker in your browser.")
        print(f"Navigate to the '{self.album_name}' album, select the photos, then click 'Allow access'.")
        print(f"Waiting up to 10 minutes for your selection...\n")
        webbrowser.open(picker_uri)

        if not self._wait_for_selection(session_id):
            self._delete_session(session_id)
            raise TimeoutError("Timed out waiting for photo selection in Picker (10 min limit).")

        all_items = self._list_picker_items(session_id)
        self._delete_session(session_id)

        if self.verbose:
            print(f"You selected {len(all_items)} photos.")

        new_items = self._filter_new_items(all_items)

        if not new_items:
            if self.verbose:
                print("All selected photos already exist locally — nothing to download.")
            return 0

        if self.verbose:
            print(f"Downloading {len(new_items)} new photos...")

        downloaded = 0
        for item in new_items:
            if self._download_item(item):
                downloaded += 1

        if self.verbose:
            print(f"Downloaded {downloaded}/{len(new_items)} photos.")

        return downloaded
