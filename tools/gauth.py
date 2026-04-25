"""
Shared Google OAuth2 helper.

First run: opens a browser window to authenticate.
Subsequent runs: loads the cached token from token.json.

Required env vars (or set them in .env):
  GOOGLE_CREDENTIALS_FILE  — path to credentials.json from Google Cloud Console
                             (defaults to ./credentials.json)

Scopes used:
  - gmail.send
  - calendar.events
"""
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
]

_CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE", "credentials.json")
_TOKEN_FILE       = os.environ.get("GOOGLE_TOKEN_FILE", "token.json")

_cached_creds: Credentials | None = None


def get_credentials() -> Credentials:
    global _cached_creds

    if _cached_creds and _cached_creds.valid:
        return _cached_creds

    creds: Credentials | None = None

    if Path(_TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(_TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(_CREDENTIALS_FILE).exists():
                raise FileNotFoundError(
                    f"Google credentials file not found: {_CREDENTIALS_FILE}\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(_CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        Path(_TOKEN_FILE).write_text(creds.to_json())

    _cached_creds = creds
    return creds
