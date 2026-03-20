#!/usr/bin/env python3
"""Download stock data folder from Google Drive into data/raw/stocks.

Usage:
  export GDRIVE_STOCKS_URL="https://drive.google.com/drive/folders/..."
  python scripts/download_stocks_from_gdrive.py

  # Interactive file picker (requires fzf + Drive API credentials)
  python scripts/download_stocks_from_gdrive.py --interactive

Optional:
  python scripts/download_stocks_from_gdrive.py --url "$GDRIVE_STOCKS_URL" --dest data/raw/stocks

Interactive mode authentication (same env vars as upload script):
  GDRIVE_SERVICE_ACCOUNT_FILE  — path to service-account JSON key
  GDRIVE_CLIENT_SECRET_FILE    — path to OAuth client secrets JSON (alternative)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logger import setup_component_logger

logger = setup_component_logger("download_stocks_from_gdrive")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Google Drive folder into data/raw/stocks."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Google Drive folder URL. Defaults to GDRIVE_STOCKS_URL env var.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/stocks"),
        help="Destination directory (default: data/raw/stocks).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce download output verbosity.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pick individual files from the Drive folder with fzf (requires fzf + Drive API credentials).",
    )
    return parser.parse_args()


def extract_folder_id(url: str) -> str:
    patterns = [
        r"/folders/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not parse folder ID from URL: {url}")


def get_drive_service():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google.oauth2.service_account import Credentials as SACredentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "Missing Google Drive dependencies. Install with: "
            "uv add google-api-python-client google-auth google-auth-oauthlib"
        ) from exc

    sa_file = os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE")
    if sa_file:
        creds = SACredentials.from_service_account_file(sa_file, scopes=SCOPES)
        return build("drive", "v3", credentials=creds)

    client_secret = os.getenv("GDRIVE_CLIENT_SECRET_FILE")
    if not client_secret:
        raise EnvironmentError(
            "Set either GDRIVE_SERVICE_ACCOUNT_FILE or GDRIVE_CLIENT_SECRET_FILE."
        )

    token_file = os.getenv(
        "GDRIVE_TOKEN_FILE", str(Path("scripts/.gdrive_download_token.json"))
    )
    creds = None
    token_path = Path(token_file)
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("drive", "v3", credentials=creds)


def download_file_via_api(service, file_id: str, dest_path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload
    import io

    request = service.files().get_media(fileId=file_id)
    with dest_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def list_folder_files(service, folder_id: str) -> list[dict]:
    files = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=(
                    f"'{folder_id}' in parents and trashed = false "
                    "and mimeType != 'application/vnd.google-apps.folder'"
                ),
                fields="nextPageToken, files(id, name)",
                pageSize=1000,
                pageToken=page_token,
            )
            .execute()
        )
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def pick_files_with_fzf(files: list[dict]) -> list[dict]:
    if not shutil.which("fzf"):
        raise RuntimeError("fzf not found. Install with: brew install fzf")

    entries = "\n".join(f["name"] for f in files)
    result = subprocess.run(
        ["fzf", "--multi", "--prompt=Select files to download (TAB=multi-select, ENTER=confirm)> "],
        input=entries,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    selected_names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return [f for f in files if f["name"] in selected_names]


def _resolve_url(cli_url: str | None) -> str:
    url = cli_url or os.getenv("GDRIVE_STOCKS_URL")
    if not url:
        raise ValueError(
            "Missing Google Drive URL. Set GDRIVE_STOCKS_URL or pass --url."
        )
    return url


def main() -> int:
    args = parse_args()

    try:
        url = _resolve_url(args.url)
    except ValueError as exc:
        logger.error(exc)
        return 2

    destination = args.dest.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        try:
            folder_id = extract_folder_id(url)
            service = get_drive_service()
            remote_files = list_folder_files(service, folder_id)
        except Exception as exc:
            logger.error(exc)
            return 2

        if not remote_files:
            logger.warning("No files found in the Drive folder.")
            return 1

        selected = pick_files_with_fzf(remote_files)
        if not selected:
            logger.info("No files selected. Exiting.")
            return 0

        logger.info("Downloading %d file(s) into: %s", len(selected), destination)
        downloaded = 0
        for f in selected:
            out_path = destination / f["name"]
            logger.info("Downloading %s...", f["name"])
            try:
                download_file_via_api(service, f["id"], out_path)
                downloaded += 1
                logger.info("OK: %s", f["name"])
            except Exception as exc:
                logger.error("FAIL: %s — %s", f["name"], exc)

        logger.info("Download complete. Successful: %d/%d", downloaded, len(selected))
        return 0 if downloaded == len(selected) else 1

    # Use Drive API (authenticated) if credentials are available, otherwise fall
    # back to gdown which requires the folder to be publicly shared.
    has_credentials = bool(
        os.getenv("GDRIVE_SERVICE_ACCOUNT_FILE") or os.getenv("GDRIVE_CLIENT_SECRET_FILE")
    )

    if has_credentials:
        try:
            folder_id = extract_folder_id(url)
            service = get_drive_service()
            remote_files = list_folder_files(service, folder_id)
        except Exception as exc:
            logger.error(exc)
            return 2

        if not remote_files:
            logger.warning("No files found in the Drive folder.")
            return 1

        logger.info("Downloading %d file(s) into: %s", len(remote_files), destination)
        downloaded = 0
        for f in remote_files:
            out_path = destination / f["name"]
            logger.info("Downloading %s...", f["name"])
            try:
                download_file_via_api(service, f["id"], out_path)
                downloaded += 1
                logger.info("OK: %s", f["name"])
            except Exception as exc:
                logger.error("FAIL: %s — %s", f["name"], exc)

        logger.info("Download complete. Successful: %d/%d", downloaded, len(remote_files))
        return 0 if downloaded == len(remote_files) else 1

    try:
        import gdown
    except ImportError:
        logger.error("Missing dependency 'gdown'. Install with: uv add gdown")
        return 3

    logger.info("Downloading Google Drive folder into: %s", destination)
    logger.info("URL source: %s", "--url" if args.url else "GDRIVE_STOCKS_URL")

    downloaded_files = gdown.download_folder(
        url=url,
        output=str(destination),
        quiet=args.quiet,
        remaining_ok=True,
    )

    if not downloaded_files:
        logger.warning("No files were downloaded. Verify folder sharing permissions and URL.")
        return 1

    logger.info("Download complete. Files downloaded: %d", len(downloaded_files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
