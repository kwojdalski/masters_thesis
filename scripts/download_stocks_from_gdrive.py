#!/usr/bin/env python3
"""Download stock data folder from Google Drive into data/raw/stocks.

Usage:
  export GDRIVE_STOCKS_URL="https://drive.google.com/drive/folders/..."
  source .venv/bin/activate
  python scripts/download_stocks_from_gdrive.py

Optional:
  python scripts/download_stocks_from_gdrive.py --url "$GDRIVE_STOCKS_URL" --dest data/raw/stocks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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
    return parser.parse_args()


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
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    try:
        import gdown
    except ImportError:
        print(
            "[ERROR] Missing dependency 'gdown'. Install it in the active environment: "
            "pip install gdown",
            file=sys.stderr,
        )
        return 3

    destination = args.dest.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading Google Drive folder into: {destination}")
    print("[INFO] URL source: " + ("--url" if args.url else "GDRIVE_STOCKS_URL"))

    downloaded_files = gdown.download_folder(
        url=url,
        output=str(destination),
        quiet=args.quiet,
        remaining_ok=True,
    )

    if not downloaded_files:
        print(
            "[WARN] No files were downloaded. Verify folder sharing permissions and URL."
        )
        return 1

    print(f"[INFO] Download complete. Files downloaded: {len(downloaded_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
