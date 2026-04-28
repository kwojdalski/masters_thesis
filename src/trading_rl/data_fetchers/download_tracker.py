"""Download tracking and rate limiting for market data fetchers.

Prevents redundant downloads and implements daily rate limiting.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


class DownloadTracker:
    """Tracks downloaded files and enforces rate limits.

    Uses JSON-based cache to remember:
    - What was downloaded (hash of parameters)
    - When it was downloaded
    - Output file path
    - File size and status

    Features:
    - Skip files downloaded within rate limit window
    - Track download history
    - Automatic cache cleanup
    """

    def __init__(
        self,
        cache_dir: str = "data/.download_cache",
        rate_limit_hours: int = 24,
    ):
        """Initialize download tracker.

        Args:
            cache_dir: Directory to store download cache
            rate_limit_hours: Hours to wait before re-downloading (default: 24)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "download_history.json"
        self.rate_limit_hours = rate_limit_hours
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        """Load download cache from JSON file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("load cache failed starting fresh err=%s", e)
                return {}
        return {}

    def _save_cache(self) -> None:
        """Save download cache to JSON file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f, indent=2, default=str)
        except IOError as e:
            logger.error("save cache failed err=%s", e)

    def _compute_hash(self, params: dict[str, Any]) -> str:
        """Compute hash of download parameters.

        Args:
            params: Download parameters (symbols, dates, source, etc.)

        Returns:
            SHA256 hash of sorted parameters
        """
        # Sort keys for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(sorted_params.encode()).hexdigest()

    def should_download(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        source: str,
        dataset: str | None = None,
        schema: str = "trades",
        timeframe: str | None = None,
        aggregate: bool = True,
    ) -> tuple[bool, str | None]:
        """Check if data should be downloaded or can be skipped.

        Args:
            symbols: Stock symbols
            start_date: Start date
            end_date: End date
            source: Data source (databento, polygon)
            dataset: Dataset identifier
            schema: Data schema
            timeframe: Timeframe for aggregation
            aggregate: Whether to aggregate

        Returns:
            Tuple of (should_download, reason)
            - should_download: True if download is needed
            - reason: Explanation string (why skip or why download)
        """
        # Create parameter hash
        params = {
            "symbols": sorted(symbols),  # Sort for consistent hashing
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
            "dataset": dataset,
            "schema": schema,
            "timeframe": timeframe,
            "aggregate": aggregate,
        }
        param_hash = self._compute_hash(params)

        # Check if in cache
        if param_hash not in self.cache:
            return True, "No previous download found"

        download_info = self.cache[param_hash]
        last_download = datetime.fromisoformat(download_info["timestamp"])
        time_since_download = datetime.now() - last_download

        # Check rate limit
        if time_since_download < timedelta(hours=self.rate_limit_hours):
            hours_remaining = self.rate_limit_hours - (time_since_download.total_seconds() / 3600)
            reason = (
                f"Downloaded {time_since_download.total_seconds() / 3600:.1f}h ago "
                f"(rate limit: {self.rate_limit_hours}h, {hours_remaining:.1f}h remaining)"
            )
            return False, reason

        return True, f"Last download was {time_since_download.total_seconds() / 3600:.1f}h ago (beyond rate limit)"

    def record_download(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        source: str,
        output_files: list[str],
        dataset: str | None = None,
        schema: str = "trades",
        timeframe: str | None = None,
        aggregate: bool = True,
        rows_downloaded: int = 0,
    ) -> None:
        """Record a successful download in the cache.

        Args:
            symbols: Stock symbols
            start_date: Start date
            end_date: End date
            source: Data source
            output_files: List of output file paths
            dataset: Dataset identifier
            schema: Data schema
            timeframe: Timeframe for aggregation
            aggregate: Whether to aggregate
            rows_downloaded: Number of rows downloaded
        """
        params = {
            "symbols": sorted(symbols),
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
            "dataset": dataset,
            "schema": schema,
            "timeframe": timeframe,
            "aggregate": aggregate,
        }
        param_hash = self._compute_hash(params)

        # Get file sizes
        file_info = []
        for filepath in output_files:
            path = Path(filepath)
            if path.exists():
                file_info.append({
                    "path": str(filepath),
                    "size_bytes": path.stat().st_size,
                    "exists": True,
                })
            else:
                file_info.append({
                    "path": str(filepath),
                    "size_bytes": 0,
                    "exists": False,
                })

        self.cache[param_hash] = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
            "dataset": dataset,
            "schema": schema,
            "timeframe": timeframe,
            "aggregate": aggregate,
            "timestamp": datetime.now().isoformat(),
            "output_files": file_info,
            "rows_downloaded": rows_downloaded,
        }

        self._save_cache()
        logger.info("record download n_symbols=%d n_rows=%d", len(symbols), rows_downloaded)

    def get_recent_downloads(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get downloads from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of download records
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []

        for param_hash, info in self.cache.items():
            download_time = datetime.fromisoformat(info["timestamp"])
            if download_time > cutoff:
                info["param_hash"] = param_hash
                info["hours_ago"] = (datetime.now() - download_time).total_seconds() / 3600
                recent.append(info)

        # Sort by timestamp (newest first)
        recent.sort(key=lambda x: x["timestamp"], reverse=True)
        return recent

    def cleanup_old_entries(self, days: int = 30) -> int:
        """Remove cache entries older than N days.

        Args:
            days: Number of days to keep

        Returns:
            Number of entries removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        to_remove = []

        for param_hash, info in self.cache.items():
            download_time = datetime.fromisoformat(info["timestamp"])
            if download_time < cutoff:
                to_remove.append(param_hash)

        for param_hash in to_remove:
            del self.cache[param_hash]

        if to_remove:
            self._save_cache()
            logger.info("cleanup cache n_removed=%d", len(to_remove))

        return len(to_remove)

    def clear_cache(self) -> None:
        """Clear all download cache."""
        self.cache = {}
        self._save_cache()
        logger.info("Download cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about cached downloads.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache:
            return {
                "total_downloads": 0,
                "total_symbols": 0,
                "total_files": 0,
                "total_size_mb": 0,
            }

        total_symbols = sum(len(info["symbols"]) for info in self.cache.values())
        total_files = sum(len(info["output_files"]) for info in self.cache.values())
        total_size = sum(
            file["size_bytes"]
            for info in self.cache.values()
            for file in info["output_files"]
        )

        return {
            "total_downloads": len(self.cache),
            "total_symbols": total_symbols,
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_download": min(
                (datetime.fromisoformat(info["timestamp"]) for info in self.cache.values()),
                default=None,
            ),
            "newest_download": max(
                (datetime.fromisoformat(info["timestamp"]) for info in self.cache.values()),
                default=None,
            ),
        }
