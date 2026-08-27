"""Regression test: get_recent_downloads must not mutate self.cache (issue #465)."""

from __future__ import annotations

from datetime import UTC, datetime

from trading_rl.data_fetchers.download_tracker import DownloadTracker


class TestGetRecentDownloadsDoesNotMutateCache:
    def test_cache_entry_unchanged_after_query(self, tmp_path):
        tracker = DownloadTracker(cache_dir=str(tmp_path))
        tracker.cache["abc123"] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbols": ["AAPL"],
        }

        recent = tracker.get_recent_downloads(hours=24)

        assert len(recent) == 1
        assert "param_hash" in recent[0]
        assert "hours_ago" in recent[0]
        # the underlying cache record must remain exactly as stored -- no
        # param_hash/hours_ago leaking into the persisted structure
        assert tracker.cache["abc123"] == {
            "timestamp": recent[0]["timestamp"],
            "symbols": ["AAPL"],
        }

    def test_hours_ago_does_not_get_persisted_to_disk(self, tmp_path):
        tracker = DownloadTracker(cache_dir=str(tmp_path))
        tracker.cache["abc123"] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "symbols": ["AAPL"],
        }

        tracker.get_recent_downloads(hours=24)
        tracker._save_cache()

        reloaded = DownloadTracker(cache_dir=str(tmp_path))
        assert "hours_ago" not in reloaded.cache["abc123"]
        assert "param_hash" not in reloaded.cache["abc123"]
