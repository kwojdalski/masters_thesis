"""Cache directory utilities: inspect and clear cached feature/data artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

# The only cache directory with a fixed, well-known default location (see
# EnvConfig.feature_cache_dir in trading_rl/config.py). prepared_data_dir and
# memmap_dir are per-experiment configured with no fixed default, so they are
# not included here -- clearing them blindly could delete arbitrary experiment
# output rather than a cache.
DEFAULT_CACHE_DIRS = (".cache/feature_transformation",)


def get_cache_info(path: str) -> dict:
    """Return existence, total file size (MB), and entry count for a cache directory."""
    cache_dir = Path(path)
    if not cache_dir.exists():
        return {"exists": False, "size_mb": 0, "num_files": 0}
    entries = list(cache_dir.rglob("*"))
    size_bytes = sum(f.stat().st_size for f in entries if f.is_file())
    return {
        "exists": True,
        "size_mb": size_bytes / (1024 * 1024),
        "num_files": len(entries),
    }


def clear_all_caches(path: str | None = None) -> None:
    """Remove a specific cache directory, or all default project cache dirs if none given."""
    paths = [path] if path is not None else list(DEFAULT_CACHE_DIRS)
    for p in paths:
        cache_dir = Path(p)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
