"""trading_rl.data — data loading, preparation, validation, and HFT utilities."""

from __future__ import annotations

from trading_rl.data.loading import (
    PreparedDataset,
    download_trading_data,
    load_trading_data,
)
from trading_rl.data.validation import (
    DataValidator,
    validate_prepared_data,
)
from trading_rl.data.hft import (
    _HFT_MIN_TIMESTAMP_GAP_NS,
    _deduplicate_hft_index_single,
    _derive_close_hft_single,
    _map_splits,
    ensure_close_column_for_hft,
    ensure_unique_index_for_hft_tradingenv,
)
from trading_rl.data.cache import (
    _cached_split_rows,
    _config_cache_signature,
    _data_source_signature,
    _expected_cached_split_rows,
    _feature_cache_key,
    _memmap_cache_compatible,
    _parquet_cache_exists,
    _prepared_cache_compatible,
    _prepared_cache_metadata_path,
    _write_prepared_cache_metadata,
)
from trading_rl.data.preparation import (
    PrepareDataConfig,
    _build_per_day_splits,
    _build_pooled_splits,
    _build_single_symbol_splits,
    _finalize_splits,
    _make_dataset,
    _resolve_feature_pipeline,
    build_prepared_dataset,
    prepare_data,
)

__all__ = [
    # loading
    "PreparedDataset",
    "download_trading_data",
    "load_trading_data",
    # validation
    "DataValidator",
    "validate_prepared_data",
    # hft
    "_HFT_MIN_TIMESTAMP_GAP_NS",
    "_deduplicate_hft_index_single",
    "_derive_close_hft_single",
    "_map_splits",
    "ensure_close_column_for_hft",
    "ensure_unique_index_for_hft_tradingenv",
    # cache
    "_cached_split_rows",
    "_config_cache_signature",
    "_data_source_signature",
    "_expected_cached_split_rows",
    "_feature_cache_key",
    "_memmap_cache_compatible",
    "_parquet_cache_exists",
    "_prepared_cache_compatible",
    "_prepared_cache_metadata_path",
    "_write_prepared_cache_metadata",
    # preparation
    "PrepareDataConfig",
    "_build_per_day_splits",
    "_build_pooled_splits",
    "_build_single_symbol_splits",
    "_finalize_splits",
    "_make_dataset",
    "_resolve_feature_pipeline",
    "build_prepared_dataset",
    "prepare_data",
]
