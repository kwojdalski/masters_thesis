"""Shared utilities for feature engineering."""

from typing import Any, Callable

import pandas as pd


def detect_session_breaks(
    index: pd.Index,
    threshold_hours: float = 1.0,
) -> list[int]:
    """Detect session boundaries based on time gaps.

    Session breaks are identified as time gaps exceeding the threshold,
    typically used for overnight or weekend gaps in trading data.

    Args:
        index: Series index, ideally a DatetimeIndex
        threshold_hours: Minimum gap in hours to consider a session break.
            Default: 1.0 hour (accounts for lunch gaps vs overnight gaps).

    Returns:
        List of starting indices for each session. Always includes 0 as the
        first session start. If no DatetimeIndex or no breaks detected,
        returns [0].

    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01 09:00', periods=100, freq='1min')
        >>> dates = dates.append(pd.date_range('2024-01-02 09:00', periods=50, freq='1min'))
        >>> breaks = detect_session_breaks(dates, threshold_hours=1.0)
        >>> breaks  # Session starts at index 0 and 100
        [0, 100]
    """
    if not isinstance(index, pd.DatetimeIndex):
        # Non-datetime index: no session detection possible
        return [0]

    if len(index) == 0:
        return []

    # Detect gaps exceeding threshold
    threshold = pd.Timedelta(hours=threshold_hours)
    time_gaps = index.to_series().diff() > threshold

    # Find session boundaries
    session_starts = [0]  # First session always starts at index 0
    for i, is_break in enumerate(time_gaps):
        if is_break and i > 0:
            session_starts.append(i)

    return session_starts


def apply_per_session(
    series: pd.Series,
    func: Callable[[pd.Series], pd.Series],
    threshold_hours: float = 1.0,
) -> pd.Series:
    """Apply function independently per trading session.

    Splits the series at session boundaries (detected by time gaps) and
    applies the function to each session separately. This is useful for
    rolling statistics, normalization, or any computation that should
    reset at session boundaries.

    Args:
        series: Input series with DatetimeIndex for session detection
        func: Function to apply to each session. Takes a Series, returns a Series.
        threshold_hours: Minimum gap in hours to consider a session break.

    Returns:
        Concatenated results from all sessions, with same index as input.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> dates = pd.date_range('2024-01-01 09:00', periods=50, freq='1min')
        >>> dates = dates.append(pd.date_range('2024-01-02 09:00', periods=50, freq='1min'))
        >>> s = pd.Series(np.random.randn(100), index=dates)
        >>> result = apply_per_session(s, lambda x: x.rolling(10).mean())
        >>> # Rolling mean resets at index 50 (session break)
    """
    breaks = detect_session_breaks(series.index, threshold_hours)
    if not breaks:
        return func(series)

    results = []
    for i in range(len(breaks)):
        start_idx = breaks[i]
        end_idx = breaks[i + 1] if i + 1 < len(breaks) else len(series)

        if start_idx >= len(series):
            break

        session_data = series.iloc[start_idx:end_idx]
        session_result = func(session_data)
        results.append(session_result)

    if results:
        return pd.concat(results)
    else:
        return func(series)


def apply_per_session_with_params(
    series: pd.Series,
    func: Callable[[pd.Series, Any], pd.Series],
    *func_args,
    threshold_hours: float = 1.0,
    **func_kwargs,
) -> pd.Series:
    """Apply function with parameters independently per trading session.

    Extends apply_per_session to support functions that require additional
    arguments beyond the input series.

    Args:
        series: Input series with DatetimeIndex for session detection
        func: Function to apply to each session. Takes (Series, *args, **kwargs).
        *func_args: Additional positional arguments passed to func
        threshold_hours: Minimum gap in hours to consider a session break
        **func_kwargs: Additional keyword arguments passed to func

    Returns:
        Concatenated results from all sessions, with same index as input.

    Examples:
        >>> import pandas as pd
        >>> s = pd.Series([1,2,3,4,5,6], index=pd.date_range('2024-01-01', periods=6, freq='H'))
        >>> # Apply rolling mean with window=2 to each session
        >>> result = apply_per_session_with_params(
        ...     s, lambda x, w: x.rolling(w).mean(), 2, threshold_hours=24
        ... )
    """
    breaks = detect_session_breaks(series.index, threshold_hours)
    if not breaks:
        return func(series, *func_args, **func_kwargs)

    results = []
    for i in range(len(breaks)):
        start_idx = breaks[i]
        end_idx = breaks[i + 1] if i + 1 < len(breaks) else len(series)

        if start_idx >= len(series):
            break

        session_data = series.iloc[start_idx:end_idx]
        session_result = func(session_data, *func_args, **func_kwargs)
        results.append(session_result)

    if results:
        return pd.concat(results)
    else:
        return func(series, *func_args, **func_kwargs)
